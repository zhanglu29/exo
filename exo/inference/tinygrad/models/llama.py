from typing import Tuple, Union, Optional, Dict, Any, List
from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv
from collections import OrderedDict
#
#
# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=dtypes.half,
#                          rope_scaling: Optional[Dict[str, float]] = None) -> Tensor:
#     print("【DEBUG 1】Entering precompute_freqs_cis")
#     freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
#
#     if rope_scaling:
#         print("【DEBUG 2】Applying rope scaling")
#         factor = rope_scaling.get('factor', 1.0)
#         low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
#         high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
#         original_max_pos_emb = rope_scaling.get('original_max_position_embeddings', end)
#
#         # 打印低频部分的具体值
#         print(f"【DEBUG】Before applying low_freq_factor: {freqs[:dim // 4]}, shape: {freqs[:dim // 4].shape}")
#         print(f"【DEBUG】low_freq_factor: {low_freq_factor}")
#         # 打印为 NumPy 数组
#         print("freqs Tensor 数据:", freqs.numpy())
#         freqs[:dim // 4] *= low_freq_factor
#
#
#         freqs[dim // 4:] = freqs[dim // 4:].contiguous() * high_freq_factor
#         freqs *= (original_max_pos_emb / end) ** (1.0 / factor)
#
#     freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
#     print("【DEBUG 3】Exiting precompute_freqs_cis")
#     return Tensor.stack(freqs.cos().cast(dtype), freqs.sin().cast(dtype), dim=-1).reshape(1, end, 1, dim // 2, 2)
# # from tinygrad import Tensor
# # from typing import Optional, Dict

from tinygrad import Tensor
from typing import Optional, Dict

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=None,
                         rope_scaling: Optional[Dict[str, float]] = None) -> Tensor:
    print("【DEBUG 1】Entering precompute_freqs_cis")

    try:
        freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
        print(f"【DEBUG】Initial freqs: {freqs}, shape: {freqs.shape}")

        # print(f"【DEBUG】freqs 前五值: {freqs[:5]}")

        # print(f"【DEBUG】freqs具体值: {freqs.tolist()}")
        # 打印 freqs 的具体值
        # print(f"【DEBUG】freqs具体值: {freqs.tolist() if hasattr(freqs, 'numpy') else freqs}")

        if rope_scaling:
            print("【DEBUG 2】Applying rope scaling")
            factor = rope_scaling.get('factor', 1.0)
            low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
            high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
            original_max_pos_emb = rope_scaling.get('original_max_position_embeddings', end)

            # print("【DEBUG】freqs before scaling: ", freqs.numpy() if hasattr(freqs, 'numpy') else freqs)

            # freqs[:dim // 4] *= low_freq_factor
            #
            print("【DEBUG 11】Applying rope scaling")
            lower_freqs = freqs[:dim // 4] * low_freq_factor
            print("【DEBUG 12】Applying rope scaling")
            higher_freqs = freqs[dim // 4:].contiguous() * high_freq_factor
            print("【DEBUG 13】Applying rope scaling")
            # 使用cat或concatenate合并
            freqs = Tensor.cat([lower_freqs, higher_freqs])
            print("【DEBUG 14】Applying rope scaling")
            print("【DEBUG】freqs after low_freq scaling: ", freqs.numpy() if hasattr(freqs, 'numpy') else freqs)

            # freqs[dim // 4:] = freqs[dim // 4:].contiguous() * high_freq_factor
            # print("【DEBUG】freqs after high_freq scaling: ", freqs.numpy() if hasattr(freqs, 'numpy') else freqs)

            freqs *= (original_max_pos_emb / end) ** (1.0 / factor)
            print("【DEBUG 15】Applying rope scaling")
            print("【DEBUG】freqs after original max position scaling: ", freqs.numpy() if hasattr(freqs, 'numpy') else freqs)

        try:
            # 创建范围张量
            arange_tensor = Tensor.arange(end)
            print(f"【DEBUG】arange_tensor: {arange_tensor.numpy() if hasattr(arange_tensor, 'numpy') else arange_tensor}, shape: {arange_tensor.shape}")

            # 扩展维度
            unsqueezed_arange = arange_tensor.unsqueeze(dim=1)
            print(f"【DEBUG】unsqueezed_arange: {unsqueezed_arange.numpy() if hasattr(unsqueezed_arange, 'numpy') else unsqueezed_arange}, shape: {unsqueezed_arange.shape}")

            # 扩展 freq 维度
            unsqueezed_freqs = freqs.unsqueeze(dim=0)
            print(f"【DEBUG】unsqueezed_freqs: {unsqueezed_freqs.numpy() if hasattr(unsqueezed_freqs, 'numpy') else unsqueezed_freqs}, shape: {unsqueezed_freqs.shape}")

            # 执行乘法
            freqs = unsqueezed_arange * unsqueezed_freqs
            print(f"【DEBUG】freqs after multiplication: {freqs.numpy() if hasattr(freqs, 'numpy') else freqs}, shape: {freqs.shape}")
            print(f"【DEBUG】freqs具体值: {freqs.numpy() if hasattr(freqs, 'numpy') else freqs}")

        except Exception as e:
            print(f"【ERROR】Exception in computing freqs: {e}")
            print(f"【DEBUG】Variables at error: dim: {dim}, end: {end}, freqs shape: {freqs.shape if 'freqs' in locals() else 'not defined'}")
            raise

        print("【DEBUG 3】Exiting precompute_freqs_cis")
        return Tensor.stack(freqs.cos().cast(dtype), freqs.sin().cast(dtype), dim=-1).reshape(1, end, 1, dim // 2, 2)

    except Exception as e:
        print(f"【ERROR】Exception in precompute_freqs_cis: {e}")
        raise

def complex_mult(A, c, d):
    print("【DEBUG 1】Entering complex_mult")
    a, b = A[..., 0:1], A[..., 1:2]
    ro = a * c - b * d
    co = a * d + b * c
    print("【DEBUG 2】Exiting complex_mult")
    return ro.cat(co, dim=-1)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    print("【DEBUG 1】Entering apply_rotary_emb")
    assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[
        1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    xq = xq.reshape(*xq.shape[0:-1], -1, 2)
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)
    assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
    c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
    xq_out = complex_mult(xq, c, d)
    xk_out = complex_mult(xk, c, d)
    print("【DEBUG 2】Exiting apply_rotary_emb")
    return xq_out.flatten(3), xk_out.flatten(3)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    print("【DEBUG 1】Entering repeat_kv")
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        print("【DEBUG 2】n_rep is 1, returning x")
        return x
    result = x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
    print("【DEBUG 3】Exiting repeat_kv")
    return result


class Attention:
    def __init__(self, dim, n_heads, n_kv_heads, max_context, linear=nn.Linear):
        print("【DEBUG 1】Initializing Attention")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.max_context = max_context

        self.wq = linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = linear(self.n_heads * self.head_dim, dim, bias=False)
        print("【DEBUG 2】Attention initialized")

    def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor],
                 cache: Optional[Tensor] = None) -> Tensor:
        print("【DEBUG 1】Entering Attention __call__")
        if getenv("WQKV"):
            if not hasattr(self, 'wqkv'):
                self.wqkv = Tensor.cat(self.wq.weight, self.wk.weight, self.wv.weight)
            xqkv = x @ self.wqkv.T
            xq, xk, xv = xqkv.split([self.wq.weight.shape[0], self.wk.weight.shape[0], self.wv.weight.shape[0]], dim=2)
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
        xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
        xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        bsz, seqlen, _, _ = xq.shape

        if cache is not None:
            print("【DEBUG 2】Updating cache")
            assert xk.dtype == xv.dtype == cache.dtype, f"{xk.dtype=}, {xv.dtype=}, {cache.dtype=}"
            cache.shrink((None, None, (start_pos, start_pos + seqlen), None, None)).assign(
                Tensor.stack(xk, xv)).realize()

            keys = cache[0].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xk
            values = cache[1].shrink((None, (0, start_pos + seqlen), None, None)) if start_pos > 0 else xv
        else:
            keys = xk
            values = xv

        keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
        xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
        attn = attn.reshape(bsz, seqlen, -1)
        print("【DEBUG 3】Exiting Attention __call__")
        return self.wo(attn)


class FeedForward:
    def __init__(self, dim: int, hidden_dim: int, linear=nn.Linear):
        print("【DEBUG 1】Initializing FeedForward")
        self.w1 = linear(dim, hidden_dim, bias=False)
        self.w2 = linear(hidden_dim, dim, bias=False)
        self.w3 = linear(dim, hidden_dim, bias=False)  # the gate in Gated Linear Unit
        print("【DEBUG 2】FeedForward initialized")

    def __call__(self, x: Tensor) -> Tensor:
        print("【DEBUG 1】Entering FeedForward __call__")
        result = self.w2(self.w1(x).silu() * self.w3(x))  # SwiGLU
        print("【DEBUG 2】Exiting FeedForward __call__")
        return result


class TransformerBlock:
    def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, norm_eps: float, max_context: int,
                 linear=nn.Linear, feed_forward=FeedForward):
        print("【DEBUG 1】Initializing TransformerBlock")
        self.attention = Attention(dim, n_heads, n_kv_heads, max_context, linear)
        self.feed_forward = feed_forward(dim, hidden_dim, linear)
        self.attention_norm = nn.RMSNorm(dim, norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, norm_eps)
        print("【DEBUG 2】TransformerBlock initialized")

    def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor],
                 cache: Optional[Tensor] = None):
        print("【DEBUG 1】Entering TransformerBlock __call__")
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, cache=cache)
        result = (h + self.feed_forward(self.ffn_norm(h))).contiguous()
        print("【DEBUG 2】Exiting TransformerBlock __call__")
        return result


def sample_logits(logits: Tensor, temp: float, k: int, p: float, af: float, ap: float):
    print("【DEBUG 1】Entering sample_logits")
    assert logits.ndim == 1, "only works on 1d tensors"
    assert 0 <= p <= 1, "p must be between 0 and 1"
    assert 0 <= k <= logits.numel(), "k must be between 0 and numel"

    if temp < 1e-6:
        print("【DEBUG 2】Temperature is very low, using argmax")
        return logits.argmax().reshape(1)

    if af or ap:
        if not hasattr(sample, "alpha_counter"):
            setattr(sample, "alpha_counter", Tensor.zeros_like(logits, dtype=dtypes.int32).contiguous())
        logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

    logits = (logits != logits).where(-float("inf"), logits)

    t = (logits / temp).softmax()

    counter, counter2 = Tensor.arange(t.numel(), device=logits.device).contiguous(), Tensor.arange(t.numel() - 1, -1,
                                                                                                   -1,
                                                                                                   device=logits.device).contiguous()
    if k:
        output, output_indices = Tensor.zeros(k, device=logits.device).contiguous(), Tensor.zeros(k,
                                                                                                  device=logits.device,
                                                                                                  dtype=dtypes.int32).contiguous()
        for i in range(k):
            t_argmax = (t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1).cast(dtypes.default_int)
            output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
            output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
            t = (counter == t_argmax).where(0, t)

        output_cumsum = output[::-1]._cumsum()[::-1] + t.sum()
        output = (output_cumsum >= (1 - p)) * output
        output_indices = (output_cumsum >= (1 - p)) * output_indices

        output_idx = output.multinomial()
        output_token = output_indices[output_idx]
    else:
        output_token = t.multinomial()

    if af or ap:
        sample.alpha_counter = (counter == output_token).where(sample.alpha_counter + 1, sample.alpha_counter)

    print("【DEBUG 2】Exiting sample_logits")
    return output_token


from exo.inference.shard import Shard


class Transformer:
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            n_heads: int,
            n_layers: int,
            norm_eps: float,
            vocab_size,
            shard: Shard = None,
            linear=nn.Linear,
            n_kv_heads=None,
            rope_theta=10000,
            max_context=1024,
            jit=True,
            feed_forward=FeedForward,
            rope_scaling: Optional[Dict[str, float]] = None,
            tie_word_embeddings=False,
    ):
        print("【DEBUG 1】Initializing Transformer")
        self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context, linear,
                                        feed_forward=feed_forward) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(dim, norm_eps)
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        if tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight
        self.max_context = max_context
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, self.max_context * 2, rope_theta,
                                              rope_scaling=rope_scaling).contiguous()
        self.forward_jit = TinyJit(self.forward_base) if jit else None
        self.shard = shard
        print("【DEBUG 2】Transformer initialized")

    def forward_base(self, x: Tensor, start_pos: Union[Variable, int], cache: Optional[List[Tensor]] = None):
        print("【DEBUG 1】Entering forward_base")
        seqlen = x.shape[1]
        freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos + seqlen), None, None, None))
        mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-100000000"), dtype=x.dtype,
                           device=x.device).triu(start_pos + 1).realize() if seqlen > 1 else None

        h = x

        if cache is None:
            cache = [None for _ in range(self.shard.start_layer, self.shard.end_layer + 1)]
        for i, c in zip(range(self.shard.start_layer, self.shard.end_layer + 1), cache):
            layer = self.layers[i]
            h = layer(h, start_pos, freqs_cis, mask, cache=c)

        if self.shard.is_last_layer():
            logits = self.output(self.norm(h)).float().realize()
            print("【DEBUG 2】Exiting forward_base with logits")
            return logits
        else:
            print("【DEBUG 3】Exiting forward_base")
            return h

    def embed(self, inputs: Tensor):
        print("【DEBUG 1】Entering embed")
        if self.shard.is_first_layer():
            h = self.tok_embeddings(inputs)
        else:
            h = inputs
        print("【DEBUG 2】Exiting embed")
        return h

    def forward(self, x: Tensor, start_pos: int, cache: Optional[List[Tensor]] = None):
        print("【DEBUG 1】Entering forward")
        if x.shape[0:2] == (1, 1) and self.forward_jit is not None and start_pos != 0:
            return self.forward_jit(x, Variable("start_pos", 1, self.max_context).bind(start_pos), cache=cache)
        return self.forward_base(x, start_pos, cache=cache)

    def __call__(self, tokens: Tensor, start_pos: Variable, cache: Optional[List[Tensor]] = None):
        print("【DEBUG 1】Entering Transformer __call__")
        h = self.embed(tokens)
        print("【DEBUG 2】Exiting Transformer __call__")
        return self.forward(h, start_pos, cache=cache)


# *** helpers ***

def convert_from_huggingface(weights: Dict[str, Tensor], model: Transformer, n_heads: int, n_kv_heads: int):
    print("【DEBUG 1】Entering convert_from_huggingface")

    def permute(v: Tensor, n_heads: int):
        return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1]).transpose(1, 2).reshape(*v.shape[:2])

    keymap = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in
           range(len(model.layers))},
        **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in
           ["q", "k", "v", "o"] for l in range(len(model.layers))},
        **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in
           range(len(model.layers))},
        **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight" for x, y in
           {"gate": "1", "down": "2", "up": "3"}.items() for l in range(len(model.layers))},
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    sd = {}
    for k, v in weights.items():
        if ".rotary_emb." in k:
            continue
        v = v.to(Device.DEFAULT)
        if "model.layers" in k:
            if "q_proj" in k:
                v = permute(v, n_heads)
            elif "k_proj" in k:
                v = permute(v, n_kv_heads)
        if k in keymap:
            sd[keymap[k]] = v
        else:
            sd[k] = v

    print("【DEBUG 2】Exiting convert_from_huggingface")
    return sd


def fix_bf16(weights: Dict[Any, Tensor]):
    print("【DEBUG 1】Entering fix_bf16")
    if getenv("SUPPORT_BF16", 1):
        return {k: v.cast(dtypes.float16) if v.dtype == dtypes.bfloat16 else v for k, v in weights.items()}
    return {k: v.llvm_bf16_cast(dtypes.half).to(v.device) if v.dtype == dtypes.bfloat16 else v for k, v in
            weights.items()}