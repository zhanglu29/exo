import grpc
import numpy as np
import inspect
from typing import Optional, Tuple, List
from . import node_service_pb2
from . import node_service_pb2_grpc
from ..peer_handle import PeerHandle
from exo.inference.shard import Shard
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.helpers import DEBUG
import sys
import time
import asyncio

# def log_execution_info(func):
#     """装饰器，用于记录函数执行时间、参数字节大小和调用关系"""
#     async def wrapper(*args, **kwargs):
#         # 计算参数字节大小
#         args_size = sum(sys.getsizeof(arg) for arg in args)
#         kwargs_size = sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in kwargs.items())
#         total_size = args_size + kwargs_size
#
#         # 获取调用前时间
#         start_time = time.time()
#
#         # 获取调用者信息
#         frame = inspect.currentframe().f_back
#         caller_function = frame.f_code.co_name
#         caller_line = frame.f_lineno
#         caller_file = frame.f_code.co_filename
#
#         # 执行原始函数
#         result = await func(*args, **kwargs)
#
#         # 获取调用后时间
#         end_time = time.time()
#
#         # 计算时间差
#         duration = end_time - start_time
#
#         # 打印日志
#         print(f"[INFO] STUB Function '{func.__name__}' called from {caller_function} "
#               f"({caller_file}:{caller_line}) executed in {duration:.6f} seconds. "
#               f"Parameter size: {total_size} bytes.")
#
#         return result
#
#     return wrapper


def get_object_size(obj):
    """递归计算对象的字节大小，包括多层引用和 NumPy 数组"""
    if isinstance(obj, (str, bytes)):
        return len(obj)
    elif isinstance(obj, (list, tuple, set)):
        return sum(get_object_size(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_object_size(key) + get_object_size(value) for key, value in obj.items())
    elif isinstance(obj, np.ndarray):
        return obj.nbytes  # NumPy 数组的实际字节数
    elif hasattr(obj, '__dict__'):  # 如果对象有属性字典
        return get_object_size(obj.__dict__)  # 递归计算所有属性的大小
    else:
        return sys.getsizeof(obj)

def log_execution_info(func):
    """装饰器，用于记录函数执行时间和参数字节大小"""
    async def wrapper(*args, **kwargs):
        # 计算参数字节大小
        args_size = sum(get_object_size(arg) for arg in args)
        kwargs_size = sum(get_object_size(key) + get_object_size(value) for key, value in kwargs.items())
        total_size = args_size + kwargs_size

        # 获取调用前时间
        start_time = time.time()

        # 执行原始函数
        result = await func(*args, **kwargs)

        # 获取调用后时间
        end_time = time.time()

        # 计算时间差
        duration = end_time - start_time

        # 打印日志
        print(f"[INFO] SERVER Function '{func.__name__}' executed in {duration:.6f} seconds. "
              f"Parameter size: {total_size} bytes.")

        return result

    return wrapper

class GRPCPeerHandle(PeerHandle):
    def __init__(self, _id: str, address: str, device_capabilities: DeviceCapabilities):
        self._id = _id
        self.address = address
        self._device_capabilities = device_capabilities
        self.channel = None
        self.stub = None

    def id(self) -> str:
        return self._id

    def addr(self) -> str:
        return self.address

    def device_capabilities(self) -> DeviceCapabilities:
        return self._device_capabilities

    async def connect(self):
        if self.channel is None:
            self.channel = grpc.aio.insecure_channel(self.address, options=[
                ("grpc.max_metadata_size", 32 * 1024 * 1024),
                ('grpc.max_receive_message_length', 32 * 1024 * 1024),
                ('grpc.max_send_message_length', 32 * 1024 * 1024)
            ])
            self.stub = node_service_pb2_grpc.NodeServiceStub(self.channel)
        await self.channel.channel_ready()

    async def is_connected(self) -> bool:
        return self.channel is not None and self.channel.get_state() == grpc.ChannelConnectivity.READY

    async def disconnect(self):
        if self.channel:
            await self.channel.close()
        self.channel = None
        self.stub = None

    async def _ensure_connected(self):
        if not await self.is_connected():
            await asyncio.wait_for(self.connect(), timeout=60)

    async def health_check(self) -> bool:
        try:
            await self._ensure_connected()
            request = node_service_pb2.HealthCheckRequest()
            response = await asyncio.wait_for(self.stub.HealthCheck(request), timeout=5)
            return response.is_healthy
        except asyncio.TimeoutError:
            return False
        except Exception:
            if DEBUG >= 4:
                print(f"Health check failed for {self._id}@{self.address}.")
                import traceback
                traceback.print_exc()
            return False

    @log_execution_info
    async def send_prompt(self, shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.ndarray]:
        request = node_service_pb2.PromptRequest(
            prompt=prompt,
            shard=node_service_pb2.Shard(
                model_id=shard.model_id,
                start_layer=shard.start_layer,
                end_layer=shard.end_layer,
                n_layers=shard.n_layers,
            ),
            request_id=request_id,
        )

        try:
            response = await self.stub.SendPrompt(request)

            if not response.tensor_data or not response.shape or not response.dtype:
                print(f"响应数据不完整: tensor_data={response.tensor_data}, shape={response.shape}, dtype={response.dtype}")
                return None

            result = np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)
            return result

        except grpc.aio.AioRpcError as rpc_error:
            print(f"RPC错误: {rpc_error}")
            print(f"状态码: {rpc_error.code()}")
            print(f"错误细节: {rpc_error.details()}")
            print(f"调试信息: {rpc_error.debug_error_string()}")

        except Exception as e:
            print(f"发送请求时发生其他错误: {e}")

        return None

    @log_execution_info
    async def send_tensor(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None) -> Optional[np.ndarray]:
        request = node_service_pb2.TensorRequest(
            shard=node_service_pb2.Shard(
                model_id=shard.model_id,
                start_layer=shard.start_layer,
                end_layer=shard.end_layer,
                n_layers=shard.n_layers,
            ),
            tensor=node_service_pb2.Tensor(tensor_data=tensor.tobytes(), shape=tensor.shape, dtype=str(tensor.dtype)),
            request_id=request_id,
        )
        response = await self.stub.SendTensor(request)

        if not response.tensor_data or not response.shape or not response.dtype:
            return None

        return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

    @log_execution_info
    async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
        request = node_service_pb2.GetInferenceResultRequest(request_id=request_id)
        response = await self.stub.GetInferenceResult(request)
        if response.tensor is None:
            return None, response.is_finished
        return (
            np.frombuffer(response.tensor.tensor_data, dtype=np.dtype(response.tensor.dtype)).reshape(response.tensor.shape),
            response.is_finished,
        )

    async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
        request = node_service_pb2.CollectTopologyRequest(visited=visited, max_depth=max_depth)
        response = await self.stub.CollectTopology(request)
        topology = Topology()
        for node_id, capabilities in response.nodes.items():
            device_capabilities = DeviceCapabilities(
                model=capabilities.model, chip=capabilities.chip, memory=capabilities.memory,
                flops=DeviceFlops(fp16=capabilities.flops.fp16, fp32=capabilities.flops.fp32, int8=capabilities.flops.int8)
            )
            topology.update_node(node_id, device_capabilities)
        for node_id, peers in response.peer_graph.items():
            for peer_id in peers.peer_ids:
                topology.add_edge(node_id, peer_id)
        return topology

    @log_execution_info
    async def send_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
        request = node_service_pb2.SendResultRequest(request_id=request_id, result=result, is_finished=is_finished)
        await self.stub.SendResult(request)

    async def send_opaque_status(self, request_id: str, status: str) -> None:
        request = node_service_pb2.SendOpaqueStatusRequest(request_id=request_id, status=status)
        await self.stub.SendOpaqueStatus(request)