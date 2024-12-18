import os
import json
from safetensors import safe_open

# 配置模型路径
model_path = "/root/.cache/huggingface/hub/models--unsloth--Llama-3.2-3B-Instruct/snapshots/bc836a93eabc97432d7be9faedddf045ca7ad8fc"
output_file = os.path.join(model_path, "model.safetensors.index.json")

# 遍历 safetensors 文件
weight_map = {}
total_size = 0

for file_name in os.listdir(model_path):
    if file_name.endswith(".safetensors"):
        file_path = os.path.join(model_path, file_name)
        file_size = os.path.getsize(file_path)  # 计算文件实际大小（字节）
        total_size += file_size  # 累加总大小

        with safe_open(file_path, framework="pt") as f:
            for tensor_name in f.keys():
                weight_map[tensor_name] = file_name

# 写入 index.json 文件
index_data = {
    "weight_map": weight_map,
    "metadata": {"total_size": total_size},
}

with open(output_file, "w") as f:
    json.dump(index_data, f, indent=2)

print(f"Index file created at: {output_file}")
