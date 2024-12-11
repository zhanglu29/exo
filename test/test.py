import sys
import time
import asyncio

def get_object_size(obj):
    """递归计算对象的字节大小，包括多层引用"""
    if isinstance(obj, (str, bytes)):
        return len(obj)
    elif isinstance(obj, (list, tuple, set)):
        return sum(get_object_size(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_object_size(key) + get_object_size(value) for key, value in obj.items())
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

# 示例异步函数
@log_execution_info
async def example_function(data, count=1):
    """示例异步函数，用于测试装饰器"""
    await asyncio.sleep(1)  # 模拟一些异步操作
    return {"data": data, "count": count}

# 测试代码
async def test():
    # 测试用例
    data = {"key1": [1, 2, 3, 4], "key2": "value", "key3": {"nested_key": "nested_value"}}
    response = await example_function(data, count=5)
    print("Response:", response)

# 运行测试
if __name__ == "__main__":
    asyncio.run(test())