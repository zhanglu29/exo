import os
import sys
import asyncio
from typing import Callable, TypeVar, Optional, Dict, Generic, Tuple, List
import socket
import random
import platform
import psutil
import uuid
import netifaces
from pathlib import Path
import tempfile
import inspect
import time

DEBUG = int(os.getenv("DEBUG", default="0"))
DEBUG_DISCOVERY = int(os.getenv("DEBUG_DISCOVERY", default="0"))
VERSION = "0.0.1"

exo_text = r"""
  _____  _____  
 / _ \ \/ / _ \ 
|  __/>  < (_) |
 \___/_/\_\___/ 
    """


def get_system_info():
  if psutil.MACOS:
    if platform.machine() == "arm64":
      return "Apple Silicon Mac"
    if platform.machine() in ["x86_64", "i386"]:
      return "Intel Mac"
    return "Unknown Mac architecture"
  if psutil.LINUX:
    return "Linux"
  return "Non-Mac, non-Linux system"


def find_available_port(host: str = "", min_port: int = 49152, max_port: int = 65535) -> int:
  used_ports_file = os.path.join(tempfile.gettempdir(), "exo_used_ports")

  def read_used_ports():
    if os.path.exists(used_ports_file):
      with open(used_ports_file, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]
    return []

  def write_used_port(port, used_ports):
    with open(used_ports_file, "w") as f:
      print(used_ports[-19:])
      for p in used_ports[-19:] + [port]:
        f.write(f"{p}\n")

  used_ports = read_used_ports()
  available_ports = set(range(min_port, max_port + 1)) - set(used_ports)

  while available_ports:
    port = random.choice(list(available_ports))
    if DEBUG >= 2: print(f"Trying to find available port {port=}")
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
      write_used_port(port, used_ports)
      return port
    except socket.error:
      available_ports.remove(port)

  raise RuntimeError("No available ports in the specified range")


def print_exo():
  print(exo_text)


def print_yellow_exo():
  yellow = "\033[93m"  # ANSI escape code for yellow
  reset = "\033[0m"  # ANSI escape code to reset color
  print(f"{yellow}{exo_text}{reset}")


def terminal_link(uri, label=None):
  if label is None:
    label = uri
  parameters = ""

  # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
  escape_mask = "\033]8;{};{}\033\\{}\033]8;;\033\\"

  return escape_mask.format(parameters, uri, label)


T = TypeVar("T")
K = TypeVar("K")


class AsyncCallback(Generic[T]):
  def __init__(self) -> None:
    self.condition: asyncio.Condition = asyncio.Condition()
    self.result: Optional[Tuple[T, ...]] = None
    self.observers: list[Callable[..., None]] = []

  async def wait(self, check_condition: Callable[..., bool], timeout: Optional[float] = None) -> Tuple[T, ...]:
    async with self.condition:
      await asyncio.wait_for(self.condition.wait_for(lambda: self.result is not None and check_condition(*self.result)), timeout)
      assert self.result is not None  # for type checking
      return self.result

  def on_next(self, callback: Callable[..., None]) -> None:
    self.observers.append(callback)

  def set(self, *args: T) -> None:
    self.result = args
    for observer in self.observers:
      observer(*args)
    asyncio.create_task(self.notify())

  async def notify(self) -> None:
    async with self.condition:
      self.condition.notify_all()


class AsyncCallbackSystem(Generic[K, T]):
  def __init__(self) -> None:
    self.callbacks: Dict[K, AsyncCallback[T]] = {}

  def register(self, name: K) -> AsyncCallback[T]:
    if name not in self.callbacks:
      self.callbacks[name] = AsyncCallback[T]()
    return self.callbacks[name]

  def deregister(self, name: K) -> None:
    if name in self.callbacks:
      del self.callbacks[name]

  def trigger(self, name: K, *args: T) -> None:
    if name in self.callbacks:
      self.callbacks[name].set(*args)

  def trigger_all(self, *args: T) -> None:
    for callback in self.callbacks.values():
      callback.set(*args)


K = TypeVar('K', bound=str)
V = TypeVar('V')


class PrefixDict(Generic[K, V]):
  def __init__(self):
    self.items: Dict[K, V] = {}

  def add(self, key: K, value: V) -> None:
    self.items[key] = value

  def find_prefix(self, argument: str) -> List[Tuple[K, V]]:
    return [(key, value) for key, value in self.items.items() if argument.startswith(key)]

  def find_longest_prefix(self, argument: str) -> Optional[Tuple[K, V]]:
    matches = self.find_prefix(argument)
    if len(matches) == 0:
      return None

    return max(matches, key=lambda x: len(x[0]))


def is_valid_uuid(val):
  try:
    uuid.UUID(str(val))
    return True
  except ValueError:
    return False


def get_or_create_node_id():
  NODE_ID_FILE = Path(tempfile.gettempdir())/".exo_node_id"
  try:
    if NODE_ID_FILE.is_file():
      with open(NODE_ID_FILE, "r") as f:
        stored_id = f.read().strip()
      if is_valid_uuid(stored_id):
        if DEBUG >= 2: print(f"Retrieved existing node ID: {stored_id}")
        return stored_id
      else:
        if DEBUG >= 2: print("Stored ID is not a valid UUID. Generating a new one.")

    new_id = str(uuid.uuid4())
    with open(NODE_ID_FILE, "w") as f:
      f.write(new_id)

    if DEBUG >= 2: print(f"Generated and stored new node ID: {new_id}")
    return new_id
  except IOError as e:
    if DEBUG >= 2: print(f"IO error creating node_id: {e}")
    return str(uuid.uuid4())
  except Exception as e:
    if DEBUG >= 2: print(f"Unexpected error creating node_id: {e}")
    return str(uuid.uuid4())


def pretty_print_bytes(size_in_bytes: int) -> str:
  if size_in_bytes < 1024:
    return f"{size_in_bytes} B"
  elif size_in_bytes < 1024**2:
    return f"{size_in_bytes / 1024:.2f} KB"
  elif size_in_bytes < 1024**3:
    return f"{size_in_bytes / (1024 ** 2):.2f} MB"
  elif size_in_bytes < 1024**4:
    return f"{size_in_bytes / (1024 ** 3):.2f} GB"
  else:
    return f"{size_in_bytes / (1024 ** 4):.2f} TB"


def pretty_print_bytes_per_second(bytes_per_second: int) -> str:
  if bytes_per_second < 1024:
    return f"{bytes_per_second} B/s"
  elif bytes_per_second < 1024**2:
    return f"{bytes_per_second / 1024:.2f} KB/s"
  elif bytes_per_second < 1024**3:
    return f"{bytes_per_second / (1024 ** 2):.2f} MB/s"
  elif bytes_per_second < 1024**4:
    return f"{bytes_per_second / (1024 ** 3):.2f} GB/s"
  else:
    return f"{bytes_per_second / (1024 ** 4):.2f} TB/s"


def get_all_ip_addresses():
  """获取本地机器的所有 IP 地址"""
  ip_addresses = set()  # 使用集合以避免重复

  # 添加 localhost
  ip_addresses.add("127.0.0.1")
  # 获取所有本地 IP 地址
  hostname = socket.gethostname()  # 获取本地主机名
  local_ip = socket.gethostbyname(hostname)  # 获取主机名对应的 IP 地址
  ip_addresses.add(local_ip)
  # 创建一个临时的 socket 连接
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # 连接到一个外部地址（不需要实际连接）
    ip_addresses.add(s.getsockname()[0])  # 添加当前的连接 IP 地址
    s.close()
  except Exception as e:
    print(f"Error while getting IP address: {e}")
  return list(ip_addresses)


async def shutdown(signal, loop, server):
  """Gracefully shutdown the server and close the asyncio loop."""
  print(f"Received exit signal {signal.name}...")
  print("Thank you for using exo.")
  print_yellow_exo()
  server_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
  [task.cancel() for task in server_tasks]
  print(f"Cancelling {len(server_tasks)} outstanding tasks")
  await asyncio.gather(*server_tasks, return_exceptions=True)
  await server.stop()


def is_frozen():
  return getattr(sys, 'frozen', False) or os.path.basename(sys.executable) == "exo" \
    or ('Contents/MacOS' in str(os.path.dirname(sys.executable))) \
    or '__nuitka__' in globals() or getattr(sys, '__compiled__', False)

# 定义颜色代码
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# 保存上次调用时间
_last_call_time = {}


def log_caller_info(func):
  if DEBUG != -3:
    def wrapper(*args, **kwargs):
      # 执行原始函数
      return func(*args, **kwargs)

    return wrapper

  def wrapper(*args, **kwargs):
    # 获取调用栈信息
    stack = inspect.stack()
    caller = stack[1]  # 调用此函数的直接调用者
    filename = caller.filename
    line_number = caller.lineno
    function_name = caller.function

    # 获取调用前时间
    start_time = time.time()

    # 获取前一次调用时间差
    func_key = f"{caller.filename}:{caller.lineno}:{caller.function}"
    last_call_time = _last_call_time.get(func_key, None)
    time_diff = (
      f"Time difference from last call: {start_time - last_call_time:.6f} seconds"
      if last_call_time is not None
      else "Time difference from last call: N/A"
    )
    _last_call_time[func_key] = start_time  # 更新调用时间

    # 打印调用前信息（日志）
    log_message_before = (
      f"[INFO] Function '{func.__name__}' was called by '{function_name}' in {filename}:{line_number}. "
      f"Call before time: {start_time}. {time_diff}"
    )
    print(log_message_before)

    # 执行原始函数
    result = func(*args, **kwargs)

    # 获取调用后时间
    end_time = time.time()

    # 计算时间差
    duration = end_time - start_time

    # 打印调用后信息（日志）
    log_message_after = (
      f"[INFO] Function '{func.__name__}' finished execution. "
      f"Call after time: {end_time}. Execution time: {duration:.6f} seconds."
    )
    print(log_message_after)

    return result

  return wrapper

def log_cost_info(func):
  def wrapper(*args, **kwargs):
    # 获取调用栈信息
    stack = inspect.stack()
    caller = stack[1]  # 调用此函数的直接调用者
    filename = caller.filename
    line_number = caller.lineno
    function_name = caller.function

    # 获取调用前时间
    start_time = time.time()

    # 获取前一次调用时间差
    func_key = f"{caller.filename}:{caller.lineno}:{caller.function}"
    last_call_time = _last_call_time.get(func_key, None)
    time_diff = (
      f"Time difference from last call: {start_time - last_call_time:.6f} seconds"
      if last_call_time is not None
      else "Time difference from last call: N/A"
    )
    _last_call_time[func_key] = start_time  # 更新调用时间

    # 计算参数字节大小
    args_size = sys.getsizeof(args)
    kwargs_size = sys.getsizeof(kwargs)
    total_size = args_size + kwargs_size

    # 打印调用前信息（日志）
    log_message_before = (
      f"[INFO] Function '{func.__name__}' was called by '{function_name}' in {filename}:{line_number}. "
      f"Call before time: {start_time}. {time_diff}. "
      f"Parameter size: {total_size} bytes."
    )
    print(log_message_before)

    # 执行原始函数
    result = func(*args, **kwargs)

    # 获取调用后时间
    end_time = time.time()

    # 计算时间差
    duration = end_time - start_time

    # 打印调用后信息（日志）
    log_message_after = (
      f"[INFO] Function '{func.__name__}' finished execution. "
      f"Call after time: {end_time}. Execution time: {duration:.6f} seconds."
    )
    print(log_message_after)

    return result

  return wrapper