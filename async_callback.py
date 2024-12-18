from typing import Callable, Dict, Generic, TypeVar

K = TypeVar('K')
T = TypeVar('T')

class AsyncCallback(Generic[T]):
    def __init__(self) -> None:
        self.callbacks: List[Callable[..., None]] = []

    def add(self, callback: Callable[..., None]) -> None:
        self.callbacks.append(callback)

    def set(self, *args: T) -> None:
        for callback in self.callbacks:
            callback(*args)

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

# 使用示例
system = AsyncCallbackSystem[str, int]()

# 注册回调
callback = system.register('task_done')
callback.add(lambda x: print(f"Task completed with result: {x}"))
callback.add(lambda x: print(f"Task completed wit2h result: {x}"))

# 触发回调
system.trigger('task_done', 42)  # 输出: Task completed with result: 42

# 触发所有回调
system.trigger_all(100)  # 所有已注册的回调都会被触发
