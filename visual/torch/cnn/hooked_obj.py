from abc import ABCMeta, abstractmethod


class HookedObj(metaclass=ABCMeta):
    def __init__(self, model, device, verbose):

        self._handle = []
        self.model = model
        self.device = device
        self.verbose = verbose
        self.model.eval()
        self.model.to(self.device)

    def _release_hook(self):
        """
        释放所有挂钩的函数。

        遍历所有挂钩的句柄，并调用remove方法移除它们。
        如果设置了verbose标志，将打印释放所有挂钩的消息。
        """
        for handle in self._handle:
            handle.remove()
        if self.verbose:
            print("release all hooks")

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._release_hook()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred with block: {exc_type}. Message: {exc_value}")
            return True

    def __del__(self):
        self._release_hook()

    @classmethod
    def _hooked(self, func):
        """
        A decorator for hooking functions, used to wrap functions that need to intercept and process layers.

        Purpose:
        To ensure that the `_hook_layers` method is called before the execution of the wrapped function,
        and `_release_hook` is called after the execution, thereby managing the interception and release of layers.

        Parameters:
        - func: The function to be wrapped, which will be intercepted and processed.

        Returns:
        - wrapper: A wrapped function that executes the layer interception, the original function, and the release of interception in order.

        Usage:
        ```python
        @HookedObj._hooked
        def my_function(self, ...):
            pass
        ```
        """

        def wrapper(self, *args, **kwargs):
            self._hook_layers()
            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                print(e)
                self._release_hook()
            self._release_hook()
            return result

        return wrapper

    @abstractmethod
    def _hook_layers(self):
        pass
