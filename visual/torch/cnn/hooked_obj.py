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

    @classmethod
    def _hooked(self, func):
        def wrapper(self, *args, **kwargs):
            self._hook_layers()
            result = func(self, *args, **kwargs)
            self._release_hook()
            return result

        return wrapper

    @abstractmethod
    def _hook_layers(self):
        pass
