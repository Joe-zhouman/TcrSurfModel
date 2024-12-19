import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple
from .hooked_obj import HookedObj


class LayerVisualization(HookedObj):
    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = False,
    ):
        super(LayerVisualization, self).__init__(model, device, verbose)

        self.layer_info_dict = self.get_conv_layer_info(self.model)
        if self.verbose:
            self._show_layer_info()
        self._handle = []
        self._target_layer_conv_output = None
        self._all_layer_outputs = OrderedDict()

    def _show_layer_info(self):
        for name, info in self.layer_info_dict.items():
            print(
                f"{name}: The index is {info[0]} and the number of filters is {info[1]}"
            )

    @staticmethod
    def get_conv_layer_info(model: nn.Module) -> OrderedDict[str, Tuple[int, int]]:
        layer_info_dict = OrderedDict()
        i = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_info_dict[name] = (i, getattr(module, "out_channels"), module)
                i += 1
        return layer_info_dict

    @HookedObj._hooked
    def visual_all_layers(self, batch):
        self._hook_all_layers()
        batch = [item.to(self.device).requires_grad_() for item in batch]
        _ = self.model(*batch[0:-1])
        self.release_hook()
        return self._all_layer_outputs

    def _hook_layers(self):
        def __create_hook(module_name):
            def __hook_func(module, grad_in, grad_out):
                if self.verbose:
                    print(f"{module_name} is hooked!")
                self._all_layer_outputs[module_name] = grad_out[0]

            return __hook_func

        for name in self.layer_info_dict:

            self._handle.append(
                self.layer_info_dict[name][2].register_forward_hook(__create_hook(name))
            )
