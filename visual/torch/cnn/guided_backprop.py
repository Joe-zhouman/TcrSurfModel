import torch
import torch.nn as nn

from typing import List


class GuidedBackprop:
    """
    Produces gradients generated with guided back propagation from the given image
    """

    def __init__(
        self,
        model: nn.Module,
        relu_modules: List[nn.Module],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus(relu_modules)
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first conv layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                print(f"{name} is hooked!")
                module.register_full_backward_hook(hook_function)
                break

    def update_relus(self, relu_modules: List[nn.Module]):
        """
        Add hooks to all ReLU layers in the feature layers
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            If there is a positive gradient, change it to 1
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs.pop()
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(
                grad_in[0], min=0.0
            )
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for parent_module in relu_modules:
            for pos, module in parent_module.named_modules():
                if isinstance(module, nn.ReLU):
                    print(f"{pos} is hooked!")
                    module.register_full_backward_hook(relu_backward_hook_function)
                    module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, batch, loss_func):
        batch = [item.requires_grad_().to(self.device) for item in batch]
        #! IMPORTANT. The gradient will not be computed without requires_grad_()
        batch[0] = batch[0].requires_grad_()
        targets = batch[-1]
        outputs = self.model(*batch[0:-1])
        self.model.zero_grad()
        current_loss = loss_func(outputs, targets)
        print(current_loss)
        current_loss.backward()
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr


