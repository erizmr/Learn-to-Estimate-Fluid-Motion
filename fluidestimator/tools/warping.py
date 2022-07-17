import torch 

class FieldWarper:
    backward_warp_tensor_grid = {}

    @classmethod
    def backward_warp(cls, tensorInput, tensorFlow, device="cuda"):
        if str(tensorFlow.size()) not in cls.backward_warp_tensor_grid:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1,
                                                    tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1,
                                                    tensorFlow.size(3))

            cls.backward_warp_tensor_grid[str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).to(device)
        # end
        tensorFlow = torch.cat([
            tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
        ], 1)
        return torch.nn.functional.grid_sample(
            input=tensorInput,
            grid=(cls.backward_warp_tensor_grid[str(tensorFlow.size())] +
                tensorFlow).permute(0, 2, 3, 1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)