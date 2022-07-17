import torch
import torch.nn.functional as F


class CalculusOperators:
    device = "cuda"

    def construct_laplace_kernel(device="cuda"):
        out_channels = 2  # u and v
        in_channels = 1  # u or v
        kh, kw = 3, 3

        filter_x = torch.FloatTensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
        filter_y = torch.FloatTensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]])

        weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
        weight[0, 0, :, :] = filter_x
        weight[1, 0, :, :] = filter_y
        return weight.to(device)
    
    def construct_gradient_kernel(device="cuda"):
        out_channels = 2  # u and v
        in_channels = 1  # u or v
        kh, kw = 3, 3

        filter_x = torch.FloatTensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        filter_y = torch.FloatTensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]])

        weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
        weight[0, 0, :, :] = filter_x
        weight[1, 0, :, :] = filter_y
        return weight.to(device)
    
    def construct_forward_difference_kernels(device="cuda"):
        filter_x = torch.ones(1, 1, 3, 3, requires_grad=False)
        filter_y = torch.ones(1, 1, 3, 3, requires_grad=False)
        filter_x[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        filter_y[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
        return filter_x.to(device), filter_y.to(device)
    
    laplace_kernel = construct_laplace_kernel(device)
    gradient_kernel = construct_gradient_kernel(device)
    divergence_kernel_x, divergence_kernel_y = construct_forward_difference_kernels(device)
    
    @classmethod
    def laplace(cls, flow):
        u, v = torch.split(flow, split_size_or_sections=1, dim=1)
        laplace_u = F.conv2d(u, cls.laplace_kernel, padding=1)
        laplace_v = F.conv2d(v, cls.laplace_kernel, padding=1)
        return laplace_u, laplace_v
    
    @classmethod
    def gradient(cls, flow):
        u, v = torch.split(flow, split_size_or_sections=1, dim=1)
        grad_u = F.conv2d(u, cls.gradient_kernel, padding=1)
        grad_v = F.conv2d(v, cls.gradient_kernel, padding=1)
        return grad_u, grad_v
    
    @classmethod
    def divergence(cls, flow):
        u, v = torch.split(flow, split_size_or_sections=1, dim=1)
        grad_ux = F.conv2d(u, cls.divergence_kernel_x, padding=1)
        grad_vy = F.conv2d(v, cls.divergence_kernel_y, padding=1)
        return grad_ux + grad_vy

    @classmethod
    def curl(cls, flow):
        u, v = torch.split(flow, split_size_or_sections=1, dim=1)
        grad_uy = F.conv2d(u, cls.divergence_kernel_y, padding=1)
        grad_vx = F.conv2d(v, cls.divergence_kernel_x, padding=1)
        return grad_vx - grad_uy