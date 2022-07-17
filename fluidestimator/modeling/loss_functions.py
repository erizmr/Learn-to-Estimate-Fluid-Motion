import torch
import logging
import numpy as np
import torch.nn.functional as F
from fluidestimator.tools import FieldWarper

device = 'cuda'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

loss_registry = {}

# Backward_tensorGrid = {}


# def Backward(tensorInput, tensorFlow):
#     if str(tensorFlow.size()) not in Backward_tensorGrid:
#         tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
#             1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1,
#                                                 tensorFlow.size(2), -1)
#         tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
#             1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1,
#                                                 tensorFlow.size(3))

#         Backward_tensorGrid[str(tensorFlow.size())] = torch.cat(
#             [tensorHorizontal, tensorVertical], 1).to(device)
#     # end
#     tensorFlow = torch.cat([
#         tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
#         tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
#     ], 1)
#     return torch.nn.functional.grid_sample(
#         input=tensorInput,
#         grid=(Backward_tensorGrid[str(tensorFlow.size())] +
#               tensorFlow).permute(0, 2, 3, 1),
#         mode='bilinear',
#         padding_mode='zeros',
#         align_corners=True)


def error_rate(u, v, u_t, v_t):
    """Error rate is defined as
      if the error is larger than 3 pixels and
      5% of the true vector
      then the estimation is wrong
    """
    v_pre = np.sqrt(u**2 + v**2)
    v_label = np.sqrt(u_t**2 + v_t**2)
    norm_2 = np.sqrt((u-u_t)**2+(v-v_t)**2)
    cond_1 = np.where(norm_2 > 0.25, True, False)
    cond_2 = np.where((norm_2 / v_label) > 0.05, True, False)
    er = np.sum(
        cond_1 * cond_2
        ) / np.prod(v_pre.shape)
    return er


def epe_loss(f, f_gt):
    return np.sqrt(np.mean((np.array(f)-np.array(f_gt))**2))


def EPE(input_flow, target_flow, mean=True):
    # Calculate the EPE along the second dimension
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size


def real_EPE(output, target):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h, w),
                                     mode='bilinear',
                                     align_corners=False)
    return EPE(upsampled_output, target, mean=True)


def construct_sobel_kernels():
    """

    Returns: sobel kernels for both x and y direction grads

    """
    sobel_x = torch.randn(1, 1, 3, 3, requires_grad=False)
    sobel_y = torch.randn(1, 1, 3, 3, requires_grad=False)
    sobel_x[0, 0, :, :] = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y[0, 0, :, :] = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sobel_x, sobel_y


def construct_divergence_kernels():
    """
    Construct the x and y grad kernel seperately, typically for computing divergence use
    Returns: 1st order kernels for both x and y direction grads

    """
    filter_x = torch.randn(1, 1, 3, 3, requires_grad=False)
    filter_y = torch.randn(1, 1, 3, 3, requires_grad=False)
    filter_x[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    filter_y[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
    return filter_x, filter_y


def construct_gradient_kernels():
    """
    Construct a combined the x and y grad kernel, typically for computing gradient use
    Returns:

    """
    filter_xy = torch.randn(2, 1, 3, 3, requires_grad=False)
    filter_xy[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    filter_xy[1, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
    return filter_xy


def multiscale_supervised_error(tensorFlowForward,
                                  gt,
                                  weights=None,
                                  cfg=None):
    h, w = tensorFlowForward[-1].shape[-2], tensorFlowForward[-1].shape[-1]

    def average_epe(f, g):
        return torch.mean(torch.sqrt(torch.sum((f - g)**2, dim=1)), dim=(0, 1, 2))

    if weights is None:
        weights = [12.7, 5.5, 4.35, 3.9, 3.4, 1.1]
    if len(cfg.MODEL.PREDICTOR.FLOWNET_DEFAULT_WEIGHTS) > 0:
        weights = cfg.MODEL.PREDICTOR.FLOWNET_DEFAULT_WEIGHTS

    assert (len(weights) == len(tensorFlowForward))

    loss_dict = {}
    for i, weight in enumerate(weights):
        scale_factor = 1.0 / (2 ** i)

        flow = tensorFlowForward[-1 - i]
        losses = average_epe(
                           flow * scale_factor,
                           gt * scale_factor
                            )

        loss_dict['{}_LOSS'.format(i)] = weight * losses

        h = h // 2
        w = w // 2

        gt = F.interpolate(gt, (h, w), mode='bilinear', align_corners=False)

    return loss_dict


def multiscale_unsupervised_error(tensorFlowForward,
                                  tensorFlowBackward,
                                  pressure,
                                  vorticity,
                                  tensorFirst,
                                  tensorSecond,
                                  weights=None,
                                  cfg=None):

    h, w = tensorFirst.shape[-2], tensorFirst.shape[-1]

    if not loss_registry:
        # Collect all losses
        for k, v in cfg.TRAIN.LOSS.items():
            if v > 0.0:
                loss_registry[k] = (loss_collection[k], v)
        _logger = logging.getLogger(__name__)
        _logger.info("loss functions used: {}".format(loss_registry))
        print("loss functions used: {}".format(loss_registry))

    def one_scale(tensorFirst,
                  tensorSecond,
                  tensorFlowForward,
                  tensorFlowBackward,
                  pressure,
                  vorticity,
                  loss_registry,
                  layer):
        local_losses = {}
        for k, f in loss_registry.items():
            # f[0]: loss function, f[1]:corresponding weight
            func = f[0]
            if k == 'PHOTOMETRIC':
                local_losses[k] = (func(tensorFirst,
                                        tensorSecond,
                                        tensorFlowForward,
                                        tensorFlowBackward),
                                   f[1])
            elif k == 'PRESSURE_DIV':
                if layer == 0:
                    local_losses[k] = (func(tensorFlowForward, pressure), f[1])
            elif k == 'TEMPORAL_VORTICITY':
                if layer == 0:
                    local_losses[k] = (func(tensorFlowForward, vorticity), f[1])
            else:
                local_losses[k] = (func(tensorFlowForward, tensorFlowBackward), f[1])
        return local_losses

    if weights is None:
        weights = [12.7, 5.5, 4.35, 3.9, 3.4, 1.1]
    if len(cfg.MODEL.PREDICTOR.FLOWNET_DEFAULT_WEIGHTS) > 0:
        weights = cfg.MODEL.PREDICTOR.FLOWNET_DEFAULT_WEIGHTS

    assert (len(weights) == len(tensorFlowForward))
    assert (len(weights) == len(tensorFlowBackward))

    loss_dict = {}
    for i, weight in enumerate(weights):

        scale_factor = 1.0 / (2**i)
        losses = one_scale(tensorFirst,
                           tensorSecond,
                           tensorFlowForward[-1 - i] * scale_factor,
                           tensorFlowBackward[-1 - i] * scale_factor,
                           pressure,
                           vorticity,
                           loss_registry,
                           i)
        for k, v in losses.items():
            loss_dict['{}_LOSS_{}'.format(i, k)] = weight * v[0] * v[1]

        h = h // 2
        w = w // 2
        tensorFirst = F.interpolate(tensorFirst, (h, w),
                                    mode='bilinear',
                                    align_corners=False)
        tensorSecond = F.interpolate(tensorSecond, (h, w),
                                     mode='bilinear',
                                     align_corners=False)

    return loss_dict


def charbonnier_loss(x, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss for x
    Args:
        x(tesnor): [batch, channels, height, width]
    Returns:
        loss
    """
    batch, channels, height, width = x.shape
    normalization = torch.tensor(batch * height * width * channels,
                                 requires_grad=False)

    error = torch.pow(
        (x * torch.tensor(beta)).pow(2) + torch.tensor(epsilon).pow(2), alpha)

    return torch.sum(error) / normalization


def warp_loss(img_first, img_second, flow):
    """Differentiable Charbonnier penalty function"""
    difference = img_first - FieldWarper.backward_warp(tensorInput=img_second,
                                              tensorFlow=flow)
    return charbonnier_loss(difference, beta=255.0)
    # return charbonnier_loss(difference, beta=10.0)


def bidirectional_warp_loss(img_first, img_second, flow_forward,
                            flow_backward):
    """Compute bidirectional photometric loss"""
    return warp_loss(img_first, img_second, flow_forward) + warp_loss(
        img_second, img_first, flow_backward)


def consistency_loss(flow_forward, flow_backward):
    """ Differentiable Charbonnier penalty function"""
    difference = flow_forward + FieldWarper.backward_warp(
        tensorInput=flow_backward, tensorFlow=flow_forward)
    return charbonnier_loss(difference)


def bidirectional_consistency_loss(flow_forward, flow_backward):
    return consistency_loss(flow_forward,
                            flow_backward) + consistency_loss(
                               flow_backward, flow_forward)


def _smoothness_deltas(flow):
    """1st order smoothness, compute smoothness loss components"""
    out_channels = 2  # u and v
    in_channels = 1  # u or v
    kh, kw = 3, 3

    filter_x = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    filter_y = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])

    weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
    weight[0, 0, :, :] = filter_x
    weight[1, 0, :, :] = filter_y

    uFlow, vFlow = torch.split(flow, split_size_or_sections=1, dim=1)

    delta_u = F.conv2d(uFlow, weight.to(device))
    delta_v = F.conv2d(vFlow, weight.to(device))
    return delta_u, delta_v


def smoothness_loss(flow):
    """Compute 1st order smoothness loss"""
    delta_u, delta_v = _smoothness_deltas(flow)
    return charbonnier_loss(delta_u) + charbonnier_loss(delta_v)


def bi_smoothness_loss(tensorFlowForward, tensorFlowBackward):
    """Compute bidirectional 1st order smoothness loss"""
    return smoothness_loss(tensorFlowForward) + smoothness_loss(
        tensorFlowBackward)


# 2nd order smoothness loss
def _second_order_deltas(flow):
    """2nd order smoothness, compute smoothness loss components"""
    out_channels = 4
    in_channels = 1
    kh, kw = 3, 3

    filter_x = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
    filter_y = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
    filter_diag1 = [[1, 0, 0], [0, -2, 0], [0, 0, 1]]
    filter_diag2 = [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
    weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
    weight[0, 0, :, :] = torch.FloatTensor(filter_x)
    weight[1, 0, :, :] = torch.FloatTensor(filter_y)
    weight[2, 0, :, :] = torch.FloatTensor(filter_diag1)
    weight[3, 0, :, :] = torch.FloatTensor(filter_diag2)

    uFlow, vFlow = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(uFlow, weight.to(device))
    delta_v = F.conv2d(vFlow, weight.to(device))
    return delta_u, delta_v


def second_smoothness_loss(flow):
    """Compute 2nd order smoothness loss"""
    delta_u, delta_v = _second_order_deltas(flow)
    return charbonnier_loss(delta_u) + charbonnier_loss(delta_v)


def bi_second_smoothness_loss(flow_forward, flow_backward):
    """Compute bidirectional 2nd order smoothness loss"""
    return second_smoothness_loss(flow_forward) + second_smoothness_loss(
        flow_backward)


def divergence_operator(flow):
    """
    Compute the divergence loss
    """
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    filter_x, filter_y = construct_divergence_kernels()
    grad_u = F.conv2d(flow_u, filter_x.to(device), padding=1)
    grad_v = F.conv2d(flow_v, filter_y.to(device), padding=1)
    return grad_u + grad_v


def divergence_loss(flow):
    """
    Compute the divergence loss
    """
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    filter_x, filter_y = construct_divergence_kernels()
    grad_u = F.conv2d(flow_u, filter_x.to(device), padding=1)
    grad_v = F.conv2d(flow_v, filter_y.to(device), padding=1)
    return charbonnier_loss(grad_u + grad_v)


def divergence_pressure_loss(flow, pressure):
    return charbonnier_loss(divergence_operator(flow) * pressure)


def _second_divergence_loss(flow):
    """
    Compute the second order divergence loss
    """
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    filter_x, filter_y = construct_divergence_kernels()
    grad_u = F.conv2d(flow_u, filter_x.to(device), padding=1)
    grad_v = F.conv2d(flow_v, filter_y.to(device), padding=1)
    div_uv = grad_u + grad_v
    grad_div_x = F.conv2d(div_uv, filter_x.to(device), padding=1)
    grad_div_y = F.conv2d(div_uv, filter_y.to(device), padding=1)
    return grad_div_x, grad_div_y


def second_divergence_loss(flow):
    grad_div_x, grad_div_y = _second_divergence_loss(flow)
    return charbonnier_loss(grad_div_x) + charbonnier_loss(grad_div_y)


def bi_divergence_loss(flow_forward, flow_backward):
    """compute the bidirectional first order divergence loss"""
    return divergence_loss(flow_forward)+divergence_loss(flow_backward)


def bi_second_divergence_loss(flow_forward, flow_backward):
    """compute the bidirectional second order divergence loss"""
    return second_divergence_loss(flow_forward)+second_divergence_loss(flow_backward)


def curl_operator(flow):
    """
    Compute the curl of 2D flow field
    """
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    filter_x, filter_y = construct_divergence_kernels()
    grad_u = F.conv2d(flow_u, filter_y.to(device), padding=1)
    grad_v = F.conv2d(flow_v, filter_x.to(device), padding=1)
    return grad_v - grad_u


def temporal_vorticity_loss(flow, vorticity_hat):
    return charbonnier_loss(curl_operator(flow) - vorticity_hat)


def curl_loss(flow):
    """
    Compute the first order curl loss
    """
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    filter_x, filter_y = construct_divergence_kernels()
    grad_u = F.conv2d(flow_u, filter_y.to(device), padding=1)
    grad_v = F.conv2d(flow_v, filter_x.to(device), padding=1)
    return charbonnier_loss(grad_v - grad_u)


def _second_curl_loss(flow):
    """
    Compute the second order curl loss
    """
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    filter_x, filter_y = construct_divergence_kernels()
    grad_u = F.conv2d(flow_u, filter_y.to(device), padding=1)
    grad_v = F.conv2d(flow_v, filter_x.to(device), padding=1)
    curl_uv = grad_v - grad_u
    grad_div_x = F.conv2d(curl_uv, filter_x.to(device), padding=1)
    grad_div_y = F.conv2d(curl_uv, filter_y.to(device), padding=1)
    return grad_div_x, grad_div_y


def second_curl_loss(flow):
    """compute the second order curl loss"""
    grad_curl_x, grad_curl_y = _second_curl_loss(flow)
    return charbonnier_loss(grad_curl_x) + charbonnier_loss(grad_curl_y)


def bi_curl_loss(flow_forward, flow_backward):
    """compute the bidirectional first order curl loss"""
    return curl_loss(flow_forward) + curl_loss(flow_backward)


def bi_second_curl_loss(flow_forward, flow_backward):
    """compute the bidirectional second order curl loss"""
    return second_curl_loss(flow_forward)+second_curl_loss(flow_backward)


loss_collection = {'PHOTOMETRIC': bidirectional_warp_loss,
                   'SMOOTHNESS_1ST': bi_smoothness_loss,
                   'SMOOTHNESS_2ND': bi_second_smoothness_loss,
                   'CONSISTENCY': bidirectional_consistency_loss,
                   'DIVERGENCE_1ST': bi_divergence_loss,
                   'DIVERGENCE_2ND': bi_second_divergence_loss,
                   'CURL_1ST': bi_curl_loss,
                   'CURL_2ND': bi_second_curl_loss,
                   'PRESSURE_DIV': divergence_pressure_loss,
                   'TEMPORAL_VORTICITY': temporal_vorticity_loss}
