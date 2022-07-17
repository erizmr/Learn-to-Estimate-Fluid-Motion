import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import convolve
from fluidestimator.modeling.loss_functions import warp_loss, charbonnier_loss, smoothness_loss, divergence_loss


def horn_schunck(img1, img2, alpha, iteration, tol=1e-6):
    """
    Implementation of Horn-Schunck Optical flow method (1981)
    Args:
        img1: first frame
        img2: second frame
        alpha: weight for smoothness constrain
        iteration: max iteration steps
        tol: tolerance for iteration stop

    Returns: estimated optical flow U (x-axis) and V (y-axis)

    """
    # HS kernel
    HornSchunckKernel = np.array(
        [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]],
        dtype="float32")

    img1 = np.array(img1, dtype="float32")
    img2 = np.array(img2, dtype="float32")

    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype="float32")
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype="float32")

    Ix = convolve(img1, sobel_x)
    Iy = convolve(img1, sobel_y)
    It = np.subtract(img2, img1)

    U = np.zeros([img1.shape[0], img1.shape[1]])
    V = np.zeros([img1.shape[0], img1.shape[1]])
    U_prev = U
    V_prev = V
    for i in range(iteration):
        u_average = convolve(U, HornSchunckKernel)
        v_average = convolve(V, HornSchunckKernel)
        P = Ix * u_average + Iy * v_average + It
        D = alpha + Ix**2 + Iy**2
        U = u_average - Ix * P / D
        V = v_average - Iy * P / D
        residual = np.sqrt(np.mean((U - U_prev)**2)) + np.sqrt(np.mean((V - V_prev)**2))
        U_prev = U
        V_prev = V
        if residual < tol:
            break
        if i % 100 == 0:
            print('iter: {}, residual: {}'.format(i, residual))

    return U, V


def horn_schunck_gpu(img1, img2, alpha, iteration, tol=1e-6, device='cuda'):
    # HS kernel
    HornSchunckKernel = torch.randn(1, 1, 3, 3)
    HornSchunckKernel[0, 0, :, :] = torch.FloatTensor(
        [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]])
    HornSchunckKernel = HornSchunckKernel.to(device)

    if not torch.is_tensor(img1):
        img1 = torch.from_numpy(np.array(img1, dtype="float32"))
    img1 = img1.to(device)
    if not torch.is_tensor(img2):
        img2 = torch.from_numpy(np.array(img2, dtype="float32"))
    img2 = img2.to(device)

    sobel_x = torch.randn(1, 1, 3, 3)
    sobel_y = torch.randn(1, 1, 3, 3)
    sobel_x[0, 0, :, :] = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y[0, 0, :, :] = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)

    Ix = F.conv2d(img1, sobel_x, padding=1)
    Iy = F.conv2d(img1, sobel_y, padding=1)
    It = torch.subtract(img2, img1)

    U = torch.zeros(img1.shape).to(device)
    V = torch.zeros(img1.shape).to(device)

    U_prev = U.clone()
    V_prev = V.clone()
    for i in range(iteration):
        u_average = F.conv2d(U, HornSchunckKernel, padding=1)
        v_average = F.conv2d(V, HornSchunckKernel, padding=1)
        P = Ix * u_average + Iy * v_average + It
        D = alpha + Ix ** 2 + Iy ** 2
        U = u_average - Ix * P / D
        V = v_average - Iy * P / D
        residual = torch.sqrt(torch.mean((U - U_prev) ** 2)) + torch.sqrt(torch.mean((V - V_prev) ** 2))
        U_prev = U.clone()
        V_prev = V.clone()
        if residual < tol:
            break
        if i % 100 == 0:
            print('iter: {}, residual: {}'.format(i, residual))
    return U, V


def horn_schunck_adam(img1, img2, iteration, device='cuda'):
    img1 = torch.from_numpy(np.array(img1, dtype="float32")).to(device)
    img2 = torch.from_numpy(np.array(img2, dtype="float32")).to(device)

    bs, c, h, w = img1.shape
    # Rand Intialization (usually worse performance)
    # flow = torch.rand([1, 2, h, w], device=device).requires_grad_()
    # Zero Intialization
    flow = torch.zeros([1, 2, h, w], device=device).requires_grad_()
    optimizer = torch.optim.Adam([flow], lr=1e-2, eps=1e-3)

    for k in range(iteration):
        optimizer.zero_grad()

        photo_loss = warp_loss(img1, img2, flow)
        smooth_loss = smoothness_loss(flow)
        div_loss = divergence_loss(flow)

        loss = photo_loss + 10.5 * smooth_loss + 10.0 * div_loss
        loss.backward()
        optimizer.step()

        if k % 1000 == 0:
            print('iter: {}, loss: {}, photo loss: {}, smooth loos: {}, div loss {}'.format(
                k,
                loss.item(),
                photo_loss.item(),
                smooth_loss.item(),
                div_loss.item()))

    return flow[:, 0, :, :].view(-1, 1, h, w), flow[:, 1, :, :].view(-1, 1, h, w)
