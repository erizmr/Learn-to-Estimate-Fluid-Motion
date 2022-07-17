import torch
import torch.nn as nn
from fluidestimator.tools import FieldWarper, CalculusOperators


class Corrector(nn.Module):
    def diffusion_term(flow):
        laplace_u, laplace_v = CalculusOperators.laplace(flow)
        u_diffusion = torch.sum(laplace_u, dim=1)
        v_diffusion = torch.sum(laplace_v, dim=1)
        return torch.stack([u_diffusion, v_diffusion], dim=1)

    def convection_term(flow):
        delta_u, delta_v = CalculusOperators.gradient(flow)
        u_convection = torch.sum(flow * delta_u, dim=1)
        v_convection = torch.sum(flow * delta_v, dim=1)
        return torch.stack([u_convection, v_convection], dim=1)
    
    def __init__(self, cfg, input_dim=2, hidden_dim=6, kernel_size=(7, 7), bias=True):
        super(Corrector, self).__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # Compute values of different orders differentiation operators
        self.conv_hidden_physics = nn.Conv2d(in_channels=input_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                                    stride=(1, 1), padding=self.padding)
        self.conv_weighted_sum = nn.Conv2d(in_channels=self.hidden_dim, out_channels=2, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0))
        
        self.convgate = nn.Conv2d(in_channels=2*self.input_dim,
                        out_channels= self.input_dim,
                        kernel_size=(3, 3),
                        padding=(1, 1), bias=self.bias)

    def forward(self, u_prev, u_hat):
        # Compute phi
        phi = self.conv_weighted_sum(self.conv_hidden_physics(u_hat - u_prev))

        # Physical Advection-diffusion
        u_star = u_prev + self.cfg.MODEL.CORRECTOR.COEF_DIFFUSION * Corrector.diffusion_term(u_prev) - self.cfg.MODEL.CORRECTOR.COEF_CONVECTION * Corrector.convection_term(u_prev)

        # Compute gain
        gate_input = self.convgate(torch.cat([u_star, u_hat], dim=1))
        K_t = torch.sigmoid(gate_input)

        # Compute refinement
        refinement = (1-K_t)*u_star + phi
        
        # Compute correction
        u_t = K_t*u_hat + refinement
        if self.training:
            loss_dict = {}
            prev_vorticity = CalculusOperators.curl(u_prev)
            current_vorticity = CalculusOperators.curl(u_t)
            current_divergence = CalculusOperators.divergence(u_t)
            # backwarp the vorticity prediction at current step
            warped_prev = FieldWarper.backward_warp(tensorInput=current_vorticity, tensorFlow=u_t)
            loss_dict['LOSS_VORTICITY_WARP'] = self.cfg.TRAIN.LOSS.VORTICITY_WARP * nn.MSELoss()(prev_vorticity, warped_prev)
            loss_dict['LOSS_REFINE_REGULARIZER'] = self.cfg.TRAIN.LOSS.REFINE_REGULARIZER * torch.mean(refinement**2)
            loss_dict['LOSS_CORRECTOR_DIVERGENCE'] = self.cfg.TRAIN.LOSS.CORRECTOR_DIVERGENCE * torch.mean(current_divergence**2)
            return loss_dict, u_t
        else:
            return u_t, u_star, refinement, phi