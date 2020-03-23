import torch.nn as nn
import torch
from torch.autograd import Variable



class DSDHLoss_margin(nn.Module):
    def __init__(self, eta, margin):
        super(DSDHLoss_margin, self).__init__()
        self.eta = eta
        self.margin = margin

    def forward(self, U_batch, U, S, B):
        theta = U.t() @ U_batch / 2

        # Prevent exp overflow
        theta = torch.clamp(theta, min=-100, max=50)

        # metric_loss = ((1-S) * torch.log(1 + torch.exp(self.margin + theta)) + S * torch.log(1 + torch.exp(self.margin - theta))).mean()  
        metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()  # Without Margin
        quantization_loss = (B - U_batch).pow(2).mean()
        loss = metric_loss + self.eta * quantization_loss

        return loss