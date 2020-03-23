from torch.autograd import Variable
import numpy as np
import torch


def Center_gradient(Y, B, V, mu, vul, nta):

    """
    GD: relax C to V.
    """
    alpha = 0.03
    num = 1
    T = 2*torch.eye(Y.size(0)) - torch.ones(Y.size(0))
    TK = V.size(0)*T 
    TK = torch.FloatTensor(Variable(TK, requires_grad = False))

    for i in range(200):
        intra_loss = (V@Y - B).pow(2).mean()
        inter_loss = (V.t()@V - TK.cuda()).pow(2).mean()
        quantization_loss = (V - V.sign()).pow(2).mean()

        loss = intra_loss + (vul) * inter_loss + (nta) * quantization_loss
        loss.backward()

        num += 1        
        if num ==150 or num ==180:
            alpha = alpha*0.1

        V.data = V.data - alpha * V.grad.data
        V.grad.data.zero_()

    V_u = V.data.cpu()
    Center_u = V_u.sign()

    return Center_u, V_u    


def Dis_Center_gradient(Y, B, C, mu, vul):

    """
    Discrete update. C = sign(C - lr*dJdC)
    """
    alpha = 0.03
    num = 1
    T = 2*torch.eye(Y.size(0)) - torch.ones(Y.size(0))
    TK = V.size(0)*T 
    TK = torch.FloatTensor(Variable(TK, requires_grad = False))

    for i in range(200):

        intra_loss = (C@Y - B).pow(2).mean()
        inter_loss = (C.t()@C - TK.cuda()).pow(2).mean()

        loss = intra_loss + (vul) * inter_loss
        loss.backward()

        num += 1        
        if num ==150 or num ==180:
            alpha = alpha*0.1

        C.data = (C.data - alpha * C.grad.data).sign()
        C.grad.data.zero_()

    Center_u = C.data.cpu()

    return Center_u    
