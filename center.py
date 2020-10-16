from torch.autograd import Variable
import numpy as np
import torch


def Relaxcenter(Y, B, V, mu, vul, nta):

    """
    GD: relax C to V.  Corresponding to the relaxation optimization approach 
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



def Discretecenter(Y, B, C, mu, vul):
    """Solve DCC(Discrete Cyclic Coordinate Descent) problem. 
    """
    ones_vector = torch.ones([C.size(0) - 1])
    for i in range(C.shape[0]):
        Q = Y @ B.t()
        q = Q[i, :]
        v = Y[i, :]
        Y_prime = torch.cat((Y[:i, :], Y[i+1:, :]))
        C_prime = torch.cat((C[:i, :], C[i+1:, :]))

        C[i, :] = (q - C_prime.t() @ Y_prime @ v - vul *C_prime.t()@ones_vector).sign()

    return C.t() 
