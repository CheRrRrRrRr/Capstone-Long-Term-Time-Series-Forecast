import torch
import numpy as np

def filter_pred(pred):
    # check whether prediction is 0, is so ,change it to -1
    return torch.where(pred == 0, -1, pred)

def epmae(pred,true):
    # can also use the following 'where' condition but the loss declines inefficiently 
    # fx = torch.where(pred*true > 0, 1, -1)
    # pmae1 = (fx+1)*0.5*abs(pred-true)
    # pmae2 = (fx-1)*(-0.5)*(torch.exp(abs(pred-true))-1)

    # adj_pred = filter_pred(pred)

    fx1 = pred/abs(pred)
    fx2 = true/abs(true)

    fx = fx1*fx2 

    pmae1 = (fx+1)*0.5*abs(pred-true)
    pmae2 = (fx-1)*(-0.5)*(torch.exp(abs(pred-true))-1)

    loss = torch.mean(pmae1+pmae2)

    return loss


def qpmae(alpha):
    def qpmse_alpha(pred,true):

        fx1 = pred/abs(pred)
        fx2 = true/abs(true)

        fx = fx1*fx2 

        pmse = abs(pred-true) + (fx-1)*(-0.5)*alpha*(pred-true)**2
        
        loss = torch.mean(pmse)

        return loss
    
    return qpmse_alpha


def lepmae(pred,true):

    # adj_pred = filter_pred(pred)
    # fx = torch.where(true*adj_pred > 0, 1, -1)
    
    fx1 = pred/abs(pred)
    fx2 = true/abs(true)

    fx = fx1*fx2 

    pmae1 = (fx+1)*0.5*abs(pred-true)
    pmae2 = (fx-1)*(-0.5)*(torch.exp(abs(pred-true))+abs(true)-torch.exp(abs(true)))
    # pmae2 = (fx-1)*(-0.5)*(abs(pred-true)*torch.exp(abs(pred-true)))
 

    loss = torch.mean(pmae1+pmae2)

    return loss

def alepmae(pred,true):
    
    fx1 = pred/abs(pred)
    fx2 = true/abs(true)

    fx = fx1*fx2 

    pmae1 = (fx+1)*0.5*abs(pred-true)
    pmae2 = (fx-1)*(-0.5)*((abs(pred-true)+1)*torch.exp(abs(pred-true))-1)

    loss = torch.mean(pmae1+pmae2)

    return loss

def aqpmae(alpha):
    def aqpmae_alpha(pred,true):

        fx1 = pred/abs(pred)
        fx2 = true/abs(true)

        fx = fx1*fx2 

        pmse = abs(pred-true) + (fx-1)*(-0.5)*alpha*(abs(pred-true)+1)**2

        loss = torch.mean(pmse)

        return loss
    
    return aqpmae_alpha