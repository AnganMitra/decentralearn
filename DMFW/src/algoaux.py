from lib import *

def proj_2shape(x,s=8):
    shape = x.shape
    if torch.linalg.norm(x,ord=1)==s and torch.all(x>0):
        return x
    u,_ = torch.sort(torch.abs(x),dim=0,descending=True)
    cumsum = torch.cumsum(u, dim=0)
    arange = torch.arange(1, shape[0]+1)
    rep_arange = arange.unsqueeze(1).repeat(1,shape[1])
    rho = torch.count_nonzero((u*rep_arange > (cumsum - s)), dim=0)
    theta = (cumsum[rho-1, torch.arange(0,shape[1])] - s)/rho
    proj = (torch.abs(x)-theta).clamp(min=0)
    proj*= torch.sign(x)
    return proj

def proj_l1(x, s=8):
    shape = x.shape
    if len(shape) == 4:
        proj = torch.zeros_like(x)
        for first_dim in range(x.shape[0]):
            for second_dim in range(x.shape[1]):
                inner_tensor = x[first_dim][second_dim]
                inner_proj = proj_2shape(inner_tensor,s=s)
                proj[first_dim][second_dim] = inner_proj
                
    elif len(shape) == 3:
        proj = torch.zeros_like(x)
        for first_dim in range(x.shape[0]):
            inner_tensor = x[first_dim]
            inner_proj = proj_2shape(inner_tensor,s=s)
            proj[first_dim] = inner_proj
        
    elif len(shape) == 2:
        proj = proj_2shape(x,s=s)
        
    elif len(shape) == 1:
        u,_ = torch.sort(torch.abs(x),descending=True)
        cumsum = torch.cumsum(u,dim=0)
        arange = torch.arange(1,shape[0]+1)
        rho = torch.count_nonzero((u*arange > (cumsum - s)))
        theta = (cumsum[rho-1] - s)/rho
        proj = (torch.abs(x)-theta).clamp(min=0)
        proj*= torch.sign(x)
    return proj

def lmo(x,radius):
    """Returns v with norm(v, self.p) <= r minimizing v*x"""
    shape = x.shape
    if len(shape) == 4:
        v = torch.zeros_like(x)
        for first_dim in range(shape[0]):
            for second_dim in range(shape[1]):
                inner_x = x[first_dim][second_dim]
                rows, cols = x[first_dim][second_dim].shape
                v[first_dim][second_dim] = torch.zeros_like(inner_x)
                maxIdx = torch.argmax(torch.abs(inner_x),0)
                for col in range(cols):
                    v[first_dim][second_dim][maxIdx[col],col] = -radius*torch.sign(inner_x[maxIdx[col],col])
    elif len(shape) == 3:
        v = torch.zeros_like(x)
        for first_dim in range(shape[0]):
            inner_x = x[first_dim]
            rows, cols = x[first_dim].shape
            v[first_dim] = torch.zeros_like(inner_x)
            maxIdx = torch.argmax(torch.abs(inner_x),0)
            for col in range(cols):
                v[first_dim][maxIdx[col],col] = -radius*torch.sign(inner_x[maxIdx[col],col])
                    
    elif len(shape)==2:
        rows, cols = x.shape
        v = torch.zeros_like(x)
        maxIdx = torch.argmax(torch.abs(x),0)
        for col in range(cols):
            v[maxIdx[col],col] = -radius*torch.sign(x[maxIdx[col],col])
                
    else : 
        v = torch.zeros_like(x)
        maxIdx = torch.argmax(torch.abs(x))
        v.view(-1)[maxIdx] = -radius * torch.sign(x.view(-1)[maxIdx])
    return v