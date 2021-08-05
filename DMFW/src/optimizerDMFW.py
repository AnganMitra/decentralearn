from lib import *
from algoaux import *

class DMFW(optim.Optimizer):
    def __init__(self, params, eta_coef=required, eta_exp=required, L=required, matrix_line=required, reg_coef=required,radius=required):
        if eta_coef is not required and eta_coef <=0.:
            raise ValueError("Invalid eta : {}".format(eta_coef))
        if eta_exp is not required and (eta_exp == 0.5):
            raise ValueError("Invalid eta_exp : {}".format(eta_exp))
        defaults = dict(eta_coef=eta_coef, eta_exp=eta_exp,L=L, matrix_line=matrix_line,reg_coef = reg_coef,radius=radius)
        super(DMFW,self).__init__(params,defaults)

        for group in self.param_groups:
            self.eta_coef = group["eta_coef"]
            self.eta_exp = group["eta_exp"]
            self.reg_coef = group["reg_coef"]
            self.A = group["matrix_line"]
            self.L = group["L"]
            self.radius = group["radius"]
        self.num_layers = len(self.param_groups[0]['params'])
        self.dim = [k.shape for k in self.param_groups[0]['params']]
        self.G = [[torch.rand(k) for k in self.dim] for l in range(self.L)]
        
    @torch.no_grad()
    def initValue(self,closure):
        self.w_dict = defaultdict(dict)
        for group in self.param_groups:
            if closure is not None:
                with torch.enable_grad():
                    closure()
            for k,weight in enumerate(group["params"]):
                if weight.grad is None:
                    raise ValueError("Gradient is None")
                self.w_dict[k]["g"] = weight.grad.detach().clone()
                self.w_dict[k]["w"] = weight.detach().clone()
                
    def neighborsAverage(self, neighbors):
        for group in self.param_groups:
            for k,weight in enumerate(group["params"]):
                weighted_tmp = torch.zeros(self.dim[k])
                weighted_grad_tmp = torch.zeros(self.dim[k])
                for j in range(len(neighbors)):
                    weighted_tmp += self.A[j]*neighbors[j].w_dict[k]["w"]
                    weighted_grad_tmp += self.A[j]*neighbors[j].w_dict[k]["g"]
                self.w_dict[k]["y"] = weighted_tmp
                self.w_dict[k]["ds"] = weighted_grad_tmp
                

    def step(self, l, closure):
        if l == 0:
            self.init_gap = 0
        eta = min(self.eta_coef/(l+1)**self.eta_exp, 1)
        for group in self.param_groups:
            if closure is not None:
                with torch.enable_grad():
                    closure()
            self.gap = 0
            for k,weight in enumerate(group["params"]):
                a = lmo(weight.grad.data,self.radius)
                self.gap += torch.sum(torch.mul(weight.grad.data, weight.data - a))
                v = proj_l1(self.G[l][k], s=self.radius)
                #v = lmo(self.G[l][k] - 0.5 + torch.rand_like(self.G[l][k]), s= self.radius)
                if weight.grad is None:
                    raise ValueError("Grad is None")
                self.w_dict[k]["grad_old"] = weight.grad.detach().clone()
                weight.data = self.w_dict[k]['y']*(1-eta) + eta*v
                #print("weight {}".format(torch.linalg.norm(weight.data,ord=1,dim=0)))
                self.w_dict[k]["w"] = weight.detach().clone()
                
            self.init_gap += self.gap
            self.init_gap /= (l+1)
            
            
            with torch.enable_grad():
                closure()
            for k,weight in enumerate(group["params"]):
                self.G[l][k] -= 0.5*self.w_dict[k]["ds"]*self.reg_coef
                #self.G[l][k] += self.w_dict[k]["ds"]*self.reg_coef
                if weight.grad is None :
                    raise ValueError("Grad is none")
                weight.grad.add_(-self.w_dict[k]["grad_old"])
                self.w_dict[k]["g"] = weight.grad.detach().clone() + self.w_dict[k]["ds"]
                