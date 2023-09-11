import torch
import torch.nn as nn
import torch.nn.functional as F

    
class rbf2(nn.Module):
    def __init__(self,num_x, device,range_x,first=False):
        super(rbf2, self).__init__()
        if first:
            self.aw = torch.linspace(range_x[0],range_x[1],num_x,dtype=torch.float32,device=device, requires_grad=False)
        else:
            self.aw = 0.5*torch.ones(num_x,dtype=torch.float32,device=device, requires_grad=False)
        self.sigma2 = torch.tensor((range_x[1]-range_x[0])**2/12,dtype=torch.float32,device=device, requires_grad=False)

        self.a = nn.Parameter(torch.ones(num_x,dtype=torch.float32, requires_grad=True))
        self.b = nn.Parameter(torch.ones(num_x,dtype=torch.float32, requires_grad=True))
    def forward(self, x):
        return torch.exp(-.5*(x-self.a*self.aw)**2/(self.b**2*self.sigma2))

class RBFs2(nn.Module):
    def __init__(self,num_x,num_t, range_x,range_t,device):
        super(RBFs2, self).__init__() 
        self.inner_num = 4*(num_x+num_t)
        self.rbf_x = rbf2(num_x=num_x,device=device,range_x=range_x,first=True)
        self.rbf_t = rbf2(num_x=num_t,device=device,range_x=range_t,first=True)

        self.net = nn.Sequential(
            nn.Linear(num_x+num_t,self.inner_num),
            rbf2(self.inner_num,device=device,range_x=(0,1),first=False), nn.Linear(self.inner_num,self.inner_num),
            rbf2(self.inner_num,device=device,range_x=(0,1),first=False), nn.Linear(self.inner_num,self.inner_num),
            rbf2(self.inner_num,device=device,range_x=(0,1),first=False), nn.Linear(self.inner_num,self.inner_num),
            rbf2(self.inner_num,device=device,range_x=(0,1),first=False), nn.Linear(self.inner_num,1)
        )
    def forward(self, x, t):
        x = self.rbf_x(x)
        t = self.rbf_t(t)
        xt = self.net(torch.cat([x,t],dim=1))

        return xt
