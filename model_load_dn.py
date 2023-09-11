import random,os,pickle,math,tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required

import matplotlib.pyplot as plt

from RBFs import RBFs2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# save_name = "PINNs_fKDV_sech_kuochong_dixing4_rangenew"
# save_name = "PINNs_fKDV_Cn_kuochong_dixing4_rangenew_new_cn"
save_name = "PINNs_fKDV_Dn(s=05)_kuochong_dixing4_rangenew_dnnew"

fun = RBFs2(num_x=35,num_t=121, range_x=(-40,40),range_t=(0,120),device=device).to(device)
optimizer = torch.optim.Adam(fun.parameters(),lr=0.0002)

#调用模型
model_param = torch.load(f"./models/{save_name}.pth")
fun.load_state_dict(model_param["net"])
loss_value = model_param['loss']

xx,tt = np.meshgrid(np.linspace(-30,30,2000),np.linspace(0,90,1000))
[mmm,nnn] = xx.shape

xx_in = torch.tensor(xx.flatten()[:,np.newaxis],dtype=torch.float32,device=device, requires_grad=True)
tt_in = torch.tensor(tt.flatten()[:,np.newaxis],dtype=torch.float32,device=device, requires_grad=True)

u = fun(xx_in,tt_in)

uu = u.cpu().detach().numpy().reshape(mmm,nnn)

#画图
plt.pcolor(xx,tt,uu,cmap='jet')
plt.colorbar()

#画动态图
x_l = np.linspace(-30,30,2000)
for i in range(1000):
    plt.plot(x_l,uu[i,:],color='blue',label='u_std')
    plt.legend()
    plt.ylim(-5,8.)
    plt.xlim(-30,30)
    plt.savefig('./imgs/'+str(i)+'.png',dpi=200)
    plt.close()
#
import imageio as imageio
gif_images = []
for d in range(0, 1000):
    gif_images.append(imageio.imread("./imgs/"+str(d)+".png"))   # 读取多张图片
imageio.mimsave("PINNs_KDV_dn_bell.gif", gif_images)