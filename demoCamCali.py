from __future__ import print_function, division
import torch
import numpy as np
import BPnP
import matplotlib.pyplot as plt
import kornia as kn
from scipy.io import savemat, loadmat

device = 'cuda'

cube = loadmat('demo_data/cube.mat')
pts3d_gt = torch.tensor(cube['pts3d'], device=device, dtype=torch.float)
n = pts3d_gt.size(0)
poses = loadmat('demo_data/poses.mat')
P = torch.tensor(poses['poses'][0],device=device).view(1,6) # camera poses in angle-axis
q_gt = kn.angle_axis_to_quaternion(P[0,0:3])

fx = 800
fy = 700
u = 400
v = 300
K = torch.tensor(
    [[fx, 0, u],
     [0, fy, v],
     [0, 0, 1]],
    device=device, dtype=torch.float
)

pts2d_gt = BPnP.batch_project(P, pts3d_gt, K)
bpnp = BPnP.BPnP.apply

theta = (1.1*torch.randn(4,device=device)).requires_grad_()
optimizer = torch.optim.SGD({theta}, lr = 0.00001)

losses = []
ite = 2000
ini_pose = torch.zeros(1,6, device=device)
ini_pose[0,5] = 999

track_Ks = np.empty([ite,4])

for i in range(ite):

    cp = 1000*torch.sigmoid(theta)
    row1 = torch.cat((cp[0].view(1),torch.zeros(1,device=device).requires_grad_(),cp[2].view(1)),dim=-1).view(1,3)
    row2 = torch.cat((torch.zeros(1,device=device).requires_grad_(),cp[1].view(1),cp[3].view(1)),dim=-1).view(1,3)
    row3 = torch.tensor([[0,0,1]], device=device, dtype=torch.float).requires_grad_()
    K_out = torch.cat((row1,row2,row3),dim=0)

    P_out = bpnp(pts2d_gt,pts3d_gt,K_out, ini_pose)
    pts2d_pro = BPnP.batch_project(P_out, pts3d_gt, K_out).squeeze()
    loss = ((pts2d_pro - pts2d_gt)**2).sum()

    print('i: {0:4d}, loss:{1:1.10f}'.format(i, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    track_Ks[i,:] = cp.detach().cpu().numpy()

    if loss.item() < 0.001:
        break
    ini_pose = P_out.detach()

plt.subplot(1,2,1)
plt.plot(list(range(len(losses))), losses)
plt.title('Loss evolution')

plt.subplot(1,2,2)
plt.plot(list(range(len(losses))), track_Ks[:len(losses),:] )
plt.title('Intrinsic parameters evolution')
plt.legend(('f_x', 'f_y', 'c_x', 'c_y'),bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# savemat('CamCali_temp.mat',{'losses':losses, 'track_Ks':track_Ks})


