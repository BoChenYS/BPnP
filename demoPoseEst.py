import torch
import numpy as np
import BPnP
import matplotlib.pyplot as plt
import torchvision
from scipy.io import loadmat, savemat
import kornia as kn

device = 'cuda'

cube = loadmat('demo_data/cube.mat')
pts3d_gt = torch.tensor(cube['pts3d'], device=device, dtype=torch.float)
n = pts3d_gt.size(0)
poses = loadmat('demo_data/poses.mat')
P = torch.tensor(poses['poses'][0],device=device).view(1,6) # camera poses in angle-axis
q_gt = kn.angle_axis_to_quaternion(P[0,0:3])

f = 300
u = 128
v = 128
K = torch.tensor(
    [[f, 0, u],
     [0, f, v],
     [0, 0, 1]],
    device=device, dtype=torch.float
)

pts2d_gt = BPnP.batch_project(P, pts3d_gt, K)
bpnp = BPnP.BPnP.apply
ite = 2000

# pts2d = pts2d_gt.clone() + 100*torch.randn_like(pts2d_gt)
# pts2d.requires_grad_()
# optimizer = torch.optim.SGD([{'params':pts2d}], lr=0.2)

model = torchvision.models.vgg11()
model.classifier = torch.nn.Linear(25088,n*2)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000004)

ini_pose = torch.zeros(1,6, device=device)
ini_pose[0,5] = 99
losses = []
track_2d = np.empty([ite,n,2])
track_2d_pro = np.empty([ite,n,2])

plt.figure()
ax3 = plt.subplot(1, 3, 3)
plt.plot(pts2d_gt[0,:,0].clone().detach().cpu().numpy(), pts2d_gt[0,:,1].clone().detach().cpu().numpy(),'rs',ms=10.5, label = 'Target locations')
plt.title('Keypoint evolution')
ax2 = plt.subplot(1, 3, 2)
plt.plot(pts2d_gt[0,:,0].clone().detach().cpu().numpy(), pts2d_gt[0,:,1].clone().detach().cpu().numpy(),'rs',ms=10.5, label = 'Target locations')
plt.title('Pose evolution')

for i in range(ite):

    pts2d = model(torch.ones(1, 3, 32, 32, device=device)).view(1, n, 2)
    track_2d[i, :, :] = pts2d.clone().cpu().detach().numpy()
    P_out = bpnp(pts2d, pts3d_gt, K, ini_pose)
    pts2d_pro = BPnP.batch_project(P_out, pts3d_gt, K)

    loss = ((pts2d_pro - pts2d_gt)**2).mean() + ((pts2d_pro - pts2d)**2).mean()

    print('i: {0:4d}, loss:{1:1.9f}'.format(i, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    track_2d_pro[i, :, :] = pts2d_pro.clone().cpu().detach().numpy()

    if loss.item() < 0.001:
        break
    ini_pose = P_out.detach()

    if i==0:
        ax3.plot(pts2d[0,:, 0].clone().detach().cpu().numpy(), pts2d[0,:, 1].clone().detach().cpu().numpy(), 'ko', ms=8.5, label='Initial location')
        ax2.plot(pts2d_pro[0,:, 0].clone().detach().cpu().numpy(), pts2d_pro[0,:, 1].clone().detach().cpu().numpy(), 'ko', ms=8.5, label='Initial location')
    else:
        ax3.plot(pts2d[0,:, 0].clone().detach().cpu().numpy(), pts2d[0,:, 1].clone().detach().cpu().numpy(), 'k.', ms=0.5)
        ax2.plot(pts2d_pro[0,:, 0].clone().detach().cpu().numpy(), pts2d_pro[0,:, 1].clone().detach().cpu().numpy(), 'k.', ms=0.5)

ax3.plot(pts2d[0,:, 0].clone().detach().cpu().numpy(), pts2d[0,:, 1].clone().detach().cpu().numpy(), 'go', ms = 6, label = 'Final location')
ax2.plot(pts2d_pro[0,:, 0].clone().detach().cpu().numpy(), pts2d_pro[0,:, 1].clone().detach().cpu().numpy(), 'go', ms = 6, label = 'Final location')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(1,3,1)
plt.plot(list(range(len(losses))), losses)
plt.title('Loss evolution')

plt.show()

# savemat('tracks_temp.mat',{'losses':losses, 'track_2d':track_2d, 'track_2d_pro':track_2d_pro, 'pts2d_gt':pts2d_gt.cpu().numpy()})

































