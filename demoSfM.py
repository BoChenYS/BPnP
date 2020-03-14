import torch
import numpy as np
import BPnP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchvision
from scipy.io import loadmat, savemat

device = 'cuda'

pl = 0.00000586
f = 0.0005
u = 0
v = 0
K = torch.tensor(
    [[f, 0, u],
     [0, f, v],
     [0, 0, 1]], dtype=torch.float, device=device
)

poses = loadmat('demo_data/poses.mat')
duck = loadmat('demo_data/duck_mesh.mat')
n = 1000
pts3d = torch.tensor(duck['pts3d'], dtype=torch.float, device=device)[0:n,:]*pl
pts3d_h = torch.cat((pts3d, torch.ones(n, 1, device=device)), dim=-1)

N = 12 # number of images/views
Ps = torch.tensor(poses['poses'],device=device)[0:N,:] # camera poses in angle-axis
pts2d = BPnP.batch_project(Ps,pts3d,K)

bpnp = BPnP.BPnP.apply

model = torchvision.models.vgg11()
model.classifier = torch.nn.Linear(25088,n*3)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

ite = 1000
ini_pose = 0*Ps
ini_pose[:, 5] = 99
pre_loss = 99
jjj = 20

losses = []
track_pts3d = np.empty([ite,n,3])

meds = pts3d.median(dim=0).values
vis = torch.zeros(N,n,1,device=device)
for i in range(N):
    if i%6 == 0:
        ids = pts3d[:,0] >= meds[0]
        vis[i,ids,:] = 1.
    if i%6 == 1:
        ids = pts3d[:,0] < meds[0]
        vis[i,ids,:] = 1.
    if i%6 == 2:
        ids = pts3d[:,1] >= meds[1]
        vis[i,ids,:] = 1.
    if i%6 == 3:
        ids = pts3d[:,1] < meds[1]
        vis[i,ids,:] = 1.
    if i % 6 == 4:
        ids = pts3d[:, 2] >= meds[2]
        vis[i, ids, :] = 1.
    if i % 6 == 5:
        ids = pts3d[:, 2] < meds[2]
        vis[i, ids, :] = 1.

for i in range(ite):

    pts3d_out = model(torch.ones(1,3,32,32, device=device)).view(n,3)
    P_out = bpnp(pts2d, pts3d_out, K, ini_pose)
    pts2d_pro = BPnP.batch_project(P_out,pts3d_out,K)
    loss = (((pts2d_pro - pts2d)*vis)**2).sum()

    print('i: {0:4d}, loss:{1:1.12f}'.format(i, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    track_pts3d[i,:,:] = pts3d_out.cpu().detach().numpy()

    if loss.item() < 1e-10:
        break
    if pre_loss - loss.item() < 1e-12 and pre_loss - loss.item() > 0:
        jjj -= 1
    if jjj == 0:
        break

    ini_pose = P_out.detach()
    pre_loss = loss.item()

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
ax1.scatter(track_pts3d[0,:,0],track_pts3d[0,:,1],track_pts3d[0,:,2],s=2)
ax2.scatter(track_pts3d[i,:,0],track_pts3d[i,:,1],track_pts3d[i,:,2],s=2)
fig.suptitle('Initial vs final output')
plt.show()

# savemat('sfm_temp.mat',{'loss':losses, 'pts3d_track':track_pts3d, 'pts2d':pts2d.cpu().numpy(), 'pts3d_gt':pts3d.cpu().numpy(), 'vis':vis.cpu().numpy()})
print('Done')
