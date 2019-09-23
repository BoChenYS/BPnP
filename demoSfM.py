import torch
import numpy as np
import BPnP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchvision
from scipy.io import loadmat

pl = 0.00000586
f = 0.0176
u = 960*pl
v = 600*pl
K = torch.cuda.FloatTensor(
    [[f, 0, u],
     [0, f, v],
     [0, 0, 1]]
)

poses = loadmat('demo_data/poses.mat')
duck = loadmat('demo_data/duck_mesh.mat')
n = 1000
pts3d = torch.cuda.FloatTensor(duck['pts3d'])[0:n,:]*pl
pts3d_h = torch.cat((pts3d, torch.ones(n, 1, device='cuda')), dim=-1)

N = 5 # number of images/views
Ps = torch.tensor(poses['poses'],device='cuda')[0:N,:] # camera poses in angle-axis
pts2d = BPnP.batch_project(Ps,pts3d,K)

bpnp = BPnP.BPnP.apply

model = torchvision.models.vgg11()
model.classifier = torch.nn.Linear(25088,n*3)
model = model.cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr = 10)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

ite = 500
ini_pose = Ps
ini_pose[:, 5] = 2
pre_loss = 99
jjj = 5

losses = []
track_pts3d = np.empty([ite,n,3])

for i in range(ite):

    pts3d_out = model(torch.ones(1,3,32,32, device='cuda')).view(n,3)
    P_out = bpnp(pts2d, pts3d_out, K, ini_pose)
    pts2d_pro = BPnP.batch_project(P_out,pts3d_out,K)
    loss = ((pts2d_pro - pts2d)**2).sum()

    print('i: {0:4d}, loss:{1:1.9f}'.format(i, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    track_pts3d[i,:,:] = pts3d_out.cpu().detach().numpy()

    if loss.item() < 0.00000001:
        break
    if pre_loss - loss.item() < 1e-10:
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

print('Done')

