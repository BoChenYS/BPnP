import torch
import cv2 as cv
import numpy as np
import kornia as kn
import math

class BPnP(torch.autograd.Function):
    """
    Back-propagatable PnP
    """
    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        bs = pts2d.size(0)
        n = pts2d.size(1)
        device = pts2d.device
        pts3d_UMat = cv.UMat(np.array(pts3d.detach().cpu()))
        K_UMat = cv.UMat(np.array(K.cpu()))
        P_6d = torch.zeros(bs,6,device=device)

        for i in range(bs):
            pts2d_UMat = cv.UMat(np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n,1,2)))
            if ini_pose is None:
                _, rvec0, T0, _ = cv.solvePnPRansac(objectPoints=pts3d_UMat, imagePoints=pts2d_UMat, cameraMatrix=K_UMat, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, confidence=0.9999 ,reprojectionError=1)
            else:
                rvec0 = cv.UMat(np.array(ini_pose[i, 0:3].cpu().view(3, 1)))
                T0 = cv.UMat(np.array(ini_pose[i, 3:6].cpu().view(3, 1)))
            _, rvec, T = cv.solvePnP(objectPoints=pts3d_UMat, imagePoints=pts2d_UMat, cameraMatrix=K_UMat, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rvec0, tvec=T0)
            angle_axis = torch.tensor(rvec.get(),device=device,dtype=torch.float).view(1, 3)
            T = torch.tensor(T.get(),device=device,dtype=torch.float).view(1, 3)
            P_6d[i,:] = torch.cat((angle_axis,T),dim=-1)

        ctx.save_for_backward(pts2d,P_6d,pts3d,K)
        return P_6d

    @staticmethod
    def backward(ctx, grad_output):

        pts2d, P_6d, pts3d, K = ctx.saved_tensors
        device = pts2d.device
        bs = pts2d.size(0)
        n = pts2d.size(1)
        m = 6

        grad_x = torch.zeros_like(pts2d)
        grad_z = torch.zeros_like(pts3d)
        grad_K = torch.zeros_like(K)

        for i in range(bs):
            J_fy = torch.zeros(m,m, device=device)
            J_fx = torch.zeros(m,2*n, device=device)
            J_fz = torch.zeros(m,3*n, device=device)
            J_fK = torch.zeros(m, 9, device=device)

            pts2d_flat = pts2d[i].clone().view(-1).detach().requires_grad_()             ############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            P_6d_flat = P_6d[i].clone().view(-1).detach().requires_grad_()
            pts3d_flat = pts3d.clone().view(-1).detach().requires_grad_()
            K_flat = K.clone().view(-1).detach().requires_grad_()

            for j in range(m):
                torch.set_grad_enabled(True)
                if j > 0:
                    pts2d_flat.grad.zero_()
                    P_6d_flat.grad.zero_()
                    pts3d_flat.grad.zero_()
                    K_flat.grad.zero_()

                R = kn.angle_axis_to_rotation_matrix(P_6d_flat[0:m-3].view(1,3))

                P = torch.cat((R[0,0:3,0:3].view(3,3), P_6d_flat[m-3:m].view(3,1)),dim=-1)
                KP = torch.mm(K_flat.view(3,3), P)
                pts2d_i = pts2d_flat.view(n,2).transpose(0,1)
                pts3d_i = torch.cat((pts3d_flat.view(n,3),torch.ones(n,1,device=device)),dim=-1).t()
                proj_i = KP.mm(pts3d_i)
                Si = proj_i[2,:].view(1,n)

                r = pts2d_i*Si-proj_i[0:2,:]
                coef = torch.randn(2,n, device=device)
                fj = (coef*r).sum()
                fj.backward()
                J_fy[j,:] = P_6d_flat.grad.clone()
                J_fx[j,:] = pts2d_flat.grad.clone()
                J_fz[j,:] = pts3d_flat.grad.clone()
                J_fK[j,:] = K_flat.grad.clone()

            inv_J_fy = torch.inverse(J_fy)

            J_yx = (-1) * torch.mm(inv_J_fy, J_fx)
            J_yx = J_yx.div(torch.norm(J_yx.view(-1), dim=0))  # normalize the Jacobian of y wrt x
            J_yz = (-1) * torch.mm(inv_J_fy, J_fz)
            J_yz = J_yz.div(torch.norm(J_yz.view(-1), dim=0))  # normalize the Jacobian of y wrt M
            J_yK = (-1) * torch.mm(inv_J_fy, J_fK)
            J_yK = J_yK.div(torch.norm(J_yK.view(-1), dim=0))  # normalize the Jacobian of y wrt K

            grad_x[i] = grad_output[i].view(1,m).mm(J_yx).view(n,2)
            grad_z += grad_output[i].view(1,m).mm(J_yz).view(n,3)
            grad_K += grad_output[i].view(1,m).mm(J_yK).view(3,3)

        return grad_x, grad_z, grad_K, None

def batch_project(P, pts3d, K):
    n = pts3d.size(0)
    bs = P.size(0)
    device = P.device
    pts3d_h = torch.cat((pts3d, torch.ones(n, 1, device=device)), dim=-1)
    R_out = kn.angle_axis_to_rotation_matrix(P[:, 0:3].view(bs, 3))
    PM = torch.cat((R_out[:,0:3,0:3], P[:, 3:6].view(bs, 3, 1)), dim=-1)
    pts3d_cam = pts3d_h.matmul(PM.transpose(1,2))
    pts2d_proj = pts3d_cam.matmul(K.t())
    S = pts2d_proj[:,:, 2].view(bs, n, 1)
    pts2d_pro = pts2d_proj[:,:,0:2].div(S)

    return pts2d_pro