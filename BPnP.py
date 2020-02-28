import torch
import cv2 as cv
import numpy as np
import kornia as kn
import rowan

class BPnP(torch.autograd.Function):
    """
    Back-propagatable PnP
    """
    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None):
        bs = pts2d.size(0)
        n = pts2d.size(1)
        device = pts2d.device
        pts3d_np = np.array(pts3d.detach().cpu())
        K_np = np.array(K.cpu())
        P_6d = torch.zeros(bs,6,device=device)

        for i in range(bs):
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n,1,2))
            if ini_pose is None:
                _, rvec0, T0, _ = cv.solvePnPRansac(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, confidence=0.9999 ,reprojectionError=3)
            else:
                rvec0 = np.array(ini_pose[i, 0:3].cpu().view(3, 1))
                T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
            _, rvec, T = cv.solvePnP(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rvec0, tvec=T0)
            angle_axis = torch.tensor(rvec,device=device,dtype=torch.float).view(1, 3)
            T = torch.tensor(T,device=device,dtype=torch.float).view(1, 3)
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

            coefs = get_coefs(P_6d[i].view(1,6), pts3d, K)

            pts2d_flat = pts2d[i].clone().view(-1).detach().requires_grad_()
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
                coef = coefs[:,:,j].transpose(0,1) # size: [2,n]
                fj = (coef*r).sum()
                fj.backward()
                J_fy[j,:] = P_6d_flat.grad.clone()
                J_fx[j,:] = pts2d_flat.grad.clone()
                J_fz[j,:] = pts3d_flat.grad.clone()
                J_fK[j,:] = K_flat.grad.clone()

            inv_J_fy = torch.inverse(J_fy)

            J_yx = (-1) * torch.mm(inv_J_fy, J_fx)
            J_yz = (-1) * torch.mm(inv_J_fy, J_fz)
            J_yK = (-1) * torch.mm(inv_J_fy, J_fK)

            grad_x[i] = grad_output[i].view(1,m).mm(J_yx).view(n,2)
            grad_z += grad_output[i].view(1,m).mm(J_yz).view(n,3)
            grad_K += grad_output[i].view(1,m).mm(J_yK).view(3,3)

        return grad_x, grad_z, grad_K, None

def get_coefs(P_6d, pts3d, K):
    device = P_6d.device
    n = pts3d.size(0)
    m = P_6d.size(-1)
    coefs = torch.zeros(n,2,m,device=device)
    torch.set_grad_enabled(True)
    y = P_6d.clone().repeat(n,1).detach().requires_grad_()
    proj = batch_project(y, pts3d.detach(), K.detach()).squeeze()
    vec = torch.diag(torch.ones(n,device=device).float())
    for k in range(2):
        torch.set_grad_enabled(True)
        y_grad = torch.autograd.grad(proj[:,:,k],y,vec, retain_graph=True)
        coefs[:,k,:] = -2*y_grad[0].clone()
    return coefs

def batch_project(P, pts3d, K, angle_axis=True):
    n = pts3d.size(0)
    bs = P.size(0)
    device = P.device
    pts3d_h = torch.cat((pts3d, torch.ones(n, 1, device=device)), dim=-1)
    if angle_axis:
        R_out = kn.angle_axis_to_rotation_matrix(P[:, 0:3].view(bs, 3))
        PM = torch.cat((R_out[:,0:3,0:3], P[:, 3:6].view(bs, 3, 1)), dim=-1)
    else:
        PM = P
    pts3d_cam = pts3d_h.matmul(PM.transpose(1,2))
    pts2d_proj = pts3d_cam.matmul(K.t())
    S = pts2d_proj[:,:, 2].view(bs, n, 1)
    pts2d_pro = pts2d_proj[:,:,0:2].div(S)

    return pts2d_pro

def get_res(pts2d, pts3d, K, P):
    n = pts2d.size(0)
    m = 6
    feas1 = P[0,m-1].item() > 0
    R = kn.angle_axis_to_rotation_matrix(P[0, 0:m - 3].view(1, 3))
    P = torch.cat((R[0, 0:3, 0:3].view(3, 3), P[0, m - 3:m].view(3, 1)), dim=-1)
    pts3d_h = torch.cat((pts3d,torch.ones(n,1,device=pts3d.device)), dim=-1)
    pts3d_cam = pts3d_h.mm(P.transpose(0, 1))
    feas2 = (pts3d_cam[:,2].min().item() >= 0)
    feas = feas1 and feas2
    pts2d_proj = pts3d_cam.mm(K.transpose(0, 1))
    S = pts2d_proj[:, 2].view(n, 1)
    res = pts2d - pts2d_proj[:, 0:2].div(S)
    return torch.norm(res,dim=1).sum().item(), feas

def P6d2PM(P6d):
    bs = P6d.size(0)
    PM = kn.angle_axis_to_rotation_matrix(P6d[:,0:3].view(bs,3))
    T = P6d[:,3:6].view(bs,3,1)
    PM = torch.cat((PM[:,0:3,0:3].view(bs,3,3),T),dim=-1)
    return PM

class BPnP_robust(torch.autograd.Function):
    """
    Back-propagatable PnP, robust version, for the case of poor 2d-3d correspondences in early stage of training, by doing RANSAC every time.
    """
    @staticmethod
    def forward(ctx, pts2d, pts3d, K, ini_pose=None, prevent_deteriorate=True):
        bs = pts2d.size(0)
        n = pts2d.size(1)
        device = pts2d.device
        pts3d_np = np.array(pts3d.detach().cpu())
        K_np = np.array(K.cpu())
        P_6d = torch.zeros(bs,6,device=device)

        for i in range(bs):
            pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n,1,2))

            _, rvec0r, T0r, _ = cv.solvePnPRansac(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np,
                                                distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, confidence=0.9999,
                                                reprojectionError=20)
            angle_axis0r = torch.tensor(rvec0r, device=device, dtype=torch.float).view(1, 3)
            tra0r = torch.tensor(T0r, device=device, dtype=torch.float).view(1, 3)
            P_6d_0r = torch.cat((angle_axis0r, tra0r), dim=-1)
            res0r, feas0r = get_res(pts2d[i], pts3d, K, P_6d_0r)

            if ini_pose is None:
                rvec0 = rvec0r
                T0 = T0r
                P_6d_i_before = P_6d_0r
                res0 = res0r
                feas0 = feas0r
            else:
                res0i, feas0i = get_res(pts2d[i], pts3d, K, ini_pose[i].view(1,6))
                if feas0r and not feas0i:
                    rvec0 = rvec0r
                    T0 = T0r
                    P_6d_i_before = P_6d_0r
                    res0 = res0r
                    feas0 = feas0r
                elif feas0i and not feas0r:
                    rvec0 = np.array(ini_pose[i, 0:3].cpu().view(3, 1))
                    T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
                    P_6d_i_before = ini_pose[i].view(1,6)
                    res0 = res0i
                    feas0 = feas0i
                else:
                    if res0i < res0r:
                        rvec0 = np.array(ini_pose[i, 0:3].cpu().view(3, 1))
                        T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
                        P_6d_i_before = ini_pose[i].view(1,6)
                        res0 = res0i
                        feas0 = feas0i
                    else:
                        rvec0 = rvec0r
                        T0 = T0r
                        P_6d_i_before = P_6d_0r
                        res0 = res0r
                        feas0 = feas0r

            _, rvec, T = cv.solvePnP(objectPoints=pts3d_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None,
                                     flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=np.copy(rvec0),
                                     tvec=np.copy(T0))

            if prevent_deteriorate==True:
                angle_axis = torch.tensor(rvec, device=device, dtype=torch.float).view(1, 3)
                tra = torch.tensor(T, device=device, dtype=torch.float).view(1, 3)
                P_6d_i_after =  torch.cat((angle_axis,tra),dim=-1)
                res, feas = get_res(pts2d[i], pts3d, K, P_6d_i_after)
                if feas0 and not feas:
                    P_6d[i, :] = P_6d_i_before
                elif feas and not feas0:
                    P_6d[i, :] = P_6d_i_after
                else:
                    if res <= res0:
                        P_6d[i,:] = P_6d_i_after
                    else:
                        P_6d[i, :] = P_6d_i_before
            else:
                angle_axis = torch.tensor(rvec, device=device, dtype=torch.float).view(1, 3)
                T = torch.tensor(T, device=device, dtype=torch.float).view(1, 3)
                P_6d[i, :] = torch.cat((angle_axis,T),dim=-1)

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

            coefs = get_coefs(P_6d[i].view(1, 6), pts3d, K)

            pts2d_flat = pts2d[i].clone().view(-1).detach().requires_grad_()
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
                coef = coefs[:,:,j].transpose(0,1) # size: [2,n]
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




