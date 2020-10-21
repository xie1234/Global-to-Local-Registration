import numpy as np

def svd(src,tgt):
    src_mean = np.mean(src, axis=0, keepdims=True)
    tgt_mean = np.mean(tgt, axis=0, keepdims=True)
    src_shifted = src - src_mean
    tgt_shifted = tgt - tgt_mean
    H = sum((src_shifted[:, :, np.newaxis] @ tgt_shifted[:, np.newaxis, :]))
    U, _, V = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        D[-1][-1] = -1
    R = V.T @ D @ U.T
    t = tgt_mean.T - R @ src_mean.T
    return R,t

def weightedSVD(src,tgt,weight,eps=1e-9):
    weightSum = np.sum(weight)
    weightNorm = weight / (weightSum + eps)
    meanSrc = np.sum(weightNorm * src,axis=0)
    meanTgt = np.sum(weightNorm * tgt,axis=0)

    shiftedSrc = weightNorm * (src - meanSrc)
    shiftedTgt = tgt - meanTgt
    H = sum((shiftedSrc[:, :, np.newaxis] @ shiftedTgt[:, np.newaxis, :]))
    U,D,V = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[-1, -1] = -1
    R = V.T  @ U.T
    t = meanTgt.T - R @ meanSrc.T
    return R, t

def calFeatGrad(point,normal,feature,kdTree):
    # n * 3; n * 3 ; n * d
    N = point.shape[0]
    d = feature.shape[1]
    grads = np.zeros([N,3,d])
    for i in range(N):
        pt = point[i,:].reshape(1,-1)
        nt = normal[i,:].reshape(1,-1)
        ft = feature[i,:].reshape(1,-1)
        _, idx = kdTree.query(pt, k=20, return_distance=True)
        neighbor_pt = point[idx, :].reshape(-1,3)
        neighbor_ft = feature[idx,:].reshape(-1,d)
        proj_pt = neighbor_pt - (neighbor_pt - pt) @ nt.T * nt
        A = proj_pt - pt
        b = neighbor_ft - ft
        A = np.concatenate((A,nt),axis=0)
        b = np.concatenate((b,np.zeros(d).reshape(1,d)))
        x = np.linalg.inv(A.T@A)@A.T@b
        grads[i,:,:] = x
    return grads

def featGN(source,target,corr,grad,lambda_geometric=0.95):
    sqrt_lambda_geometric = np.sqrt(lambda_geometric)
    lambda_photometric = 1.0 - lambda_geometric
    sqrt_lambda_photometric = np.sqrt(lambda_photometric)
    d = source.feature.shape[1]
    N = corr.shape[0]
    ps = source.point[corr[:, 0], :]
    ns = source.normal[corr[:, 0], :]
    fs = source.feature[corr[:, 0] , :]
    pt = target.point[corr[:, 1], :]
    nt = target.normal[corr[:, 1], :]
    ft = target.feature[corr[:, 1], :]
    dt = grad[corr[:, 1], :, :]
    geo_A = np.concatenate((np.cross(ps, nt), nt), axis=1)
    geo_b = np.sum((ps-pt)*nt, axis=1,keepdims=True)

    ps_proj = ps - np.sum((ps-pt)*nt, axis=1,keepdims=True) * nt
    fs_proj = (ps_proj-pt)[:,np.newaxis,:] @ dt
    fs_proj = fs_proj.reshape(N,d) + ft

    M = np.array([[1-nt[:,0]**2, -nt[:,0]*nt[:,1], -nt[:,0]*nt[:,2]],
                  [-nt[:,0]*nt[:,1], 1-nt[:,1]**2, -nt[:,2]*nt[:,1]],
                  [-nt[:,2]*nt[:,0], -nt[:,2]*nt[:,1], 1-nt[:,2]**2]])

    M = M.transpose((2,0,1))
    dmt = -dt.transpose(0,2,1) @ M

    pho_A = np.concatenate((np.cross(ps[:,np.newaxis,:], dmt),dmt),axis=-1)
    pho_b = (fs - fs_proj)[:,:,np.newaxis]

    pho_A = pho_A.reshape(-1,6)
    pho_b = pho_b.reshape(-1,1)

    Ja = np.concatenate((sqrt_lambda_geometric*geo_A,sqrt_lambda_photometric*pho_A),axis=0)
    res = np.concatenate((sqrt_lambda_geometric*geo_b,sqrt_lambda_photometric*pho_b),axis=0)

    vecTrans = -np.linalg.inv(Ja.T@Ja)@Ja.T@res
    vecTrans = np.squeeze(vecTrans)
    cx = np.cos(vecTrans[0])
    cy = np.cos(vecTrans[1])
    cz = np.cos(vecTrans[2])
    sx = np.sin(vecTrans[0])
    sy = np.sin(vecTrans[1])
    sz = np.sin(vecTrans[2])
    R = np.array([[cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz],
                  [cy*sz, cx*cz+sx*sy*sz, cx*sy*sz-sx*cz],
                  [-sy,            sx*cy,          cx*cy]])
    t = vecTrans[3:].reshape(3,1)
    return  R,t
