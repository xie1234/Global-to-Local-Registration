import open3d
import numpy as np
from sklearn.neighbors import KDTree

class PointCloud:
    def __init__(self,point,normal,feature):
        self.point = point
        self.normal = normal
        self.feature = feature

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

def gtlReg(pc_src,pc_tgt):
    xyz_src,feat_src = pc_src.point,pc_src.feature
    xyz_tgt,feat_tgt = pc_tgt.point,pc_tgt.feature
    lamda = 0.9
    resR = np.eye(3)
    resT = np.zeros([3, 1])
    for i in range(10):
        src = np.concatenate(((1 - lamda) * xyz_src, lamda * feat_src), axis=1)
        tgt = np.concatenate(((1 - lamda) * xyz_tgt, lamda * feat_tgt), axis=1)
        kdt_feat = KDTree(tgt, metric='euclidean')
        dist, corr = kdt_feat.query(src, k=1, return_distance=True)
        weight = np.exp(-dist ** 2 / ((1 - lamda) ** 2 * 0.05 ** 2 + lamda ** 2 * 0.3 ** 2))
        R, t = weightedSVD(xyz_src, xyz_tgt[corr, :].reshape(-1, 3), weight)
        resR = R @ resR
        resT = R @ resT + t.reshape(3, 1)
        xyz_src = (R @ xyz_src.T + t.reshape(3, 1)).T
        lamda *= 0.7
    trans = np.concatenate((resR,resT),axis=1)
    pc_res = pc_src
    pc_res.point = xyz_src
    pc_res.normal = (resR @ pc_src.normal.T).T
    return trans,pc_res
