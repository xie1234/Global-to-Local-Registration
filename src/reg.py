class PointCloud:
    def __init__(self,point,normal,feature):
        self.point = point
        self.normal = normal
        self.feature = feature

import open3d
import numpy as np
from sklearn.neighbors import KDTree
import  h5py
from solve import *
import copy
import  timeit

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

def featureICP(pc_src,pc_tgt):
    xyz_src, norm_src,feat_src = pc_src.point, pc_src.normal,pc_src.feature
    xyz_tgt, norm_tgt,feat_tgt = pc_tgt.point, pc_tgt.normal,pc_tgt.feature
    kdt = KDTree(xyz_tgt, metric='euclidean')
    grad = calFeatGrad(xyz_tgt, norm_tgt, feat_tgt, kdt)
    N = xyz_src.shape[0]
    pc_src_temp = pc_src
    resR = np.eye(3)
    resT = np.zeros([3, 1])
    for i in range(20):
        dist, corr_tgt = kdt.query(pc_src_temp.point, k=1, return_distance=True)
        corr_src = np.array(range(N)).reshape(N, 1)
        corr = np.concatenate((corr_src, corr_tgt), axis=1)
        R, t = featGN(pc_src_temp, pc_tgt, corr, grad,0.9)
        resR = R @ resR
        resT = R @ resT + t.reshape(3, 1)
        pc_src_temp.point = (R @ pc_src_temp.point.T + t.reshape(3, 1)).T
    trans = np.concatenate((resR, resT), axis=1)
    pc_src_temp.normal = (resR @ pc_src.normal.T).T
    return trans, pc_src_temp

def fgr(src,tgt):
    pc_src = open3d.geometry.PointCloud()
    pc_src.points = open3d.utility.Vector3dVector(src.point)
    pc_src.normals = open3d.utility.Vector3dVector(src.normal)
    pc_tgt = open3d.geometry.PointCloud()
    pc_tgt.points = open3d.utility.Vector3dVector(tgt.point)
    pc_tgt.normals = open3d.utility.Vector3dVector(tgt.normal)
    feat_src = open3d.registration.compute_fpfh_feature(pc_src, open3d.geometry.KDTreeSearchParamHybrid(radius=0.125,
                                                                                                        max_nn=100))
    feat_tgt = open3d.registration.compute_fpfh_feature(pc_tgt,
                                                        open3d.geometry.KDTreeSearchParamHybrid(radius=0.125,
                                                                                              max_nn=100))

    result = open3d.registration.registration_fast_based_on_feature_matching(
        pc_src, pc_tgt, feat_src, feat_tgt,
        open3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=0.0125))
            
    source_temp = pc_src
    source_temp.transform(result.transformation)
    res_source_temp = PointCloud(np.asarray(source_temp.points), np.asarray(source_temp.normals), None)

    return result.transformation[:3,:],res_source_temp

