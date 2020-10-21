import open3d
import  numpy as np
from sklearn.neighbors import KDTree
import copy
import  os,sys
import  h5py

from featureICP import PointCloud,calGrad,calTrans
from icp import  icp_mcc

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

def initialTrans(source,target):
    kdt_feat = KDTree(target.feature, metric='euclidean')
    dist, corr = kdt_feat.query(source.feature, k=1, return_distance=True)

    idx = np.where(dist < 0.4)[0]

    corrSrc = np.array(range(dist.shape[0]))[idx].reshape(-1,1)
    corrTgt = corr[idx,:].reshape(-1,1)

    R,t = svd(source.point[corrSrc,:].reshape(-1,3),target.point[corrTgt,:].reshape(-1,3))

    # source_feat = open3d.registration.Feature()
    # source_feat.data = source.feature[corrSrc,:].reshape(32,-1)
    #
    # target_feat = open3d.registration.Feature()
    #
    # target_feat.data = target.feature[corrTgt,:].reshape(32,-1)
    #
    # pc_src = open3d.geometry.PointCloud()
    # pc_src.points = open3d.utility.Vector3dVector(source.point[corrSrc,:].reshape(-1,3))
    # pc_src.paint_uniform_color([1, 0.706, 0])
    #
    # pc_tgt = open3d.geometry.PointCloud()
    # pc_tgt.points = open3d.utility.Vector3dVector(target.point[corrTgt,:].reshape(-1,3))
    # pc_tgt.paint_uniform_color([0, 0.65, 0.929])
    #
    # result = open3d.registration.registration_ransac_based_on_feature_matching(
    #     pc_src , pc_tgt, source_feat, target_feat, 0.05,
    #     open3d.registration.TransformationEstimationPointToPoint(False), 4,
    #     [open3d.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
    #     open3d.registration.RANSACConvergenceCriteria(80000, 1000))
    #
    # pc_temp = result.transformation[:3,:3] @ source.point.T + result.transformation[:3,3].reshape(3,1)


    # source_temp = source
    # source_temp.point = pc_temp.T
    # normal_temp = result.transformation[:3,:3] @ source.normal.T
    # source_temp.normal = normal_temp.T

    pc_temp = R @ source.point.T + t.reshape(3, 1)
    normal_temp = R @ source.normal.T
    source_temp = source
    source_temp.point = pc_temp.T
    source_temp.normal = normal_temp.T

    return source_temp

def icp(source,target):
    kdt = KDTree(target.point, metric='euclidean')
    grad = calGrad(target.point, target.normal, target.feature, kdt)
    # kdt_feat = KDTree(target.feature, metric='euclidean')
    source_temp = source
    for i in range(20):
        dist, corr = kdt.query(source_temp.point, k=1, return_distance=True)
        # _, idx = kdt_feat.query(source_temp.feature, k=1, return_distance=True)

        # corrSrc = np.array(range(source.point.shape[0])).reshape(-1,1)
        # corr = np.concatenate((corrSrc,np.array(corr).reshape(-1,1)),axis=1)

        idx = np.where(dist < 0.1)[0]
        corrSrc = np.array(range(dist.shape[0]))[idx].reshape(-1, 1)
        corrTgt = corr[idx, :].reshape(-1, 1)
        corr = np.concatenate((corrSrc, corrTgt), axis=1)

        R,t = calTrans(source_temp,target,corr,grad)
        PP = R @ source_temp.point.T + t
        source_temp.point = PP.T

    return source_temp

def fgr(src,tgt):
    pc_src = open3d.geometry.PointCloud()
    pc_src.points = open3d.utility.Vector3dVector(src.point)
    pc_src.normals = open3d.utility.Vector3dVector(src.normal)
    pc_tgt = open3d.geometry.PointCloud()
    pc_tgt.points = open3d.utility.Vector3dVector(tgt.point)
    pc_tgt.normals = open3d.utility.Vector3dVector(tgt.normal)
    feat_src = open3d.registration.compute_fpfh_feature(pc_src, open3d.geometry.KDTreeSearchParamHybrid(radius=0.125, max_nn=100))
    feat_tgt = open3d.registration.compute_fpfh_feature(pc_tgt,
                                                        open3d.geometry.KDTreeSearchParamHybrid(radius=0.125, max_nn=100))
    result = open3d.registration.registration_fast_based_on_feature_matching(
                pc_src, pc_tgt, feat_src, feat_tgt,
                open3d.registration.FastGlobalRegistrationOption(
                                maximum_correspondence_distance=0.0125))

    # result = open3d.registration.registration_ransac_based_on_feature_matching(
    #     pc_src, pc_tgt, feat_src, feat_tgt, 0.025*1.5,
    #         open3d.registration.TransformationEstimationPointToPoint(False), 4, [
    #         # open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #         open3d.registration.CorrespondenceCheckerBasedOnDistance(
    #             0.025*1.5)
    #     ], open3d.registration.RANSACConvergenceCriteria(50000, 500))

    source_temp = copy.deepcopy(pc_src)
    source_temp.transform(result.transformation)
    res_source_temp = PointCloud(np.asarray(source_temp.points), np.asarray(source_temp.normals), None)
    print(result.transformation)
    return res_source_temp
if __name__ == '__main__':
    # f1 = h5py.File('demo1.h5', 'r')
    # pc1 = f1['xyz'][0, :, :]
    # normal1 = f1['normal'][0, :, :]
    # feat1 = f1['color'][0, :, :]
    #
    # f2 = h5py.File('demo2.h5', 'r')
    # pc2 = f2['xyz'][0, :, :]
    # normal2 = f2['normal'][0, :, :]
    # feat2 = f2['color'][0, :, :]


    # f = h5py.File('xyz_normal_feature_color.h5', 'r')
    # pc1 = f['xyz_src'][0, :, :]
    # normal1 = f['norm_src'][0, :, :]
    # feat1 = f['color_src'][0, :, :]
    #
    #
    # pc2 = f['xyz_ref'][0, :, :]
    # normal2 = f['norm_ref'][0, :, :]
    # feat2 = f['color_ref'][0, :, :]

    # f = h5py.File('xyz_normal_fcgf_feature.h5', 'r')
    # pc1 = np.array(f['xyz_src'])
    # normal1 = np.array(f['norm_src'])
    # feat1 = np.array(f['feat_src'])
    #
    # pc2 = np.array(f['xyz_ref'])
    # normal2 = np.array(f['norm_ref'])
    # feat2 = np.array(f['feat_ref'])

    data = np.load('data.npy',allow_pickle=True)
    src_id,tgt_id = 8,10

    # 8,10  ---- 20,21
    pc1 = data[src_id][:,:3]
    normal1 = data[src_id][:,3:6]
    feat1 = data[src_id][:,6:]
    pc2 = data[tgt_id][:, :3]
    normal2 = data[tgt_id][:, 3:6]
    feat2 = data[tgt_id][:, 6:]

    source = PointCloud(pc1, normal1, feat1)
    target = PointCloud(pc2, normal2, feat2)

    # source_temp = icp(source,target)
    source_temp = source

    source_temp = fgr(source_temp,target)

    # source_temp = initialTrans(source,target)
    # source_temp = icp(source_temp, target)
    # R,t = icp_mcc(source_temp.point,target.point,criterion ='LS')
    # source_temp.point = (R @ source_temp.point.T + t).T



    pc_src = open3d.geometry.PointCloud()
    pc_src.points = open3d.utility.Vector3dVector(source_temp.point)
    # pc_src.colors = open3d.utility.Vector3dVector(feat1)
    pc_src.paint_uniform_color([1, 0.706, 0])


    pc_tgt = open3d.geometry.PointCloud()
    pc_tgt.points = open3d.utility.Vector3dVector(target.point)
    pc_tgt.paint_uniform_color([0, 0.65, 0.929])
    # pc_tgt.colors = open3d.utility.Vector3dVector(feat2)
    open3d.visualization.draw_geometries([pc_src, pc_tgt])




