from gtlreg import PointCloud,gtlReg
import open3d
import os
import matplotlib.pyplot as plt
import numpy as np


def preprocess():
    if os.path.exists('gt.npy'):
        return
    file = './gt.txt'
    with open(file, 'r') as f:
        count = 0
        gt_pose = []
        gt_poses = []
        for data in f.readlines():
            if count % 5 == 0:
                if len(gt_pose):
                    gt_pose_inv = np.array(gt_pose)
                    gt_R = gt_pose_inv[:3,:3].T
                    gt_T = -gt_R@gt_pose_inv[:3,3].reshape(3,1)
                    gt_pose = np.concatenate((gt_R,gt_T),axis=1)
                    gt_poses.append(gt_pose)
                gt_pose = []
            else:
                temp = data.split()
                temp = [float(x) for x in temp]
                gt_pose.append(temp)
            if count == 252 * 5:
                break
            count = (count + 1)
        np.save('gt.npy', gt_poses)
    f.close()

def odometry(data,start_id=None,end_id=None):
    poses = []

    # it depends on the length of given fragment sequences
    if start_id is None:
        start_id = 1
    if end_id is None:
        end_id = 250

    for id in range(start_id,end_id):
        src_id, tgt_id = id,id+1
        data1, data2 = data[src_id], data[tgt_id]
        source = PointCloud(data1[:, :3], data1[:, 3:6], data1[:, 6:])
        target = PointCloud(data2[:, :3], data2[:, 3:6], data2[:, 6:])
        pose, _ = gtlReg(source, target)
        poses.append(pose)
    poses_ = np.stack(poses,axis=0)
    return poses_

if __name__ == '__main__':

    preprocess()
    gts = np.load('gt.npy')
    data = np.load('data.npy', allow_pickle=True)
    startID, endID = 15, 35
    gt = gts[startID:endID]
    pre = odometry(data, startID, endID)
    pre_ = pre.reshape(-1, 8)
    np.save('pose_' + str(startID) + '_' + str(endID) + '.npy', pre_)
    errR = np.arccos(
        (np.trace(pre[:, :, :3].transpose(0, 2, 1) @ gt[:, :, :3], axis1=1, axis2=2) - 1) / 2) / np.pi * 180
    errT = np.mean(np.abs(pre[:, :, 3] - gt[:, :, 3]).reshape(-1, 3), axis=1)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(errR)), errR)

    plt.subplot(2, 1, 2)
    plt.scatter(range(len(errT)), errT)
    plt.show()




