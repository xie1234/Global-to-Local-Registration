from Reg import *
import open3d
import  os
import  matplotlib.pyplot as plt

import timeit


def odometry(data,start_id,end_id):
    poses = []
    time_FGR = 0
    time_GTL = 0
    for id in range(start_id,end_id):
        src_id, tgt_id = id,id+1
        data1, data2 = data[src_id], data[tgt_id]
        source = PointCloud(data1[:, :3], data1[:, 3:6], data1[:, 6:])
        target = PointCloud(data2[:, :3], data2[:, 3:6], data2[:, 6:])
        start = timeit.default_timer()
        transGTL, source_temp = gtlReg(source, target)
        end = timeit.default_timer()
        time_GTL += (end-start)
        source = PointCloud(data1[:, :3], data1[:, 3:6], data1[:, 6:])
        start = timeit.default_timer()
        transFGR, source_temp = fgr(source, target)
        end = timeit.default_timer()
        time_FGR += (end - start)
        pose = np.concatenate((transGTL,transFGR),axis=1)
        poses.append(pose)
    print(time_GTL,time_FGR)
    poses_ = np.stack(poses,axis=0)
    return poses_
def preprocess():
    if os.path.exists('gt.py'):
        return
    file = 'C:\\Users\\14291\\Desktop\\reg_output.txt'
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
        np.save('gt', gt_poses)
    f.close()

if __name__ == '__main__':

    preprocess()
    gts = np.load('gt.npy')
    data = np.load('data.npy', allow_pickle=True)
    stardID,endID = 1,250

    gt = gts[1:250]
    if os.path.exists('error' + str(1) + '_' + str(250) + '.txt'):
        error = np.loadtxt('error' + str(1) + '_' + str(250) + '.txt',delimiter='\t')
        errR_gtl = error[:,0]
        errT_gtl = error[:,1]
        errR_fgr = error[:, 2]
        errT_fgr = error[:, 3]
    else:
        if os.path.exists('pose'+str(1)+'_'+str(250)+'.txt'):
            pose = np.loadtxt('pose'+str(1)+'_'+str(250)+'.txt',delimiter='\t')
            pre_gtl_ = pose[:,4:8]
            pre_fgr_ = pose[:,8:12]
            pre_gtl = pre_gtl_.reshape(-1,3,4)
            pre_fgr = pre_fgr_.reshape(-1,3,4)
        else:
            pre = odometry(data, 1, 250)
            pre_gtl = pre[...,0:4]
            pre_fgr = pre[...,4:8]
            pre_ = pre.reshape(-1, 8)
            gt_ = gt.reshape(-1, 4)
            pose = np.concatenate((gt_, pre_), axis=1)
            np.savetxt('pose' + str(1) + '_' + str(250) + '.txt', pose, delimiter='\t')
        errR_gtl = np.arccos(
            (np.trace(pre_gtl[:, :, :3].transpose(0, 2, 1) @ gt[:, :, :3], axis1=1, axis2=2) - 1) / 2) / np.pi * 180
        errT_gtl = np.mean(np.abs(pre_gtl[:, :, 3] - gt[:, :, 3]).reshape(-1, 3), axis=1)
        errR_fgr = np.arccos(
            (np.trace(pre_fgr[:, :, :3].transpose(0, 2, 1) @ gt[:, :, :3], axis1=1, axis2=2) - 1) / 2) / np.pi * 180
        errT_fgr = np.mean(np.abs(pre_fgr[:, :, 3] - gt[:, :, 3]).reshape(-1, 3), axis=1)
        error = np.concatenate((errR_gtl.reshape(-1, 1), errT_gtl.reshape(-1, 1),errR_fgr.reshape(-1, 1), errT_fgr.reshape(-1, 1)), axis=-1)
        np.savetxt('error' + str(1) + '_' + str(250) + '.txt', error, delimiter='\t')
    temp_error_R_fgr = errR_fgr[stardID-1:endID-1]
    temp_error_T_fgr = errT_fgr[stardID-1:endID-1]
    temp_error_R_gtl = errR_gtl[stardID - 1:endID - 1]
    temp_error_T_gtl = errT_gtl[stardID - 1:endID - 1]

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.scatter(range(len(temp_error_R_gtl)),temp_error_R_gtl)
    plt.scatter(range(len(temp_error_R_fgr)), temp_error_R_fgr)
    plt.subplot(2,1,2)
    plt.scatter(range(len(temp_error_T_gtl)),temp_error_T_gtl)
    plt.scatter(range(len(temp_error_T_fgr)), temp_error_T_fgr)
    plt.show()

