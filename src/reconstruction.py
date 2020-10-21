import open3d as o3d
import  numpy as np
import  os,sys

def getPose(filename,method,startID,endID):
    poses = np.loadtxt(filename,delimiter='\t')
    if method == 'gtl':
        k = 2
    elif method == 'gt':
        k = 1
    elif method == 'fgr':
        k = 3
    resPose = poses[(startID-1)*3:(endID-1)*3,(k-1)*4:k*4]

    return resPose

def build_pose_graph(filename,startID,endID,method):
    poses = getPose(filename,method,startID,endID)
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    for id_ in range(startID,endID):
            source_id = id_ - startID
            pose = poses[source_id*3:(source_id+1)*3,:]
            pose = np.concatenate((pose,np.array([[0,0,0,1]])),axis=0)
            if method == 'fgr':
                data = np.load('data.npy',allow_pickle=True)
                pc_src = o3d.geometry.PointCloud()
                pc_src.points = o3d.utility.Vector3dVector(data[id_+1][:,:3])
                pc_tgt = o3d.geometry.PointCloud()
                pc_tgt.points = o3d.utility.Vector3dVector(data[id_+2][:,:3])
                result = o3d.registration.registration_icp(pc_src,pc_tgt,max_correspondence_distance=0.05,init=pose)
                pose = result.transformation
            odometry = np.dot(pose,odometry)
            pose_graph.nodes.append(
                o3d.registration.PoseGraphNode(
                    np.linalg.inv(odometry)))
            # pose_graph.edges.append(
            #     o3d.registration.PoseGraphEdge(source_id,
            #                                              target_id,
            #                                              transformation_icp,
            #                                              information_icp,
            #                                              uncertain=False))
    return pose_graph

def reconstruct(file_root,pose_txt,startID,endID,method):
    pcds = []
    for id in range(startID,endID):
        filename = os.path.join(file_root,'mesh_' + str(id) + '.ply')
        pcds.append(o3d.io.read_point_cloud(filename))
    pose_graph = build_pose_graph(pose_txt,startID,endID,method)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    return pcd_combined

if __name__ == '__main__':
    file_root = './'
    pose_txt = './pose1_250.txt'
    startID,endID = 6,8
    # pcd_combined = reconstruct(file_root,pose_txt,startID,endID,'gt')
    # o3d.io.write_point_cloud('merge_gt.ply',pcd_combined,write_ascii=True)
    # pcd_combined = reconstruct(file_root,pose_txt,startID,endID,'gtl')
    # o3d.io.write_point_cloud('merge_ours.ply',pcd_combined,write_ascii=True)
    pcd_combined = reconstruct(file_root,pose_txt,startID,endID,'fgr')
    o3d.io.write_point_cloud('merge_fgr.ply',pcd_combined,write_ascii=True)
    o3d.visualization.draw_geometries([pcd_combined])

