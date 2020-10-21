from Reg import *
import open3d
import  os

def loadAndWrite(data,idList,filename):
    L = len(idList)
    resList = []
    for i in range(L):
        resIlist = []
        src_id, tgt_id = idList[i][0],idList[i][1]
        data1, data2 = data[src_id], data[tgt_id]
        source = PointCloud(data1[:, :3], data1[:, 3:6], data1[:, 6:])
        target = PointCloud(data2[:, :3], data2[:, 3:6], data2[:, 6:])

        pc_src_init = open3d.geometry.PointCloud()
        pc_src_init.points = open3d.utility.Vector3dVector(data1[:, :3])
        pc_src_init.paint_uniform_color([0, 1, 0])

        resIlist.append(np.array([src_id, tgt_id]))
        trans, source_temp = gtlReg(source, target)

        # resIlist.append(trans)
        # trans, source_temp = fgr(source, target)

        pc_src = open3d.geometry.PointCloud()
        pc_src.points = open3d.utility.Vector3dVector(source_temp.point)
        pc_src.paint_uniform_color([1, 0.706, 0])

        pc_tgt = open3d.geometry.PointCloud()
        pc_tgt.points = open3d.utility.Vector3dVector(target.point)
        pc_tgt.paint_uniform_color([0, 0.65, 0.929])



        open3d.visualization.draw_geometries([pc_src,pc_tgt])
        return trans
        resIlist.append(np.array([src_id, tgt_id]))
        resIlist.append(trans)

        resList.append(resIlist)


    np.save(filename, np.array(resList, dtype=object))
if __name__ == '__main__':
    data = np.load('data.npy', allow_pickle=True)
    a,b = 40,41
    idlist = [[a,b]]

    trans = loadAndWrite(data,idlist,'res')


    #
    root_dir = 'C:\\Users\\14291\\Desktop'
    pc_src = open3d.io.read_point_cloud(os.path.join(root_dir,'mesh_'+str(a)+'.ply'))
    pc_tgt = open3d.io.read_point_cloud(os.path.join(root_dir,'mesh_'+str(b)+'.ply'))
    #
    #
    # # pc_src.transform(np.array(
    # #     [[ 0.99597474, -0.03001128,  0.08446094,  0.11958821],
    # #     [ 0.0488293 ,  0.97185208 ,-0.23047611 ,-0.40720948],
    # #     [-0.07516666 , 0.23367255 , 0.96940555 ,-0.5140173 ],
    # #      [0,0,0,1]]
    # # ))

    pc_src.transform(np.array(np.concatenate((trans,np.array([[0, 0, 0, 1]]).reshape(1,-1)),axis=0)))
    # pc_src.colors = open3d.utility.Vector3dVector(np.array([[1,0,0]]))
    # pc_tgt.colors = open3d.utility.Vector3dVector(np.array([[0,1, 0]]))
    # pc_src.paint_uniform_color([1, 0.706, 0])
    # pc_tgt.paint_uniform_color([0, 0.65, 0.929])
    # pc_src.paint_uniform_color([0.8, 0.8, 0.8])
    # pc_tgt.paint_uniform_color([0.8, 0.8, 0.8])
    # pc_tgt = pc_tgt.voxel_down_sample(voxel_size=0.015)
    # pc_src = pc_src.voxel_down_sample(voxel_size=0.015)
    # visual = open3d.visualization.Visualizer()
    # visual.add_geometry(pc_src)
    open3d.visualization.draw_geometries([pc_src,pc_tgt])
