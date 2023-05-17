#
# 樋の左右から取得した２点群の合成
# （Open3D 0.12.0 使用: pip install open3d）
#
# 2021-01-12 hshimada@ice.ous.ac.jp
import concurrent
from datetime import datetime
import open3d as o3d
import numpy as np
import copy
import sys
import asyncio

###########################################################################
###########################################################################
###########################################################################
# メイン
if __name__ == "__main__":
    print("Process start time: ", datetime.now().strftime("%H:%M:%S"))
    sys.stdin.reconfigure(encoding='utf-8')

    sys.stdout.reconfigure(encoding='utf-8')


    # ダウンサンプリングのボクセルサイズ(m)
    voxel_size = 0.02

    target_data = "./ShellTOICUbe.xyz"
    target_pcd = o3d.io.read_point_cloud(target_data)
    target_pcd.paint_uniform_color([1, 0.706, 0])

    # radius_normal = 0.01
    # target_pcd.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # target_pcd.orient_normals_consistent_tangent_plane(100)
    # # o3d.visualization.draw_geometries([target_pcd], point_show_normal=True)
    # print('run Poisson surface reconstruction')
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #         target_pcd, depth=9)
    # # o3d.visualization.draw_geometries([mesh])
    # densities = np.asarray(densities)
    # print('remove low density vertices')
    # vertices_to_remove = densities < np.quantile(densities, 0.01)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    # print(mesh)
    o3d.visualization.draw_geometries([target_pcd])
