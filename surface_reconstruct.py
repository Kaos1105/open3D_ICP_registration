#
# 樋の左右から取得した２点群の合成
# （Open3D 0.12.0 使用: pip install open3d）
#
# 2021-01-12 hshimada@ice.ous.ac.jp
import concurrent
from datetime import datetime
import open3d as o3d
import numpy as np
import sys


def surface_reconstruct(source, densities_threshold):
    radius_normal = voxel_size * 2

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    source.orient_normals_consistent_tangent_plane(100)

    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            source, depth=9)

    densities = np.asarray(densities)
    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, densities_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    result = o3d.geometry.PointCloud()
    result.points = mesh.vertices
    result.colors = mesh.vertex_colors
    result.normals = mesh.vertex_normals

    return result


###########################################################################
###########################################################################
###########################################################################
# メイン
if __name__ == "__main__":
    print("Process start time: ", datetime.now().strftime("%H:%M:%S"))
    sys.stdin.reconfigure(encoding='utf-8')

    sys.stdout.reconfigure(encoding='utf-8')

    voxel_size = 0.02

    target_data = "./ShellTOICUbe.xyz"
    target_pcd = o3d.io.read_point_cloud(target_data)

    pcd = surface_reconstruct(target_pcd, voxel_size)
    refine_pcd = surface_reconstruct(pcd, voxel_size)
    refine_pcd = surface_reconstruct(refine_pcd, voxel_size)
    o3d.io.write_point_cloud("surface_reconstruct.xyz", refine_pcd)

    print("Process end time: ", datetime.now().strftime("%H:%M:%S"))
