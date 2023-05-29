#
# 樋の左右から取得した２点群の合成
# （Open3D 0.12.0 使用: pip install open3d）
#
# 2021-01-12 hshimada@ice.ous.ac.jp
from datetime import datetime
import open3d as o3d
import numpy as np
import sys


def surface_reconstruct(source):
    radius_normal = voxel_size * 2

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    source.orient_normals_consistent_tangent_plane(100)

    print('run Poisson surface reconstruction')

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        source, depth=11, width=0, scale=1.5)
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
    print("Process reconstruct start time: ", datetime.now().strftime("%H:%M:%S"))
    sys.stdin.reconfigure(encoding='utf-8')

    sys.stdout.reconfigure(encoding='utf-8')

    voxel_size = 0.02
    densities_threshold = 0.01

    target_data = sys.argv[1]
    output_data = sys.argv[2]

    target_pcd = o3d.io.read_point_cloud(target_data)

    pcd = surface_reconstruct(target_pcd)
    refine_pcd = surface_reconstruct(pcd)
    refine_pcd = surface_reconstruct(refine_pcd)

    with open(output_data, 'w') as f:
        for i, p in enumerate(refine_pcd.points):
            f.write("%f %f %f %d %d %d\n" % (p[0], p[1], p[2],
                                             int(refine_pcd.colors[i][0] * 255), int(refine_pcd.colors[i][1] * 255),
                                             int(refine_pcd.colors[i][2] * 255)))

    print("Process reconstruct end time: ", datetime.now().strftime("%H:%M:%S"))
