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


def post_process(point_all, aabb_crop, align_mat, voxel_size):
    # transform with alignment matrix and crop point
    point_all.transform(align_mat)
    point_crop = o3d.geometry.PointCloud.crop(point_all, aabb_crop)

    # ダウンサンプリングしたものを表示
    # 不要ならコメントアウトを。
    point_down = point_crop.voxel_down_sample(voxel_size)

    return point_down


# 結果表示用のヘルパー関数
def draw_unify_result(left, right):
    left_temp = copy.deepcopy(left)
    right_temp = copy.deepcopy(right)
    left_temp.paint_uniform_color([1, 0.706, 0])
    right_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([left_temp, right_temp])


# 点群データの前処理
def preprocess_point_cloud(pcd, voxel_size):

    # voxel_sizeにダウンサンプリング
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 半径radius_normalの点探索による点の法線ベクトル推定
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # 半径radius_featureの点探索によるFPFH特徴の取得
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


# 大域位置合わせ
# RANSAC
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: 大域位置合わせ実行")
    print("   距離しきい値%.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


# 精密位置合わせ
# Point-to-Plane ICP
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, trans):
    distance_threshold = voxel_size * 0.4
    print(":: 精密位置合わせ実行")
    print("   距離しきい値 %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return result

# 位置合わせ
def registration(source, target, source_fpfh, target_fpfh, voxel_size):
    # 大域位置合わせ
    print("Begin registration: ", datetime.now().strftime("%H:%M:%S"))
    result_global = execute_global_registration(source, target,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    # 大域位置合わせ結果の表示
    # fitness: 領域の重複の尺度 (対応点の数 / ターゲット内の点の数)。高いほどフィットしている。
    # inlier_rmse: 全対応点のRMSE(Root Mean Square Error: )。低いほどフィットしている。
    print(result_global)

    # 精密位置合わせ
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size, result_global.transformation)

    # 精密位置合わせ結果の表示
    # fitness: 領域の重複の尺度 (対応点の数 / ターゲット内の点の数)。高いほどフィットしている。
    # inlier_rmse: 全対応点のRMSE(Root Mean Square Error)。低いほどフィットしている。
    print(result_icp)
    print("End registration: ", datetime.now().strftime("%H:%M:%S"))
    # 変換行列を返す
    return result_icp.transformation

###########################################################################
###########################################################################
###########################################################################
# メイン
if __name__ == "__main__":
    print("Process start time: ", datetime.now().strftime("%H:%M:%S"))
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

    # ターゲットデータ（位置合わせの基準とするデータ）
    target_data = "./BasePoint.pcd"
    # 左右データ
    left_data = "./Left.pcd"
    right_data = "./Right.pcd"
    # 出力ファイル名
    output_data = "./Shell.xyz"


    # ダウンサンプリングのボクセルサイズ(m)
    voxel_size = 0.02

    # 上部と下部のBounding Box定義
    aabb_under = o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 0]) # 地下100mから地表まで
    # aabb_above = o3d.geometry.AxisAlignedBoundingBox([-100, -100, 0], [100, 100, 100]) # 地表から上空100mまで
    aabb_above = o3d.geometry.AxisAlignedBoundingBox([-5, 0, 0], [5, 20, 5])

    # 出銑孔より左半分と右半分のBounding Box定義（接合部を中央から左右1mずつ，計2mオーバーラップ）
    aabb_left = o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [1, 100, 100])
    aabb_right = o3d.geometry.AxisAlignedBoundingBox([-1, -100, -100], [100, 100, 100])

    # 点群データ全体の読み込み
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_all_point = [executor.submit(o3d.io.read_point_cloud, cloud_data) for cloud_data in
                            [left_data, right_data, target_data]]

    left_all = future_all_point[0].result()
    right_all = future_all_point[1].result()
    target_all = future_all_point[2].result()

    # ターゲットデータの調整
    # 3BF4TH_20210108001_ZANSEN5_H.pcd の出銑孔をy軸が通るように調整する。
    # ■本来はターゲットデータが正確に位置合わせされているはずなので，この処理は不要。
    # コメントアウトを。

    # --- danhtnt comment ---
    # R = target_all.get_rotation_matrix_from_xyz((0, 0, -1 * np.pi / 180))
    # target_all.rotate(R, center=(0, 0, 0))
    # target_all.translate((0.08, 0, 0.85))
    # --- end comment ---

    # 上部 (0<Z) の点群データ取得
    left_above = o3d.geometry.PointCloud.crop(left_all, aabb_above)
    right_above = o3d.geometry.PointCloud.crop(right_all, aabb_above)
    target_above = o3d.geometry.PointCloud.crop(target_all, aabb_above)

    # voxel_sizeにダウンサンプリング
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_preprocess_align_point = [executor.submit(preprocess_point_cloud, all_point, voxel_size) for all_point in
                                         [left_above, right_above, target_above]]

    left_above_down, left_above_fpfh = future_preprocess_align_point[0].result()
    right_above_down, right_above_fpfh = future_preprocess_align_point[1].result()
    target_above_down, target_above_fpfh = future_preprocess_align_point[2].result()

    # 左右データをターゲットに合わせる変換行列取得
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_registration_mat = [
            executor.submit(registration, left_above_down,
                            target_above_down, left_above_fpfh,
                            target_above_fpfh, voxel_size),
            executor.submit(registration, right_above_down,
                            target_above_down, right_above_fpfh,
                            target_above_fpfh, voxel_size),
        ]
    mat_L = future_registration_mat[0].result()
    mat_R = future_registration_mat[1].result()

    # 左右データをターゲットに合わせる
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_cropped_down = [
            executor.submit(post_process, left_all,
                            aabb_right, mat_L, voxel_size),
            executor.submit(post_process, right_all,
                            aabb_left, mat_R, voxel_size),
        ]
    left_crop_down = future_cropped_down[0].result()
    right_crop_down = future_cropped_down[1].result()

    # Open3Dの点群をnumpyの配列に変換して結合

    # 座標値
    p_all = np.concatenate([
        np.asarray(left_crop_down.points),
        np.asarray(right_crop_down.points)
    ])

    # 色情報
    c = np.concatenate([
        np.asarray(left_crop_down.colors),
        np.asarray(right_crop_down.colors)
    ])

    # 合成された点群をXYZフォーマットでファイルに出力
    with open(output_data, 'w') as f:
        for i, p in enumerate(p_all):
            f.write("%f %f %f %d %d %d\n" % (p[0], p[1], p[2],
                                             int(c[i][0] * 255), int(c[i][1] * 255), int(c[i][2] * 255)))
    print("Process end time: ", datetime.now().strftime("%H:%M:%S"))
