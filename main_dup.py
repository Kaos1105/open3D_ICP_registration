#
# 樋の左右から取得した２点群の合成
# （Open3D 0.12.0 使用: pip install open3d）
#
# 2021-01-12 hshimada@ice.ous.ac.jp
import concurrent
import contextvars
import functools
from datetime import datetime
import multiprocessing as mp
import open3d as o3d
import numpy as np
import copy
import sys
import asyncio


# 結果表示用のヘルパー関数
def draw_unify_result(left, right):
    left_temp = copy.deepcopy(left)
    right_temp = copy.deepcopy(right)
    left_temp.paint_uniform_color([1, 0.706, 0])
    # left_temp.paint_uniform_color([0, 0.651, 0.929])
    right_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([left_temp, right_temp])


def read_point_cloud_data(data):
    result = o3d.io.read_point_cloud(data)
    return result


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
    distance_threshold = voxel_size * 2
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
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.9))
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
    # result_global = execute_global_registration(source, target,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)

    print("Global registration start at: ", datetime.now().strftime("%H:%M:%S"))
    result_global = execute_global_registration(source, target,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Global registration end at: ", datetime.now().strftime("%H:%M:%S"))

    # 大域位置合わせ結果の表示
    # fitness: 領域の重複の尺度 (対応点の数 / ターゲット内の点の数)。高いほどフィットしている。
    # inlier_rmse: 全対応点のRMSE(Root Mean Square Error: )。低いほどフィットしている。
    print(result_global)

    # 精密位置合わせ
    print("Refine registration start at: ", datetime.now().strftime("%H:%M:%S"))
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size, result_global.transformation)
    print("Refine registration end at: ", datetime.now().strftime("%H:%M:%S"))

    # 精密位置合わせ結果の表示
    # fitness: 領域の重複の尺度 (対応点の数 / ターゲット内の点の数)。高いほどフィットしている。
    # inlier_rmse: 全対応点のRMSE(Root Mean Square Error)。低いほどフィットしている。
    print(result_icp)

    # 変換行列を返す
    return result_icp.transformation


###########################################################################
###########################################################################
###########################################################################
# メイン
if __name__ == "__main__":
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
    # voxel_size = 0.02
    voxel_size = 0.02

    # 上部と下部のBounding Box定義
    aabb_under = o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [100, 100, 0])  # 地下100mから地表まで
    # aabb_above = o3d.geometry.AxisAlignedBoundingBox([-100, -100, 0], [100, 100, 100]) # 地表から上空100mまで
    aabb_above = o3d.geometry.AxisAlignedBoundingBox([-5, 0, 0], [5, 20, 5])

    # 出銑孔より左半分と右半分のBounding Box定義（接合部を中央から左右1mずつ，計2mオーバーラップ）
    aabb_left = o3d.geometry.AxisAlignedBoundingBox([-100, -100, -100], [1, 100, 100])
    aabb_right = o3d.geometry.AxisAlignedBoundingBox([-1, -100, -100], [100, 100, 100])

    # 点群データ全体の読み込み
    print("Read data start time: ", datetime.now().strftime("%H:%M:%S"))
    left_all = o3d.io.read_point_cloud(left_data)
    right_all = o3d.io.read_point_cloud(right_data)
    target_all = o3d.io.read_point_cloud(target_data)
    print("Read data end time: ", datetime.now().strftime("%H:%M:%S"))

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
    print("Preprocess start at: ", datetime.now().strftime("%H:%M:%S"))
    target_above_down, target_above_fpfh = preprocess_point_cloud(target_above, voxel_size)
    left_above_down, left_above_fpfh = preprocess_point_cloud(left_above, voxel_size)
    right_above_down, right_above_fpfh = preprocess_point_cloud(right_above, voxel_size)
    print("Preprocess end at: ", datetime.now().strftime("%H:%M:%S"))

    # 左右データをターゲットに合わせる変換行列取得
    print("Registration start at: ", datetime.now().strftime("%H:%M:%S"))
    mat_L = registration(left_above_down, target_above_down, left_above_fpfh, target_above_fpfh, voxel_size)
    mat_R = registration(right_above_down, target_above_down, right_above_fpfh, target_above_fpfh, voxel_size)
    print("Registration end at: ", datetime.now().strftime("%H:%M:%S"))

    # 左右データをターゲットに合わせる
    left_all.transform(mat_L)
    right_all.transform(mat_R)

    # 左データの右半分取得
    left_crop = o3d.geometry.PointCloud.crop(left_all, aabb_right)

    # 右データの左半分取得
    right_crop = o3d.geometry.PointCloud.crop(right_all, aabb_left)

    # ダウンサンプリングしたものを表示
    # 不要ならコメントアウトを。
    left_crop_down, left_crop_fpfh = preprocess_point_cloud(left_crop, voxel_size)
    right_crop_down, right_crop_fpfh = preprocess_point_cloud(right_crop, voxel_size)

    # draw_unify_result(left_crop_down, right_crop_down)

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
