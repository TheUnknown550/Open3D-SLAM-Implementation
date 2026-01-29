import open3d as o3d
import numpy as np
import os

def run_calibrated_slam(data_folder):
    file_names = sorted([f for f in os.listdir(data_folder) if f.endswith('.pcd')])
    if not file_names: return

    # Frame 0 is our anchor
    global_map = o3d.io.read_point_cloud(os.path.join(data_folder, file_names[0]))
    
    # Pre-clean the initial map
    global_map, _ = global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    current_pose = np.eye(4)
    trajectory = [current_pose[:3, 3]]

    for i in range(1, len(file_names)):
        source = o3d.io.read_point_cloud(os.path.join(data_folder, file_names[i]))
        target = o3d.io.read_point_cloud(os.path.join(data_folder, file_names[i-1]))

        # --- CALIBRATION STEP 1: Cleaning & Normalization ---
        # Remove "floating" noise points that confuse registration
        source, _ = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        source_down = source.voxel_down_sample(0.02)
        target_down = target.voxel_down_sample(0.02)
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # --- CALIBRATION STEP 2: Robust Registration ---
        # We use a Huber Kernel to ignore outliers (like moving objects or reflections)
        # We also increase the max_correspondence_distance to catch faster camera movements
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, 
            max_correspondence_distance=0.1, # Increased from 0.05
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.HuberLoss(k=0.1)
            )
        )

        current_pose = current_pose @ result.transformation
        trajectory.append(current_pose[:3, 3])

        source.transform(current_pose)
        global_map += source

        # --- CALIBRATION STEP 3: Map Maintenance ---
        # Every 5 frames, clean the global map to prevent it from becoming a "ghostly" mess
        if i % 5 == 0:
            global_map = global_map.voxel_down_sample(0.01)
            print(f"Frame {i} processed and cleaned.")

    # Final visualization
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(trajectory),
        lines=o3d.utility.Vector2iVector([[i, i+1] for i in range(len(trajectory)-1)])
    )
    line_set.paint_uniform_color([1, 0, 0])
    
    o3d.visualization.draw_geometries([global_map, line_set])

run_calibrated_slam('data')