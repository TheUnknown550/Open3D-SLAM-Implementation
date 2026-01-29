import open3d as o3d
import numpy as np
import os
import re

# Function to sort filenames numerically (0, 1, 2, 10, 11...) 
# instead of lexicographically (0, 1, 10, 11, 2...)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def run_calibrated_slam(data_folder):
    # 1. Load and Sort Files Correctly
    file_names = sorted([f for f in os.listdir(data_folder) if f.endswith('.pcd')], 
                        key=natural_sort_key)
    
    if not file_names:
        print(f"No PCD files found in {data_folder}")
        return

    print(f"Starting SLAM with {len(file_names)} frames in correct order...")

    # Initialize Global Map with the first frame
    global_map = o3d.io.read_point_cloud(os.path.join(data_folder, file_names[0]))
    
    # Pre-clean the initial map to remove sensor noise
    global_map, _ = global_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Trajectory tracking
    current_pose = np.eye(4)
    trajectory_points = [current_pose[:3, 3]] 

    # 2. Iterative Registration Loop
    for i in range(1, len(file_names)):
        # Load source (new frame) and target (previous frame only)
        source = o3d.io.read_point_cloud(os.path.join(data_folder, file_names[i]))
        target = o3d.io.read_point_cloud(os.path.join(data_folder, file_names[i-1]))

        # Clean source data
        source, _ = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Pre-processing for ICP
        source_down = source.voxel_down_sample(voxel_size=0.05)
        target_down = target.voxel_down_sample(voxel_size=0.05)
        
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 3. Compute FPFH features for global registration
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        
        # 4. Global registration using RANSAC with FPFH features
        global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True, max_correspondence_distance=0.2,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            ransac_n=3, checkers=[], criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        # 5. Fine alignment with ICP using global registration result as initialization
        reg_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, max_correspondence_distance=0.1,
            init=global_result.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.HuberLoss(k=0.1)
            ),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        
        # 6. Accumulate pose transformation
        current_pose = current_pose @ reg_result.transformation
        trajectory_points.append(current_pose[:3, 3])
        
        # 7. Transform source by accumulated pose and merge with global map
        source.transform(current_pose)
        global_map += source

        # Print progress for every frame
        print(f"Frame {i} aligned successfully (fitness: {reg_result.fitness:.4f}).")

        # Periodic map cleaning to keep it sharp (less frequently)
        if i % 5 == 0:
            global_map = global_map.voxel_down_sample(voxel_size=0.01)

    # 5. Final Output
    global_map = global_map.voxel_down_sample(voxel_size=0.01)
    o3d.io.write_point_cloud("global_map.pcd", global_map)

    # 6. Visualization with Trajectory
    lines = [[i, i+1] for i in range(len(trajectory_points)-1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(trajectory_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color([1, 0, 0]) # Red trajectory line

    print("Opening 3D Classroom Visualization...")
    o3d.visualization.draw_geometries([global_map, line_set], 
                                      window_name="Corrected SLAM Map")

if __name__ == "__main__":
    # Ensure 'data' is the folder name where your .pcd files are kept
    run_calibrated_slam('data')