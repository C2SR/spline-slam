import numpy as np
from matplotlib import pyplot as plt
import sys

"""
Compute the transform that takes xi to xj (j_T_i)
"""
def compute_transform(xi, xj):
    theta = xi[2] - xj[2] 
    s, c = np.sin(xj[2]), np.cos(xj[2])
    Rj = np.array([[c, -s],[s, c]])
    translation = np.matmul(Rj.T,  xi[0:2]) - np.matmul(Rj.T,  xj[0:2])

    while theta > np.pi:
        theta -= 2*np.pi
    while theta < -np.pi:
        theta += 2*np.pi

    return np.array([translation[0], translation[1], theta])  

def main():
    if len(sys.argv) < 2:
        print("You must enter a file name")
        sys.exit(-1)

    relations_filename = sys.argv[1]
    slam_filename = sys.argv[2]

    # Opening files
    relations_file_handle = open(relations_filename, "r")
    slam_file_handle = open(slam_filename, "r")
    
    counter = 0.
    miss = 0
    total_angular_error = 0.
    total_translation_error = 0.
    translation_error = np.zeros(2)
    orientation_error = np.zeros(1)    
    for relations_data in relations_file_handle:
        ######### Collecting data from log ##########
        relations_data = np.fromstring(relations_data, dtype=np.float, sep=' ' )
        relations_relative_pose = np.array([relations_data[2], relations_data[3], relations_data[7]])
        timestamp_init = relations_data[0]
        timestamp_final = relations_data[1]

        # Search same time interval in the SLAM file
        pose_init = None 
        pose_final = None
        for slam_data in slam_file_handle:
            slam_data = np.fromstring(slam_data, dtype=np.float, sep=' ' )  
            timestamp = slam_data[0]
            if np.abs(timestamp - timestamp_init) < 1e-3:
                timestamp_init_log = timestamp
                pose_init = np.array([slam_data[1], slam_data[2], slam_data[3]])
            elif np.abs(timestamp - timestamp_final) < 1e-3:
                timestamp_final_log = timestamp
                pose_final = np.array([slam_data[1], slam_data[2], slam_data[3]])
            if pose_init is not None and pose_final is not None:
                break
        slam_file_handle.seek(0)

        # Compute slam relativepose    
        if pose_init is None or pose_final is None:
            continue
        slam_relative_pose = compute_transform(pose_final, pose_init)                    
        relative_pose_error = compute_transform(relations_relative_pose, slam_relative_pose)

        translation_error = np.vstack([translation_error, relative_pose_error[0:2]])
        orientation_error = np.vstack([orientation_error, relative_pose_error[2]])
        counter = counter + 1
        print(miss, counter, timestamp_init_log - timestamp_init, timestamp_final_log - timestamp_final, np.mean(np.linalg.norm(translation_error, axis=1)))            

    print('----------------------------')
    print(np.mean(np.linalg.norm(translation_error, axis=1)), np.std(np.linalg.norm(translation_error, axis=1)))
    print(np.mean(np.linalg.norm(translation_error, axis=1)**2), np.std(np.linalg.norm(translation_error, axis=1)**2))
    print((180./np.pi)*np.mean(np.linalg.norm(orientation_error, axis=1)), np.std(np.linalg.norm((180./np.pi)*orientation_error, axis=1)))
    print(np.mean(np.linalg.norm((180./np.pi)*orientation_error, axis=1)**2), np.std(np.linalg.norm((180./np.pi)*orientation_error, axis=1)**2))

if __name__ == "__main__":
    main()