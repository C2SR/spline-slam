import numpy as np
from matplotlib import pyplot as plt
import sys


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
    total_angular_error = 0.
    total_translation_error = 0.

    for relations_data in relations_file_handle:
        ######### Collecting data from log ##########
        relations_data = np.fromstring(relations_data, dtype=np.float, sep=' ' )
        timestamp_init = relations_data[0]
        timestamp_final = relations_data[1]

        # Search same time interval in the SLAM file
        for slam_data in slam_file_handle:
            slam_data = np.fromstring(slam_data, dtype=np.float, sep=' ' )  
            timestamp = slam_data[0]
            if timestamp == timestamp_init:
                pose_init = np.array([slam_data[1], slam_data[2], slam_data[3]])
            elif timestamp == timestamp_final:
                pose_final = np.array([slam_data[1], slam_data[2], slam_data[3]])
                break
        slam_file_handle.seek(0)

        # Statistic
        if pose_init is None or pose_final is None:
            continue
        relations_relative_pose = np.array([relations_data[2], relations_data[3], relations_data[7]])
        slam_relative_pose = pose_final - pose_init
        relative_translation_error = np.linalg.norm(slam_relative_pose[0:2]) - np.linalg.norm(relations_relative_pose[0:2])
        while slam_relative_pose[2] > np.pi:
            slam_relative_pose[2] -= 2*np.pi
        while slam_relative_pose[2] < -np.pi:
            slam_relative_pose[2] += 2*np.pi                    
        relative_angular_error = relations_relative_pose[2] - slam_relative_pose[2]

        total_translation_error += np.abs(relative_translation_error)
        total_angular_error += np.abs(relative_angular_error)

        print(counter, relative_translation_error, relative_angular_error, relations_relative_pose[2], slam_relative_pose[2])
        counter = counter + 1
        pose_init = None 
        pose_final = None


    print('----------------------------')
    print(total_translation_error, total_angular_error)                              
    print(total_translation_error/counter, total_angular_error/counter)    
if __name__ == "__main__":
    main()