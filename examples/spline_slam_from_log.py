import numpy as np
from matplotlib import pyplot as plt

from spline_slam.spline import SplineLocalization
from spline_slam.spline import SplineMap

import sys
import time

def main():
    if len(sys.argv) < 2:
        print("You must enter a file name")
        sys.exit(-1)

    file_name = sys.argv[1]
    input_param = sys.argv[2]#int(sys.argv[2])

    # Instantiating the grid map object
    multi_res_localization = {}
    multi_res_mapping = {}
    nb_resolution = 3
    for res in range(0,nb_resolution):
        kwargs_spline= {'knot_space': .05*(2**(nb_resolution-res-1)), 
                        'map_size': np.array([200.,200.]),
                        'min_angle': -90*np.pi/180., # -130*np.pi/180,
                        'max_angle': 90*np.pi/180., #129.75*np.pi/180,
                        'angle_increment': 1*np.pi/360., #.25*np.pi/180,
                        'range_min': 0.1,
                        'range_max': 50.0, 
                        'logodd_occupied': 1., #logodd_occupied[res],#1 - .2*(nb_resolution - res - 1), #./(2**(nb_resolution - res - 1)),
                        'logodd_free': .1, #logodd_free[res], #.1,
                        'logodd_min_free': -25.,
                        'logodd_max_occupied': 25., 
                        'nb_iteration_max': 25,
                        'max_nb_rays': 360,
                        'alpha': 1}

        multi_res_localization[res] = SplineLocalization(**kwargs_spline)
        multi_res_mapping[res] = SplineMap(**kwargs_spline)
    # Opening log file
    file_handle = open(file_name, "r")
    # Retrieving sensor parameters
    data = file_handle.readline()  
    data = np.fromstring( data, dtype=np.float, sep=' ' )
    
    # Plot
    fig, axs = plt.subplots(1, nb_resolution, figsize=(25, 30), dpi=80)
    #axs=[axs]
    fig.tight_layout()
    #plt.show(block=False)
    #axs = [axs]
    #path_marker = axs.plot([],[], linestyle='-', color='k')[0]
    prev_position_est_marker = []
    position_est_marker= []
    prev_laser_marker = []
    laser_marker = []

    for res in range(nb_resolution):
        prev_laser_marker.append(axs[res].plot([],[], marker='.', markerfacecolor='y', markeredgecolor='y', linestyle='None')[0])        
        prev_position_est_marker.append(axs[res].plot([],[], marker='o', markerfacecolor='g', markeredgecolor='g', markersize=10, linestyle='None')[0])

        laser_marker.append(axs[res].plot([],[], marker='*', markerfacecolor='r', markeredgecolor='r', linestyle='None')[0])
        position_est_marker.append(axs[res].plot([],[], marker='o', markerfacecolor='b', markeredgecolor='b', markersize=10, linestyle='None')[0])


    # make these smaller to increase the map resolution
    dx, dy = 0.5, 0.5
    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[-50:40+dy:dy, -55:40+dx:dx] # intel
    #y, x = np.mgrid[-12.5:5+dy:dy, -5:5+dx:dx]
    map_pts = np.vstack([x.flatten(), y.flatten()])
    map_grid_size = x.shape
    frame_counter = 1
    frame_nb = 0

    previous_pose = np.array(multi_res_localization[nb_resolution-1].pose)    
    previous_timestamp = 0

    for num, data in enumerate(file_handle, start=1):
        #print(num)
        ######### Collecting data from log ##########
        data = np.fromstring( data, dtype=np.float, sep=' ' )
        timestamp = data[0]
        pose = data[1:4]
        ranges = data[7:]

        for res in range(0, nb_resolution):
            if num < 2:
                multi_res_localization[res].pose = np.array([0,0,0])
                path = np.copy(pose[0:2]).reshape(2,1)
            else:
                if res==0:
                   pose_estimative = np.copy(multi_res_localization[nb_resolution-1].pose) 
                else:
                   pose_estimative = np.copy(multi_res_localization[res-1].pose)
                # Update position
                prev_position_est_marker[res].set_xdata(pose_estimative[0])
                prev_position_est_marker[res].set_ydata(pose_estimative[1])
                # Sensor data
                ranges_occ, angles_occ = multi_res_mapping[res].remove_spurious_measurements(ranges)
                pts_occ_local = multi_res_mapping[res].range_to_coordinate(ranges_occ, angles_occ)
                pts_occ = multi_res_mapping[res].local_to_global_frame(pose_estimative,pts_occ_local)
                prev_laser_marker[res].set_xdata(pts_occ[0,:])
                prev_laser_marker[res].set_ydata(pts_occ[1,:])

                multi_res_localization[res].update_localization(multi_res_mapping[res], ranges, pose_estimative)


        ############# Mapping ################
        for res in range(0, nb_resolution):
            multi_res_mapping[res].update_map(multi_res_localization[nb_resolution-1].pose, ranges)

        ########### Statistics ###########
        mapping_time = 0
        localization_time = 0
        for res in range(0, nb_resolution):
            mapping_time += np.sum(multi_res_mapping[res].time)
            localization_time += np.sum(multi_res_localization[res].time)
  
        pose = np.array(multi_res_localization[nb_resolution-1].pose)
        print(timestamp, pose[0], pose[1], pose[2])
        # print( num, 
        #    1./(mapping_time/num), 
        #    1./(localization_time/num), 
        #    1./(mapping_time/num + localization_time/num) ) 

        ########## Plotting #################
        if (frame_counter > 50  and num > 100020 and num < 110000) or frame_counter>500:          
            for res in range(0, nb_resolution):
                map_value = multi_res_mapping[res].evaluate_map(map_pts).reshape(map_grid_size)
                map_value = map_value[:-1, :-1]
                axs[res].pcolormesh(x, y, map_value, cmap='binary', vmax = multi_res_mapping[res].logodd_max_occupied/1, vmin= multi_res_mapping[res].logodd_min_free/1)
                # Update path
                pose = np.copy(multi_res_localization[res].pose)
                position = multi_res_localization[res].pose[0:2]
                #print(pose)
                ##path = np.hstack([path, position.reshape(2,1)])
                #path_marker.set_xdata(path[0,:])
                #path_marker.set_ydata(path[1,:])
                #axs[res].plot(position[0], position[1],'.k')
                # Update position
                position_est_marker[res].set_xdata(position[0])
                position_est_marker[res].set_ydata(position[1])
                # Sensor data
                ranges_occ, angles_occ = multi_res_mapping[res].remove_spurious_measurements(ranges)
                pts_occ_local = multi_res_mapping[res].range_to_coordinate(ranges_occ, angles_occ)
                pts_occ = multi_res_mapping[res].local_to_global_frame(pose, pts_occ_local)
                laser_marker[res].set_xdata(pts_occ[0,:])
                laser_marker[res].set_ydata(pts_occ[1,:])

                    # Saving
                axs[res].axis('equal')

                plt.savefig('image/' + input_param + '_' + str(num).zfill(4) + '.png')
                toc = time.time()
            frame_counter = 0
            frame_nb +=1


        ########## Counters #################
        frame_counter += 1

        
 
if __name__ == '__main__':
    main()
   #plt.show()
