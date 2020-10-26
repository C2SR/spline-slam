from matplotlib import pyplot as plt
import numpy as np
import threading
import time

class SLAMPlotter(threading.Thread):
    def __init__(self, slam_map, traj, sensor, **kwargs):
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occupied = kwargs['logodd_max_occupied'] if 'logodd_max_occupied' in kwargs else 100
        sleep_time = kwargs['plot_sleep_time'] if 'plot_sleep_time' in kwargs else 15

        # SLAM
        self.slam_map = slam_map
        self.logodd_min_free = logodd_min_free
        self.logodd_max_occupied = logodd_max_occupied
        self.sleep_time = sleep_time
        
        # Figures
        self.fig, self.ax = plt.subplots()        
        self.fig.tight_layout()

        # Artists
        self.traj_marker = self.ax.plot([],[],  
                                        linestyle='--',
                                        color='b',
                                        linewidth=2.5)[0]
        self.position_est_marker = self.ax.plot([],[], 
                                                marker='o',
                                                markerfacecolor='g', 
                                                markeredgecolor='g', 
                                                markersize=10, 
                                                linestyle='None')[0]
        self.laser_marker = self.ax.plot([],[],
                                         marker='.', 
                                         markerfacecolor='r', 
                                         markeredgecolor='r',
                                         markersize=5, 
                                         linestyle='None')[0]

                                            
        self.traj = traj
        self.sensor = sensor

        dx, dy = .1, .1
        #y, x = np.mgrid[-25:10+dy:dy, -15:25+dx:dx] # INTEL
        y, x = np.mgrid[-10:6.5+dy:dy, -7.5:5+dx:dx] # INTEL
        self.map_pts = np.vstack([x.flatten(), y.flatten()])
        self.map_grid_size = x.shape
        self.x = x
        self.y = y

        self.active = True
        threading.Thread.__init__(self)

    def deactivate(self):
        self.active = False

    def run(self):
        while self.active:
            time.sleep(self.sleep_time)
            self.plot_slam()
    
    def plot_slam(self):
        traj_points = self.traj.get_trajectory()
        map_value = self.slam_map.evaluate_map(self.map_pts).reshape(self.map_grid_size)
        self.ax.pcolormesh( self.x, 
                            self.y, 
                            map_value, 
                            cmap='binary',
                            vmax = self.logodd_max_occupied, 
                            vmin= self.logodd_min_free)
        # Trajectory
        self.traj_marker.set_xdata(traj_points[0,:])
        self.traj_marker.set_ydata(traj_points[1,:])
        # current position
        self.position_est_marker.set_xdata(traj_points[0,-1])
        self.position_est_marker.set_ydata(traj_points[1,-1])        
        # Laser scan 
        pts_occ_local = self.sensor.get_occupied_pts()
        pts_occ = self.slam_map.local_to_global_frame(traj_points[:,-1], pts_occ_local)
        self.laser_marker.set_xdata(pts_occ[0,:])
        self.laser_marker.set_ydata(pts_occ[1,:])
        
        self.ax.axis('equal')
        self.fig.savefig('spline_map' + '.png')





# # Plot
# nb_resolution_plot = 1
# fig, axs = plt.subplots(1, nb_resolution_plot, figsize=(25, 30), dpi=80)
# if nb_resolution_plot == 1:
#     axs=[axs]
# fig.tight_layout()
# #plt.show(block=False)
# #axs = [axs]

# path_marker = []
# prev_position_est_marker = []
# position_est_marker= []
# prev_laser_marker = []
# laser_marker = []

# for res in range(nb_resolution_plot):
#     prev_laser_marker.append(axs[res].plot([],[], marker='*', markerfacecolor='y', markeredgecolor='y', linestyle='None')[0])        
#     prev_position_est_marker.append(axs[res].plot([],[], marker='o', markerfacecolor='c', markeredgecolor='c', markersize=15, linestyle='None')[0])
    
#     path_marker.append(axs[res].plot([],[],  linestyle='--', color='b', linewidth=5)[0])
#     laser_marker.append(axs[res].plot([],[], marker='*', markerfacecolor='r', markeredgecolor='r', linestyle='None')[0])
#     position_est_marker.append(axs[res].plot([],[], marker='o', markerfacecolor='g', markeredgecolor='g', markersize=15, linestyle='None')[0])

# # make these smaller to increase the map resolution
# dx, dy = 0.2, 0.2
# # generate 2 2d grids for the x & y bounds
# #y, x = np.mgrid[-20:30+dy:dy, -10:50+dx:dx] # DAGS
# y, x = np.mgrid[-60:10+dy:dy, -30:60+dx:dx] # ACES
# #y, x = np.mgrid[-25:10+dy:dy, -15:25+dx:dx] # INTEL
# #y, x = np.mgrid[-15:15+dy:dy, -30:25+dx:dx] # FREIBURG
# #y, x = np.mgrid[-25:50+dy:dy, -20:40+dx:dx] # MIT-CSAIL
# #y, x = np.mgrid[-100:150+dy:dy, -220:50+dx:dx] # MIT-KILLIAN

# map_pts = np.vstack([x.flatten(), y.flatten()])
# map_grid_size = x.shape
# frame_counter = 1
# frame_nb = 0



# map_value = multi_res_mapping[res].evaluate_map(map_pts).reshape(map_grid_size)
# map_value = map_value[:-1, :-1]
# axs[res-offset].pcolormesh(x, y, map_value, cmap='binary', vmax = multi_res_mapping[res].logodd_max_occupied/3., vmin= multi_res_mapping[res].logodd_min_free/3.)
# # Update position before optimization
# prev_position_est_marker[res-offset].set_xdata(pose_estimative_init[0])
# prev_position_est_marker[res-offset].set_ydata(pose_estimative_init[1])
# # Sensor data
# ranges_occ, angles_occ = multi_res_mapping[res].remove_spurious_measurements(ranges)
# pts_occ_local = multi_res_mapping[res-offset].range_to_coordinate(ranges_occ, angles_occ)
# pts_occ = multi_res_mapping[res-offset].local_to_global_frame(pose_estimative_init,pts_occ_local)
# prev_laser_marker[res-offset].set_xdata(pts_occ[0,:])
# prev_laser_marker[res-offset].set_ydata(pts_occ[1,:])

# # Update path
# pose = np.copy(multi_res_localization[res].pose)
# position = multi_res_localization[res].pose[0:2]
# path_marker[res-offset].set_xdata(path[0,:])
# path_marker[res-offset].set_ydata(path[1,:])
# #axs[res].plot(position[0], position[1],'.k')
# # Update position
# position_est_marker[res-offset].set_xdata(position[0])
# position_est_marker[res-offset].set_ydata(position[1])
# # Sensor data
# ranges_occ, angles_occ = multi_res_mapping[res].remove_spurious_measurements(ranges)
# pts_occ_local = multi_res_mapping[res].range_to_coordinate(ranges_occ, angles_occ)
# pts_occ = multi_res_mapping[res].local_to_global_frame(pose, pts_occ_local)
# laser_marker[res-offset].set_xdata(pts_occ[0,:])
# laser_marker[res-offset].set_ydata(pts_occ[1,:])

# # Saving
# axs[res-offset].axis('equal')

# plt.savefig('image/' + input_param + '_' + str(frame_nb).zfill(4) + '.png')
# toc = time.time()