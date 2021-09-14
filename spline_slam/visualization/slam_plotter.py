import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import threading
import time

class SLAMPlotter(threading.Thread):
    def __init__(self, slam_map, traj, sensor, **kwargs):
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occupied = kwargs['logodd_max_occupied'] if 'logodd_max_occupied' in kwargs else 100
        sleep_time = kwargs['plot_sleep_time'] if 'plot_sleep_time' in kwargs else 10

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
        y, x = np.mgrid[-25:20+dy:dy, -15:25+dx:dx] # INTEL
        #y, x = np.mgrid[-10:6.5+dy:dy, -7.5:5+dx:dx] # FEUP
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
