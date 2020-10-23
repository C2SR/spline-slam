import numpy as np

from spline_slam.core import SplineLocalization
from spline_slam.core import SplineMap
from spline_slam.spline import Odometry
from spline_slam.spline import SplinePlot
from spline_slam.basics import CubicSplineSurface
from spline_slam.trajectory import DiscreteTrajectory

import sys
import time

def main():
    if len(sys.argv) < 2:
        print("You must enter a file name")
        sys.exit(-1)

    file_name = sys.argv[1]
    input_param = sys.argv[2] #int(sys.argv[2])

    # Instantiating the grid map object
    multi_res_localization = {}
    multi_res_mapping = {}
    multi_res_map = {}
    nb_resolution = 3
    
    for res in range(0,nb_resolution):
        max_nb_rays = 60*(res+1)
        kwargs_spline= {'knot_space': .05*((2.5)**(nb_resolution-res-1)), #2.5 
                        'surface_size': np.array([150.,150.]),
                        'min_angle': -180*np.pi/180., # -(130-5)*np.pi/180,
                        'max_angle': 180*np.pi/180., #(129.75-5)*np.pi/180,
                        'angle_increment': 1.*np.pi/180., #.25*np.pi/180,
                        'range_min': 0.1,
                        'range_max': 49.9, #49.9, 
                        'logodd_occupied': 1., #logodd_occupied[res],#1 - .2*(nb_resolution - res - 1), #./(2**(nb_resolution - res - 1)),
                        'logodd_free': .1, #logodd_free[res], #.1,
                        'logodd_min_free': -25.,
                        'logodd_max_occupied': 25., 
                        'nb_iteration_max': 50,
                        'max_nb_rays': max_nb_rays,
                        'alpha': 1}
        
        multi_res_map[res] = CubicSplineSurface(**kwargs_spline)
        multi_res_localization[res] = SplineLocalization(multi_res_map[res], **kwargs_spline)
        multi_res_mapping[res] = SplineMap(multi_res_map[res], **kwargs_spline)

    # Odometry (Class)
    odometry = Odometry()

    # Trajectory (Class)
    traj = DiscreteTrajectory()

    # Plot
    plot_thread = SplinePlot(multi_res_mapping[nb_resolution-1], traj, **kwargs_spline)
    plot_thread.start()

    # Opening log file
    file_handle = open(file_name, "r")
    # Retrieving sensor parameters
    data = file_handle.readline()  
    data = np.fromstring( data, dtype=np.float, sep=' ' )
    
    is_odometry_poor = False
    for num, data in enumerate(file_handle, start=1):
        ######### Collecting data from log ##########
        data = np.fromstring( data, dtype=np.float, sep=' ' )
        timestamp = data[0]
        pose = data[1:4]
        ranges = data[4:]
        
        ########### Localization #############
        for res in range(0, nb_resolution):
            if num < 3:
                odom, dt = odometry.pose_to_discrete_odometry(timestamp, pose) 
            else:
                ###### Odometry ######
                if res==0:
                    odom, dt = odometry.pose_to_discrete_odometry(timestamp, pose)
                    pose_estimative = odometry.update(multi_res_localization[nb_resolution-1].pose, odom)
                    if dt > 1.:
                        is_odometry_poor = True
                else:
                    pose_estimative = np.copy(multi_res_localization[res-1].pose)                
                ###### Scan matching ######
                multi_res_localization[res].update_localization(ranges, pose_estimative, is_odometry_poor)
                is_odometry_poor = False
    
        ############# Mapping ################
        for res in range(0, nb_resolution):
            multi_res_mapping[res].update_map(multi_res_localization[nb_resolution-1].pose, ranges)

        ############# Trajectory ################
        pose = multi_res_localization[nb_resolution-1].pose
        traj.update(pose)

        ########### Statistics ###########
        # Relations file
        # print(timestamp, pose[0], pose[1], pose[2])
    plot_thread.deactivate()
    plot_thread.join()
if __name__ == '__main__':
    main()
