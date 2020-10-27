#!/usr/bin/env python3
# Python libraries
import numpy as np

# ROS libraries
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# Spline-SLAM library
from spline_slam.core import Mapping
from spline_slam.core import ScanMatching
from spline_slam.sensor import Lidar
from spline_slam.odometry import Nonholonomic
from spline_slam.basics import CubicSplineSurface
from spline_slam.trajectory import DiscreteTrajectory
from spline_slam.visualization import SLAMPlotter


class SLAMNode:
    def __init__(self, node_name):
        # ROS node
        rospy.init_node('spline_slam', anonymous=True)

        # Global parameters
        world_frame = rospy.get_param('/world_frame', 'world')
        # Private parameters
        child_frame_id = rospy.get_param('~base_frame', 'base_link')
        self.nb_resolutions = rospy.get_param('~nb_resolutions', 2)

        # Initializing spline-SLAM
        self.multi_res_localization = {}
        self.multi_res_mapping = {}
        self.multi_res_map = {}

        for res in range(0, self.nb_resolutions):
            kwargs_spline= {'knot_space': .05*((2.5)**(self.nb_resolutions-res-1)), 
                            'surface_size': np.array([150.,150.]),
                            'angle_min': 0*np.pi/180., 
                            'angle_max': 359*np.pi/180.,
                            'number_beams': 360,
                            'range_min': 0.05,
                            'range_max': 4.9, #49.9, 
                            'logodd_occupied': 1., 
                            'logodd_free': .1, 
                            'logodd_min_free': -25.,
                            'logodd_max_occupied': 25., 
                            'nb_iteration_max': 50,
                            'free_samples_interval': .15}
       
            self.multi_res_map[res] = CubicSplineSurface(**kwargs_spline)
            self.multi_res_localization[res] = ScanMatching(self.multi_res_map[res], **kwargs_spline)
            self.multi_res_mapping[res] = Mapping(self.multi_res_map[res], **kwargs_spline)

        # Sensor
        self.lidar = Lidar(**kwargs_spline)
        # Odometry 
        self.odometry = Nonholonomic()
        # Trajectory 
        self.traj = DiscreteTrajectory()
        # Plot
        plot_thread = SLAMPlotter(self.multi_res_mapping[self.nb_resolutions-1], self.traj, self.lidar, **kwargs_spline)
        plot_thread.start()

        # ROS publishers and subscribers
        self.odom_pub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.scan_pub = rospy.Subscriber('scan', LaserScan, self.scan_callback)

        # ROS loop
        self.spin()

        # Finish plot thread
        plot_thread.deactivate()
        plot_thread.join()

    def odom_callback(self, msg):
        pass

    def scan_callback(self, msg):
        ###### Pre-processing  ######
        self.lidar.process_new_measurements(np.array(msg.ranges))
        
        ###### Scan matching ######
        for res in range(0, self.nb_resolutions):
            if res==0:
                pose_estimative = np.copy(self.multi_res_localization[self.nb_resolutions-1].pose)
            else:
                pose_estimative = np.copy(self.multi_res_localization[res-1].pose)                
        
        self.multi_res_localization[res].update_localization(self.lidar, pose_estimative, False)

        ###### Mapping #####
        pose = self.multi_res_localization[self.nb_resolutions-1].pose
        for res in range(0, self.nb_resolutions):
            self.multi_res_mapping[res].update_map(self.lidar, pose)

        ############# Trajectory ################
        self.traj.update(pose)

    def spin(self):
        rospy.spin()
        
if __name__ == "__main__":
    node = SLAMNode('~')