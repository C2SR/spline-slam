import numpy as np
import math
import time
from scipy.optimize import least_squares

class ScanMatching:
    def __init__(self, spline_map, **kwargs): 
        # Parameters
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occupied = kwargs['logodd_max_occupied'] if 'logodd_max_occupied' in kwargs else 100
        nb_iteration_max = kwargs['nb_iteration_max'] if 'nb_iteration_max' in kwargs else 10

        # LogOdd Map parameters
        self.map = spline_map
        self.logodd_min_free = logodd_min_free
        self.logodd_max_occupied = logodd_max_occupied

        # Localization parameters
        self.nb_iteration_max = nb_iteration_max        
        self.pose = np.zeros(3)
        
        # Time
        self.time = np.zeros(3)  


    """ Transform an [2xn] array of (x,y) coordinates to the global frame
        Input: pose np.array<float(3,1)> describes (x,y,theta)'
    """
    def local_to_global_frame(self, pose, local):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s],[s, c]])
        return np.matmul(R, local) + pose[0:2].reshape(2,1)
    
    """ Estimate pose (core function) """
    def compute_pose(self, pose_estimate, pts_occ_local, ftol=1e-3, max_nfev=15):
        self.flag = False

        self.c_index_change = np.inf
        self.c_index = None
        self.h_occ = None

        self.threshold_c_index = .05*16*pts_occ_local.shape[1]

        res = least_squares(self.compute_cost_function, 
                            pose_estimate,
                            jac = self.compute_jacobian, 
                            verbose = 0, 
                            method='lm',
                            loss='linear',
                            ftol = ftol,
                            max_nfev = max_nfev,
                            f_scale = 1.,
                            args=pts_occ_local )

        return res.x, res.cost

    def compute_jacobian(self, pose, pts_occ_local_x, pts_occ_local_y):
        # Recompute jacobian only if change in control points is above threshold_c_index
        if self.c_index_change < self.threshold_c_index:
            return self.h_occ.T
        else:
            self.flag = False

        pts_occ_local = np.vstack([pts_occ_local_x, pts_occ_local_y])
        # Transforming occupied points to global frame
        pts_occ = self.local_to_global_frame(pose, pts_occ_local)
        # Spline tensor
        c_index_occ = self.map.compute_sparse_tensor_index(pts_occ)
        _, dBx_occ, dBy_occ = self.map.compute_tensor_spline(pts_occ, ORDER= 0x02)                
        # Rotation matrix
        cos, sin = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[-sin, -cos],[cos, -sin]])           
        # compute H and b  
        ds_occ = np.zeros([2, len(pts_occ_local_x)])
        ds_occ[0,:]=np.sum((self.map.ctrl_pts[c_index_occ]) *dBx_occ, axis=1)/ self.logodd_max_occupied 
        ds_occ[1,:]=np.sum((self.map.ctrl_pts[c_index_occ]) *dBy_occ, axis=1)/ self.logodd_max_occupied
        dpt_occ_local = R@pts_occ_local
    
        # Jacobian
        h_occ = np.zeros([3, len(pts_occ_local_x)])
        h_occ[0,:] = -ds_occ[0,:]
        h_occ[1,:] = -ds_occ[1,:]
        h_occ[2,:] = -np.sum(dpt_occ_local*ds_occ,axis=0)

        self.h_occ = h_occ

        return h_occ.T

    def compute_cost_function(self, pose, pts_occ_local_x, pts_occ_local_y):
        # computing alignment error
        pts_occ_local = np.vstack([pts_occ_local_x, pts_occ_local_y])
        pts_occ = self.local_to_global_frame(pose, pts_occ_local)
        c_index_occ = self.map.compute_sparse_tensor_index(pts_occ)
        B_occ, _, _ = self.map.compute_tensor_spline(pts_occ, ORDER=0x01)        
        s_occ = np.sum(self.map.ctrl_pts[c_index_occ]*B_occ, axis=1) /self.logodd_max_occupied       
        r = (1 - s_occ)

        if self.flag is True:
            self.c_index_change = np.sum(self.c_index != c_index_occ)
        else:
            self.c_index = c_index_occ
            self.flag = True

        return r

    """"Occupancy grid mapping routine to update map using range measurements"""
    def update_localization(self, sensor,  pose_estimative=None, unreliable_odometry=False):
        if pose_estimative is None:
            pose_estimative = np.copy(self.pose)

        pts_occ_local = sensor.get_occupied_pts()

        # Scan-matching
        tic = time.time()      
        best_cost_estimate = np.inf
        # If odometry is poor search with different orientations
        if unreliable_odometry:
            candidate = [0, np.pi/4., -np.pi/4., np.pi/2., -np.pi/2, -1.5*np.pi, -1.5*np.pi]
        else:
            candidate = [0]
        for theta in candidate:
            pose_estimate_candidate, cost_estimate = self.compute_pose(np.array(pose_estimative) + np.array([0,0,theta]), pts_occ_local, ftol=1e-2, max_nfev=5)
            if cost_estimate < best_cost_estimate:
                best_cost_estimate = cost_estimate
                best_pose_estimate = pose_estimate_candidate
        pose_self, cost_self= self.compute_pose(self.pose, pts_occ_local, ftol = 1e-2, max_nfev=5)        

        if best_cost_estimate < cost_self:
           self.pose, _ = self.compute_pose(best_pose_estimate, pts_occ_local)
        else:
           self.pose, _ = self.compute_pose(pose_self, pts_occ_local) 


        self.time[2] += time.time() - tic