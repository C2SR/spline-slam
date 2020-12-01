import numpy as np
import math
import time
from scipy.optimize import least_squares

class ScanMatching:
    def __init__(self, spline_map, **kwargs): 
        # Checking for missing parameters
        assert 'angle_min' in kwargs, "[sensor.lidar] Fail to initialize angle_min" 
        assert 'angle_max' in kwargs, "[sensor.lidar] Fail to initialize angle_max"
        assert 'number_beams' in kwargs, "[sensor.lidar] Fail to initialize number_beams"
        assert 'range_min' in kwargs, "[sensor.lidar] Fail to initialize range_min"
        assert 'range_max' in kwargs, "[sensor.lidar] Fail to initialize angle_max"

        # Parameters
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occ = kwargs['logodd_max_occ'] if 'logodd_max_occ' in kwargs else 100
        nb_iteration_max = kwargs['nb_iteration_max'] if 'nb_iteration_max' in kwargs else 10
        angle_min = kwargs['angle_min'] 
        angle_max = kwargs['angle_max']
        number_beams = kwargs['number_beams']
        range_min = kwargs['range_min']
        range_max = kwargs['range_max']
 
        # Scan parameters
        self.angle_min = angle_min 
        self.angle_max = angle_max
        self.range_min = range_min 
        self.range_max = range_max
        self.angles = np.linspace(self.angle_min, self.angle_max, number_beams)

        # LogOdd Map parameters
        self.map = spline_map
        self.logodd_min_free = logodd_min_free
        self.logodd_max_occ = logodd_max_occ

        # Localization parameters
        self.nb_iteration_max = nb_iteration_max        
        self.pose = np.zeros(3)

        self.time = np.zeros(5)           

    """
    Scan-tos-spline matching using range measurements and prior pose (if available)
    """
    def update(self, ranges,  pose_prior=None):
        if pose_prior is None:
            pose_prior = np.copy(self.pose)

        tic = time.clock()
        occ_ranges, occ_angles = self.filter_occ_ranges(ranges)
        self.time[0] += time.clock() - tic
        tic = time.clock()
        occ_pts_local = self.range_to_coordinate(occ_ranges, occ_angles)       
        self.time[1] += time.clock() - tic
        tic = time.clock()
        self.pose, cost = self.compute_pose(pose_prior, occ_pts_local) 
        self.time[2] += time.clock() - tic
        return np.copy(self.pose), cost

    """
    Remove measurements out of range
    """
    def filter_occ_ranges(self, ranges):
        index = np.logical_and(ranges >= self.range_min, ranges < self.range_max)
        occ_ranges = ranges[index]
        occ_angles = self.angles[index]
        return occ_ranges, occ_angles

    def range_to_coordinate(self, ranges, angles):
        direction = np.array([np.cos(angles), np.sin(angles)]) 
        return  ranges * direction

    """ 
    Transform an [2xn] array of (x,y) coordinates to the global frame
    Input: pose np.array<float(3,1)> describes (x,y,theta)'
    """
    def local_to_global_frame(self, pose, local):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s],[s, c]])
        return np.matmul(R, local) + pose[0:2].reshape(2,1)

    """ 
    Estimate pose via scan-matching (core function)
     """
    def compute_pose(self, pose_prior, pts_occ_local, ftol=1e-3, max_nfev=10):
        self.flag = False

        self.c_index_change = np.inf
        self.c_index = None
        self.h_occ = None

        self.threshold_c_index = .05*16*pts_occ_local.shape[1]

        res = least_squares(self.compute_cost_function, 
                            pose_prior,
                            jac = self.compute_jacobian, 
                            verbose = 0, 
                            method='lm',
                            loss='linear',
                            ftol = ftol,
                            max_nfev = max_nfev,
                            f_scale = 1.,
                            args=pts_occ_local )

        return res.x, res.cost


    """
    Computes scan-to-map alignment cost function (scalar)
    """
    def compute_cost_function(self, pose, pts_occ_local_x, pts_occ_local_y):
        # computing alignment error
        pts_occ_local = np.vstack([pts_occ_local_x, pts_occ_local_y])
        pts_occ = self.local_to_global_frame(pose, pts_occ_local)
        c_index_occ = self.map.compute_sparse_tensor_index(pts_occ)
        B_occ, _, _ = self.map.compute_tensor_spline(pts_occ, ORDER=0x01)        
        s_occ = np.sum(self.map.ctrl_pts[c_index_occ]*B_occ, axis=1) /self.logodd_max_occ       
        r = (1 - s_occ)

        if self.flag is True:
            self.c_index_change = np.sum(self.c_index != c_index_occ)
        else:
            self.c_index = c_index_occ
            self.flag = True

        return r

    """
    Computes the jacobian matrix of the scan-to-map alignment cost function (matrix)
    """
    def compute_jacobian(self, pose, pts_occ_local_x, pts_occ_local_y):
        # Recompute jacobian only if change in control points is above threshold_c_index
        if self.c_index_change < self.threshold_c_index:
            return self.h_occ.T
        else:
            self.flag = False

        pts_occ_local = np.vstack([pts_occ_local_x, pts_occ_local_y])
        # Transforming occ points to global frame
        pts_occ = self.local_to_global_frame(pose, pts_occ_local)
        # Spline tensor
        c_index_occ = self.map.compute_sparse_tensor_index(pts_occ)
        _, dBx_occ, dBy_occ = self.map.compute_tensor_spline(pts_occ, ORDER= 0x02)                
        # Rotation matrix
        cos, sin = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[-sin, -cos],[cos, -sin]])           
        # compute H and b  
        ds_occ = np.zeros([2, len(pts_occ_local_x)])
        ds_occ[0,:]=np.sum((self.map.ctrl_pts[c_index_occ]) *dBx_occ, axis=1)/ self.logodd_max_occ 
        ds_occ[1,:]=np.sum((self.map.ctrl_pts[c_index_occ]) *dBy_occ, axis=1)/ self.logodd_max_occ
        dpt_occ_local = R@pts_occ_local
    
        # Jacobian
        h_occ = np.zeros([3, len(pts_occ_local_x)])
        h_occ[0,:] = -ds_occ[0,:]
        h_occ[1,:] = -ds_occ[1,:]
        h_occ[2,:] = -np.sum(dpt_occ_local*ds_occ,axis=0)

        self.h_occ = h_occ

        return h_occ.T