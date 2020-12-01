import numpy as np
import math
import time
import random
import scipy.sparse.linalg

class Mapping:
    def __init__(self, spline_map, **kwargs):
        # Checking for missing parameters
        assert 'angle_min' in kwargs, "[sensor.lidar] Fail to initialize angle_min" 
        assert 'angle_max' in kwargs, "[sensor.lidar] Fail to initialize angle_max"
        assert 'number_beams' in kwargs, "[sensor.lidar] Fail to initialize number_beams"
        assert 'range_min' in kwargs, "[sensor.lidar] Fail to initialize range_min"
        assert 'range_max' in kwargs, "[sensor.lidar] Fail to initialize angle_max"
        assert 'knot_interval' in kwargs, "[sensor.lidar] Fail to initialize knot_interval"

        # Parameters
        logodd_occ = kwargs['logodd_occ'] if 'logodd_occ' in kwargs else .9
        logodd_free = kwargs['logodd_free'] if 'logodd_free' in kwargs else .3
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occ = kwargs['logodd_max_occ'] if 'logodd_max_occ' in kwargs else 100
        angle_min = kwargs['angle_min'] 
        angle_max = kwargs['angle_max']
        number_beams = kwargs['number_beams']
        range_min = kwargs['range_min']
        range_max = kwargs['range_max']
        knot_interval = kwargs['knot_interval']

        # Scan parameters
        self.angle_min = angle_min 
        self.angle_max = angle_max
        self.range_min = range_min 
        self.range_max = range_max
        self.angles = np.linspace(self.angle_min, self.angle_max, number_beams)
        self.beam_samples = np.arange(self.range_min, self.range_max, 3*knot_interval).reshape([-1,1])       
        # Storing beam samples in memory for speed up 
        self.beam_matrix_x = self.beam_samples.reshape(-1,1) * np.cos(self.angles)
        self.beam_matrix_y = self.beam_samples.reshape(-1,1) * np.sin(self.angles)    

        # LogOdd Map parameters
        self.map = spline_map
        self.map.ctrl_pts *=  .5*(logodd_max_occ+logodd_min_free)
        self.logodd_occ = logodd_occ
        self.logodd_free = logodd_free
        self.logodd_min_free = logodd_min_free
        self.logodd_max_occ = logodd_max_occ

        self.time = np.zeros(5)           

    """"Occupancy grid mapping routine to update map using range measurements"""
    def update(self, ranges, pose):
        # Occ space
        tic = time.clock()
        occ_ranges, occ_angles = self.filter_occ_ranges(ranges)
        occ_pts_local = self.range_to_coordinate(occ_ranges, occ_angles)       
        self.time[0] += time.clock() - tic

        # Free space
        tic = time.clock()
        free_pts_local = self.compute_free_space(ranges)
        self.time[1] += time.clock() - tic

         # Transforming metric coordinates from the local to the global frame
        tic = time.clock()
        occ_pts = self.local_to_global_frame(pose,occ_pts_local)
        free_pts = self.local_to_global_frame(pose,free_pts_local)
        self.time[2] += time.clock() - tic

        # Update spline map
        tic = time.clock()
        self.update_spline_map(occ_pts,  free_pts, pose)
        self.time[3] += time.clock() - tic

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

    def compute_free_space(self, ranges):      
        index_free =  np.where((ranges >= self.range_min) & (ranges  <= self.range_max))[0]
        index_free_matrix = self.beam_samples  < (ranges[index_free]).reshape([1,-1])
        free_pts = np.vstack([self.beam_matrix_x[:, index_free][index_free_matrix],
                              self.beam_matrix_y[:, index_free][index_free_matrix]])    
        if free_pts.size == 0:
            free_pts = np.zeros([2,1])

        return free_pts  

    """ Transform an [2xn] array of (x,y) coordinates to the global frame
        Input: pose np.array<float(3,1)> describes (x,y,theta)'
    """
    def local_to_global_frame(self, pose, local):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s],[s, c]])
        return np.matmul(R, local) + pose[0:2].reshape(2,1) 

    """"Update the control points of the spline map"""
    def update_spline_map(self, occ_pts, free_pts, pose):
        # Free space 
        c_index_free = self.map.compute_sparse_tensor_index(free_pts)
        c_index_occ = self.map.compute_sparse_tensor_index(occ_pts)

        #tic = time.clock()
        B_occ, _, _ = self.map.compute_tensor_spline(occ_pts, ORDER=0x01)
        #self.time[4] += time.clock() - tic
        s_est_occ_ant = np.sum(self.map.ctrl_pts[c_index_occ]*B_occ, axis=1)

        # Free space
        self.map.ctrl_pts[c_index_free] -= self.logodd_free
        self.map.ctrl_pts[c_index_occ] += .5*self.logodd_free

        # Occ space [FAST]
        s_est_occ = s_est_occ_ant #np.sum(self.map.ctrl_pts[c_index_occ]*B_occ, axis=1)   
        e_occ = (self.logodd_max_occ - s_est_occ) 
        B_occ_norm = np.linalg.norm(B_occ, axis=1)
        B_occ_norm_squared = B_occ_norm**2
        mag_occ =  np.minimum(self.logodd_occ/B_occ_norm_squared, np.abs(e_occ)) * np.sign(e_occ)
        np.add.at(self.map.ctrl_pts, c_index_occ, (B_occ.T*mag_occ).T)        

        # Clamping control points 
        self.map.ctrl_pts[c_index_occ] = np.maximum(np.minimum(self.map.ctrl_pts[c_index_occ], self.logodd_max_occ), self.logodd_min_free)

    """ Evaluata map """
    def evaluate_map(self, pts):
        B, _, _ = self.map.compute_tensor_spline(pts)
        c_index = self.map.compute_sparse_tensor_index(pts)
        s = np.sum(self.map.ctrl_pts[c_index]*B, axis=1)
        return s
        
