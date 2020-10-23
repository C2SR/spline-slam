import numpy as np
import math
import time
import random
import scipy.sparse.linalg

class Mapping:
    def __init__(self, spline_map, **kwargs):
        # Parameters
        min_angle = kwargs['min_angle'] if 'min_angle' in kwargs else 0.
        max_angle = kwargs['max_angle'] if 'max_angle' in kwargs else 2.*np.pi - 1.*np.pi/180.
        angle_increment = kwargs['angle_increment'] if 'angle_increment' in kwargs else 1.*np.pi/180.
        range_min = kwargs['range_min'] if 'range_min' in kwargs else 0.12
        range_max = kwargs['range_max'] if 'range_max' in kwargs else 3.6
        logodd_occupied = kwargs['logodd_occupied'] if 'logodd_occupied' in kwargs else .9
        logodd_free = kwargs['logodd_free'] if 'logodd_free' in kwargs else .3
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occupied = kwargs['logodd_max_occupied'] if 'logodd_max_occupied' in kwargs else 100
        max_nb_rays = kwargs['max_nb_rays'] if 'max_nb_rays' in kwargs else 360

        # LogOdd Map parameters
        self.map = spline_map
        self.map.ctrl_pts *=  .5*(logodd_max_occupied+logodd_min_free)
        self.logodd_occupied = logodd_occupied
        self.logodd_free = logodd_free
        self.logodd_min_free = logodd_min_free
        self.logodd_max_occupied = logodd_max_occupied
        self.free_detection_spacing = 2*self.map.knot_space 
        self.free_ranges = np.arange(max(self.map.knot_space, range_min), range_max, self.free_detection_spacing)       
        
        # Sensor scan parameters
        self.min_angle = min_angle
        self.max_angle = max_angle 
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max
        self.angles = np.arange(min_angle, max_angle, angle_increment )
        self.sensor_subsampling_factor = max(divmod(len(self.angles),max_nb_rays)[0],1)

        # Storing ranges for speed up 
        self.free_ranges_matrix = np.tile(self.free_ranges.reshape(-1,1), (1,len(self.angles)))
        self.ray_matrix_x = self.free_ranges.reshape(-1,1) * np.cos(self.angles)
        self.ray_matrix_y = self.free_ranges.reshape(-1,1) * np.sin(self.angles)    
        self.time = np.zeros(5)           

    """Removes spurious (out of range) measurements
        Input: ranges np.array<float>
    """ 
    def remove_spurious_measurements(self, ranges):
        # Finding indices of the valid ranges
        ind_occ = np.logical_and(ranges >= self.range_min, ranges < self.range_max)
        return ranges[ind_occ], self.angles[ind_occ]

    """ Transforms ranges measurements to (x,y) coordinates (local frame) """
    def range_to_coordinate(self, ranges, angles):
        direction = np.array([np.cos(angles), np.sin(angles)]) 
        return  ranges * direction
    
    """ Transform an [2xn] array of (x,y) coordinates to the global frame
        Input: pose np.array<float(3,1)> describes (x,y,theta)'
    """
    def local_to_global_frame(self, pose, local):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s],[s, c]])
        return np.matmul(R, local) + pose[0:2].reshape(2,1) 

    """ Detect free space """
    def detect_free_space(self, ranges):      
        ranges_free = ranges + self.map.knot_space
        init = int(np.random.rand()*self.sensor_subsampling_factor)
        index_free =  np.where((ranges_free >= self.range_min) & (ranges  <= self.range_max))[0][init::self.sensor_subsampling_factor]
        index_free_matrix = self.free_ranges_matrix[:,index_free] <  \
                            (ranges_free[index_free]).reshape([1,-1])

        #index_free_matrix[:,0:-1] = np.logical_and(index_free_matrix[:,0:-1], index_free_matrix[:,1:]) 
        #index_free_matrix[:,1:] = np.logical_and(index_free_matrix[:,1:], index_free_matrix[:,0:-1])  

        pts_free = np.vstack([self.ray_matrix_x[:, index_free][index_free_matrix],
                            self.ray_matrix_y[:, index_free][index_free_matrix]]) 

        if pts_free.size == 0:
            pts_free = np.zeros([2,1])

        return pts_free 

    """"Update the control points of the spline map"""
    def update_spline_map(self, pts_occ, pts_free, pose):
        # Free space 
        c_index_free = self.map.compute_sparse_tensor_index(pts_free)
        c_index_occ = self.map.compute_sparse_tensor_index(pts_occ)

        B_occ, _, _ = self.map.compute_tensor_spline(pts_occ, ORDER=0x01)
        s_est_occ_ant = np.sum(self.map.ctrl_pts[c_index_occ]*B_occ, axis=1)

        # Free space
        self.map.ctrl_pts[c_index_free] -= self.logodd_free
        self.map.ctrl_pts[c_index_occ] += .5*self.logodd_free

        # Occupied space [SLOW]
        # for i in range(0, pts_occ.shape[1]):
        #     s_est_occ = np.sum(self.map.ctrl_pts[c_index_occ[i,:]]*B_occ[i,:])   
        #     e_occ = min(self.logodd_max_occupied, (s_est_occ_ant[i] + self.logodd_occupied))-s_est_occ 
        #     B_occ_norm = np.linalg.norm(B_occ[i,:])
        #     B_occ_norm_squared = B_occ_norm**2
        #     mag_occ =  e_occ /B_occ_norm_squared
        #     np.add.at(self.map.ctrl_pts, c_index_occ[i,:], (B_occ[i,:]*mag_occ))

        # Occupied space [FAST]
        s_est_occ = s_est_occ_ant #np.sum(self.map.ctrl_pts[c_index_occ]*B_occ, axis=1)   
        e_occ = (self.logodd_max_occupied - s_est_occ) 
        B_occ_norm = np.linalg.norm(B_occ, axis=1)
        B_occ_norm_squared = B_occ_norm**2
        mag_occ =  np.minimum(self.logodd_occupied/B_occ_norm_squared, np.abs(e_occ)) * np.sign(e_occ)
        np.add.at(self.map.ctrl_pts, c_index_occ, (B_occ.T*mag_occ).T)        

        # Clamping control points 
        self.map.ctrl_pts[c_index_free] = np.maximum(np.minimum(self.map.ctrl_pts[c_index_free], self.logodd_max_occupied), self.logodd_min_free)

    """ Evaluata map """
    def evaluate_map(self, pts):
        B, _, _ = self.map.compute_tensor_spline(pts)
        c_index = self.map.compute_sparse_tensor_index(pts)
        s = np.sum(self.map.ctrl_pts[c_index]*B, axis=1)
        return s

    """"Occupancy grid mapping routine to update map using range measurements"""
    def update_map(self, pose, ranges):
        # Removing spurious measurements
        tic = time.clock()
        ranges_occ, angles_occ = self.remove_spurious_measurements(ranges)
        self.time[0] += time.clock() - tic
        # Converting range measurements to metric coordinates
        tic = time.clock()
        pts_occ_local = self.range_to_coordinate(ranges_occ, angles_occ)
        self.time[1] += time.clock() - tic
        # Detecting free cells in metric coordinates
        tic = time.clock()
        pts_free_local  = self.detect_free_space(ranges)
        self.time[2] += time.clock() - tic
        # Transforming metric coordinates from the local to the global frame
        tic = time.clock()
        pts_occ = self.local_to_global_frame(pose,pts_occ_local)
        pts_free = self.local_to_global_frame(pose,pts_free_local)
        self.time[3] += time.clock() - tic
        # Compute spline
        tic = time.clock()
        self.update_spline_map(pts_occ,  pts_free, pose)
        self.time[4] += time.clock() - tic
        
