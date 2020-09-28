import numpy as np
import math
import time

class SplineLocalization:
    def __init__(self, spline_map, **kwargs): 
        # Parameters
        min_angle = kwargs['min_angle'] if 'min_angle' in kwargs else 0.
        max_angle = kwargs['max_angle'] if 'max_angle' in kwargs else 2.*np.pi 
        angle_increment = kwargs['angle_increment'] if 'angle_increment' in kwargs else 1.*np.pi/180.
        range_min = kwargs['range_min'] if 'range_min' in kwargs else 0.12
        range_max = kwargs['range_max'] if 'range_max' in kwargs else 3.5
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occupied = kwargs['logodd_max_occupied'] if 'logodd_max_occupied' in kwargs else 100
        det_Hinv_threshold = kwargs['det_Hinv_threshold'] if 'det_Hinv_threshold' in kwargs else 1e-3
        nb_iteration_max = kwargs['nb_iteration_max'] if 'nb_iteration_max' in kwargs else 10
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 2

        # LogOdd Map parameters
        self.map = spline_map
        self.logodd_min_free = logodd_min_free
        self.logodd_max_occupied = logodd_max_occupied

        # Sensor scan parameters
        self.min_angle = min_angle
        self.max_angle = max_angle 
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max
        self.angles = np.arange(min_angle, max_angle, angle_increment )                

        # Localization parameters
        self.nb_iteration_max = nb_iteration_max        
        self.det_Hinv_threshold = det_Hinv_threshold
        self.pose = np.zeros(3)
        self.alpha = alpha
        self.sensor_subsampling_factor = 1 
        
        # Time
        self.time = np.zeros(3)  

    """Removes spurious (out of range) measurements
        Input: ranges np.array<float>
    """ 
    def remove_spurious_measurements(self, ranges):
        # Finding indices of the valid ranges
        ind_occ = np.logical_and(ranges >= self.range_min, ranges < self.range_max)
        return ranges[ind_occ], self.angles[ind_occ]

    """ Transforms ranges measurements to (x,y) coordinates (local frame) """
    def range_to_coordinate(self, ranges, angles):
        angles = np.array([np.cos(angles), np.sin(angles)]) 
        return ranges * angles 

    """ Transform an [2xn] array of (x,y) coordinates to the global frame
        Input: pose np.array<float(3,1)> describes (x,y,theta)'
    """
    def local_to_global_frame(self, pose, local):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s],[s, c]])
        return np.matmul(R, local) + pose[0:2].reshape(2,1)
    
    """ Estimate pose (core function) """
    def compute_pose(self, map, pose_estimate, pts_occ_local, nb_iteration_max, gradient_step_size = 0.1):
        # Initializing parameters
        nb_iterations = 0.
        residue = np.inf
 
        alpha = 2.
        self.c = 1. #1 #2./3 #/3

        has_updated = True
        nb_updates = 0
        while (np.abs(residue) > .0001) and nb_iterations < nb_iteration_max:
            if has_updated:
                df, cost = self.compute_pose_increment(map, pts_occ_local, pose_estimate, alpha=alpha, c=self.c)
                if np.linalg.norm(df[0:2]) > self.map.knot_space:
                    df = self.map.knot_space*df/np.linalg.norm(df[0:2])

            # Compute pose 
            delta_pose = gradient_step_size*df
            pose_estimate_candidate = pose_estimate - delta_pose
            cost_candidate = self.cost_function(map, pts_occ_local, pose_estimate_candidate, alpha=alpha, c=self.c)
            residue =  cost - cost_candidate

            if residue > 0 :
                gradient_step_size = 2.25*gradient_step_size
                pose_estimate = pose_estimate_candidate
                cost = cost_candidate
                has_updated = True
                nb_updates += 1
                nb_iterations += 1
 
            else:
                gradient_step_size /= 4
                has_updated = False

        return pose_estimate, cost, gradient_step_size

    def cost_function(self, map, pts_occ_local, pose, alpha, c):
        # computing alignment error
        pts_occ = self.local_to_global_frame(pose, pts_occ_local)
        c_index_occ = self.map.compute_sparse_tensor_index(pts_occ)
        B_occ, _, _ = self.map.compute_tensor_spline(pts_occ, ORDER=0x01)        
        s_occ = np.sum(self.map.ctrl_pts[c_index_occ]*B_occ, axis=1)        
        e_occ = (1 - s_occ/self.logodd_max_occupied)

        if alpha == 2:   # squared error (L2) loss function
            r = .5*(e_occ/c)**2
        elif alpha == 0:   # Cauchy (aka  Lorentzian) loss 
            r = np.log( .5*(e_occ/c)**2 +1  )
        elif alpha == -np.inf:  # Welsch (aka Leclerc) loss function
            r = 1 - np.exp(-.5 * (e_occ/c)**2)
        else:
            r =  (abs(alpha-2)/alpha)  * ( (((e_occ/c)**2)/abs(alpha-2) + 1)**(alpha/2.) - 1)

        return np.sum(r)

    def compute_pose_increment(self, map, pts_occ_local, pose, alpha, c):
        # Transforming occupied points to global frame
        pts_occ_global = self.local_to_global_frame(pose, pts_occ_local)
        # Spline tensor
        c_index_occ = self.map.compute_sparse_tensor_index(pts_occ_global)
        B_occ, dBx_occ, dBy_occ = self.map.compute_tensor_spline(pts_occ_global, ORDER=0x01 | 0x02)                
        #c_index_occ, B_occ, dBx_occ, dBy_occ, ddBx_occ, ddBy_occ, ddBxy_occ = self.compute_tensor_spline(map, pts_occ_global)
        # Current value on the map
        s_occ = np.sum(self.map.ctrl_pts[c_index_occ]*B_occ, axis=1) 
        # Alginment error
        e_occ = (1 - (s_occ/self.logodd_max_occupied))            
        n_occ = len(e_occ)
        # Rotation matrix
        cos, sin = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[-sin, -cos],[cos, -sin]])           
        # compute H and b  
        ds_occ = np.zeros([2, n_occ])
        ds_occ[0,:]=np.sum(self.map.ctrl_pts[c_index_occ]*dBx_occ, axis=1)
        ds_occ[1,:]=np.sum(self.map.ctrl_pts[c_index_occ]*dBy_occ, axis=1)
        dpt_occ_local = R@pts_occ_local
    
        # JAcobian
        h_occ = np.zeros([3,n_occ])
        h_occ[0,:] = -ds_occ[0,:]
        h_occ[1,:] = -ds_occ[1,:]
        h_occ[2,:] = -np.sum(dpt_occ_local*ds_occ,axis=0)

        if alpha == 2:   # Welsch/Leclerc loss function
            r = .5*(e_occ/c)**2
            df = (1./c**2)*np.sum( e_occ * h_occ, axis = 1)
            # dfx = (1./c**2)*( e_occ )
            # ddfx = (1./c**2)

        elif alpha == 0:
            r = np.log( .5*(e_occ/c)**2 +1  )
            df = np.sum( 2* e_occ /(e_occ**2 + 2*c**2)* h_occ, axis = 1)
            # dfx =  2* e_occ /(e_occ**2 + 2*c**2)
            # ddfx =  2* (-e_occ**2 + 2*c**2) /((e_occ**2 + 2*c**2)**2)

        elif alpha == -np.inf:
            z = np.exp(-.5*(e_occ/c)**2)
            r = 1 - z
            df = (1./c**2)*np.sum( e_occ * z * h_occ, axis = 1)  
            # dfx = (1./c**2)*( e_occ * z )
            # ddfx = (z/c**2)*(1 - (e_occ/c)**2 )
        else:
            z = ((e_occ/c)**2 / abs(alpha-2)) + 1
            r =  abs(alpha-2)/alpha  * (z**(alpha/2.) - 1)
            df = (1./c**2)*np.sum( e_occ * z**(alpha/2.-1) * h_occ, axis = 1)
            # dfx = (1./c**2)*( e_occ * (z**(alpha/2.-1)) )            
            # ddfx = (1./c**2)* ( (z**(alpha/2.-1)) + np.sign(alpha-2)* (z**(alpha/2.-2)) *(e_occ/c)**2  ) 

        # H = np.zeros([3,3])
        # sxx = dfx*np.sum(map.ctrl_pts[c_index_occ]*ddBx_occ, axis=1)
        # syy = dfx*np.sum(map.ctrl_pts[c_index_occ]*ddBy_occ, axis=1)
        # sxy = dfx*np.sum(map.ctrl_pts[c_index_occ]*ddBxy_occ, axis=1)                
        # sxtheta =  dpt_occ_local[0,:]*sxx + dpt_occ_local[1,:]*sxy
        # sytheta =  dpt_occ_local[0,:]*sxy + dpt_occ_local[1,:]*syy
        # sthetatheta = dpt_occ_local[0,:]*sxtheta + dpt_occ_local[1,:]*sytheta

        # H[0,0] = np.sum(sxx)
        # H[1,1] = np.sum(syy)
        # H[2,2] = np.sum(sthetatheta)
        # H[0,1] = H[1,0] = np.sum(sxy)
        # H[0,2] = H[2,0] = np.sum(sxtheta)
        # H[1,2] = H[2,1] = np.sum(sytheta)

        # ddf = ( (ddfx*h_occ) @ h_occ.T + H)         
        #if np.linalg.det(ddf) > self.det_Hinv_threshold:
        #    df = np.linalg.inv(ddf)@df   
        cost = np.sum(r)
        return df, cost 


    """"Occupancy grid mapping routine to update map using range measurements"""
    def update_localization(self, ranges, pose_estimative=None, unreliable_odometry=True):
        map = None
        if pose_estimative is None:
            pose_estimative = np.copy(self.pose)
        # Removing spurious measurements
        tic = time.time()
        ranges_occ, angles = self.remove_spurious_measurements(ranges)
        self.time[0] += time.time() - tic
        # Converting range measurements to metric coordinates
        tic = time.time()
        pts_occ_local = self.range_to_coordinate(ranges_occ, angles)
        self.pts_occ_local = pts_occ_local
        #pts_free_local = self.detect_free_space(ranges)
        self.time[1] += time.time() - tic
        # Localization
        tic = time.time()
        
        best_cost_estimate = np.inf
        if unreliable_odometry:
            candidate = [0, np.pi/4., -np.pi/4., np.pi/2., -np.pi/2]
        else:
            candidate = [0]
        for theta in candidate:
            pose_estimate_candidate, cost_estimate, step_self = self.compute_pose(map, np.array(pose_estimative) + np.array([0,0,theta]), pts_occ_local, nb_iteration_max=5)
            if cost_estimate < best_cost_estimate:
                best_cost_estimate = cost_estimate
                best_pose_estimate = pose_estimate_candidate
                best_step_estimate = step_self
        pose_self, cost_self, step_self = self.compute_pose(map,  self.pose, pts_occ_local, nb_iteration_max=5)        

        if best_cost_estimate < cost_self:
            self.pose, _, _ = self.compute_pose(map, best_pose_estimate, pts_occ_local, nb_iteration_max=self.nb_iteration_max-10, gradient_step_size=best_step_estimate)
        else:
            self.pose, _, _ = self.compute_pose(map, pose_self, pts_occ_local, nb_iteration_max=self.nb_iteration_max-10, gradient_step_size=step_self)
        self.time[2] += time.time() - tic