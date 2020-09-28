import numpy as np
import math
import time

class SplineLocalization:
    def __init__(self, **kwargs):
        # Parameters
        knot_space = kwargs['knot_space'] if 'knot_space' in kwargs else .05
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

        # Spline-map parameters
        self.degree = 3
        self.knot_space = knot_space

        # LogOdd Map parameters
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
    
    """ Compute spline coefficient index associated to the sparse representation """
    def compute_sparse_spline_index(self, tau, origin):
        mu    = -(np.ceil(-tau/self.knot_space).astype(int)) + origin
        c = np.zeros([len(tau),(self.degree+1)],dtype='int')
        for i in range(0, self.degree+1):
            c[:,i] = mu-self.degree+i
        return c

    """ Compute spline tensor coefficient index associated to the sparse representation """
    def compute_sparse_tensor_index(self, map, pts):
        # Compute spline along each axis
        cx = self.compute_sparse_spline_index(pts[0,:], map.grid_center[0,0])
        cy = self.compute_sparse_spline_index(pts[1,:], map.grid_center[1,0])

        # Kronecker product for index
        c = np.zeros([cx.shape[0],(self.degree+1)**2],dtype='int')
        for i in range(0, self.degree+1):
            for j in range(0, self.degree+1):
                c[:,i*(self.degree+1)+j] = cy[:,i]*(map.grid_size[0,0])+cx[:,j]
        return c

    """"Compute spline coefficients - 1D function """
    def compute_spline_order(self, tau, origin, ORDER=0):
        tau_bar = (tau/self.knot_space + origin) % 1 
        tau_3 = tau_bar + 3
        tau_2 = tau_bar + 2        
        tau_1 = tau_bar + 1
        tau_0 = tau_bar
        
        b = np.zeros([len(tau),self.degree+1])
        b[:,0] = 1/(6)*(-tau_3**3 + 12*tau_3**2 - 48*tau_3 + 64) 
        b[:,1] = 1/(6)*(3*tau_2**3 - 24*tau_2**2 + 60*tau_2 - 44)
        b[:,2] = 1/(6)*(-3*tau_1**3 + 12*tau_1**2 - 12*tau_1 + 4)
        b[:,3] = 1/(6)*(tau_0**3)

        if ORDER == 1:
            # 1st derivative of spline
            db = np.zeros([len(tau),self.degree+1]) 
            db[:,0] = 1/(6)*(-3*tau_3**2 + 24*tau_3 - 48 ) * (1./self.knot_space) 
            db[:,1] = 1/(6)*(9*tau_2**2 - 48*tau_2 + 60 ) * (1./self.knot_space)
            db[:,2] = 1/(6)*(-9*tau_1**2 + 24*tau_1 - 12) * (1./self.knot_space)
            db[:,3] = 1/(6)*(3*tau_0**2) * (1./self.knot_space)
            return b, db
        else:
            return b, -1 

    """"Compute spline tensor coefficients - 2D function """
    def compute_tensor_spline_order(self, map, pts, ORDER=0):
        # Storing number of points
        nb_pts = pts.shape[1]

        # Compute spline along each axis
        bx, dbx = self.compute_spline_order(pts[0,:], map.grid_center[0,0], ORDER)
        by, dby = self.compute_spline_order(pts[1,:], map.grid_center[1,0], ORDER)

        # Compute spline tensor
        B = np.zeros([nb_pts,(self.degree+1)**2])
        for i in range(0,self.degree+1):
            for j in range(0,self.degree+1):           
                B[:,i*(self.degree+1)+j] = by[:,i]*bx[:,j]


        if ORDER ==1:
            dBx = np.zeros([nb_pts,(self.degree+1)**2])
            dBy = np.zeros([nb_pts,(self.degree+1)**2])        
            for i in range(0,self.degree+1):
                for j in range(0,self.degree+1):           
                    dBx[:,i*(self.degree+1)+j] = by[:,i]*dbx[:,j]
                    dBy[:,i*(self.degree+1)+j] = dby[:,i]*bx[:,j]
            return B, dBx, dBy

        return B, -1

    """"Compute spline coefficients - 1D function """
    def compute_spline(self, tau, origin):
        # Number of points
        nb_pts = len(tau)
        # Normalize regressor
        mu    = -(np.ceil(-tau/self.knot_space).astype(int)) + origin
        tau_bar = (tau/self.knot_space + origin) % 1 

        # Compute spline function along the x-axis        
        tau_3 = tau_bar + 3
        tau_2 = tau_bar + 2        
        tau_1 = tau_bar + 1
        tau_0 = tau_bar

        # Spline
        b = np.zeros([nb_pts,self.degree+1])
        b[:,0] = 1/(6)*(-tau_3**3 + 12*tau_3**2 - 48*tau_3 + 64) 
        b[:,1] = 1/(6)*(3*tau_2**3 - 24*tau_2**2 + 60*tau_2 - 44)
        b[:,2] = 1/(6)*(-3*tau_1**3 + 12*tau_1**2 - 12*tau_1 + 4)
        b[:,3] = 1/(6)*(tau_0**3)

        # 1st derivative of spline
        db = np.zeros([nb_pts,self.degree+1])
        db[:,0] = 1/(6)*(-3*tau_3**2 + 24*tau_3 - 48 ) * (1/self.knot_space) 
        db[:,1] = 1/(6)*(9*tau_2**2 - 48*tau_2 + 60 ) * (1/self.knot_space)
        db[:,2] = 1/(6)*(-9*tau_1**2 + 24*tau_1 - 12) * (1/self.knot_space)
        db[:,3] = 1/(6)*(3*tau_0**2) * (1/self.knot_space)

        # 2nd derivative of spline
        ddb = np.zeros([nb_pts,self.degree+1])
        ddb[:,0] = 1/(6)*(-6*tau_3 + 24) * (1/self.knot_space**2)
        ddb[:,1] = 1/(6)*(18*tau_2 - 48) * (1/self.knot_space**2)
        ddb[:,2] = 1/(6)*(-18*tau_1 + 24) * (1/self.knot_space**2)
        ddb[:,3] = 1/(6)*(6*tau_0) * (1/self.knot_space**2)

        c = np.zeros([nb_pts,(self.degree+1)],dtype='int')
        for i in range(0, self.degree+1):
            c[:,i] = mu-self.degree+i

        return c, b, db, ddb

    """"Compute spline tensor coefficients - 2D function """
    def compute_tensor_spline(self, map, pts):
        # Storing number of points
        nb_pts = pts.shape[1]

        # Compute spline along each axis
        cx, bx, dbx, ddbx  = self.compute_spline(pts[0,:], map.grid_center[0,0])
        cy, by, dby, ddby  = self.compute_spline(pts[1,:], map.grid_center[1,0])

        # Compute spline tensor
        ctrl_pt_index = np.zeros([nb_pts,(self.degree+1)**2],dtype='int')
        B = np.zeros([nb_pts,(self.degree+1)**2])
        dBx = np.zeros([nb_pts,(self.degree+1)**2])
        dBy = np.zeros([nb_pts,(self.degree+1)**2])
        ddBx = np.zeros([nb_pts,(self.degree+1)**2])
        ddBy = np.zeros([nb_pts,(self.degree+1)**2])
        ddBxy = np.zeros([nb_pts,(self.degree+1)**2])       
        for i in range(0,self.degree+1):
            for j in range(0,self.degree+1):           
                ctrl_pt_index[:,i*(self.degree+1)+j] = cy[:,i]*(map.grid_size[0,0])+cx[:,j]
                B[:,i*(self.degree+1)+j] = by[:,i]*bx[:,j]
                dBx[:,i*(self.degree+1)+j] = by[:,i]*dbx[:,j]
                dBy[:,i*(self.degree+1)+j] = dby[:,i]*bx[:,j]
                ddBx[:,i*(self.degree+1)+j] = by[:,i]*ddbx[:,j]
                ddBy[:,i*(self.degree+1)+j] = ddby[:,i]*bx[:,j]
                ddBxy[:,i*(self.degree+1)+j] = dby[:,i]*dbx[:,j]                                

        return ctrl_pt_index, B, dBx, dBy, ddBx, ddBy, ddBxy

    """ Estimate pose (core function) """
    def compute_pose(self, map, pose_estimate, pts_occ_local, nb_iteration_max, gradient_step_size = .1):
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
                if np.linalg.norm(df[0:2]) > self.knot_space:
                    df = self.knot_space*df/np.linalg.norm(df[0:2])

            # Compute pose 
            delta_pose = gradient_step_size*df
            pose_estimate_candidate = pose_estimate - delta_pose
            cost_candidate = self.cost_function(map, pts_occ_local, pose_estimate_candidate, alpha=alpha, c=self.c)
            residue =  cost - cost_candidate

            if residue > 0 :
                gradient_step_size = 1.25*gradient_step_size
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
        c_index_occ = self.compute_sparse_tensor_index(map, pts_occ)
        B_occ, _ = self.compute_tensor_spline_order(map, pts_occ, ORDER=0)        
        s_occ = np.sum(map.ctrl_pts[c_index_occ]*B_occ, axis=1)        
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
        c_index_occ = self.compute_sparse_tensor_index(map, pts_occ_global)
        B_occ, dBx_occ, dBy_occ = self.compute_tensor_spline_order(map, pts_occ_global, ORDER=1)                
        #c_index_occ, B_occ, dBx_occ, dBy_occ, ddBx_occ, ddBy_occ, ddBxy_occ = self.compute_tensor_spline(map, pts_occ_global)
        # Current value on the map
        s_occ = np.sum(map.ctrl_pts[c_index_occ]*B_occ, axis=1) 
        # Alginment error
        e_occ = (1 - (s_occ/self.logodd_max_occupied))            
        n_occ = len(e_occ)
        # Rotation matrix
        cos, sin = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[-sin, -cos],[cos, -sin]])           
        # compute H and b  
        ds_occ = np.zeros([2, n_occ])
        ds_occ[0,:]=np.sum(map.ctrl_pts[c_index_occ]*dBx_occ, axis=1)
        ds_occ[1,:]=np.sum(map.ctrl_pts[c_index_occ]*dBy_occ, axis=1)
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
    def update_localization(self, map, ranges, pose_estimative=None, unreliable_odometry=True):
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