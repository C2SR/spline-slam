import numpy as np

class EKF:
    def __init__(self):
        # State: pos.x, pos.y, yaw, vel.lin.x, vel.lin.y, vel.ang.z
        self.pose = np.zeros(3)
        self.vel = np.zeros(3)
        self.Cov = 1*np.ones([6,6])

    def prediction(self, dt):
        """
        Run the prediction step using the constant velocity model
        """
        # updating the mean 
        vx, vy, w = self.vel[0], self.vel[1], self.vel[2]
        self.pose[0] = self.pose[0] + dt*vx      
        self.pose[1] = self.pose[1] + dt*vy
        self.pose[2] = self.pose[2] + dt*w        
        #self.pose[0] = self.pose[0] + dt*(c*vx - s*vy)      
        #self.pose[1] = self.pose[1] + dt*(s*vx + c*vy)
        #self.pose[2] = self.pose[2] + dt*w        

        # upate the covariance 
        Fx = np.eye(6)
        Fx[0,3], Fx[1,4], Fx[2,5] = dt, dt, dt        
        #Fx[0,2], Fx[0,3], Fx[0,4] = dt*(-s*vx - c*vy), dt*c, -dt*s 
        #Fx[1,2], Fx[1,3], Fx[1,4] = dt*(c*vx - s*vy), dt*s, dt*c 
        #Fx[2,5] = dt
        Fw = np.zeros([6,3])
        Fw[0,0] = Fw[1,1] = Fw[2,2] = .5*dt**2
        Fw[3,0] = Fw[4,1] = Fw[5,2] = dt
        Q = np.diag([.1, .1, .05])
        self.Cov = Fx@self.Cov@Fx.T + Fw@Q@Fw.T

        return np.hstack([self.pose, self.vel])

    def correction(self, pose=None, vel=None):
        x_prior = np.hstack([self.pose, self.vel]) 
        if pose is not None:
            # Update the mean
            R = (.05**2)*np.eye(3)
            Hx = np.zeros([3,6])
            Hx[0,0] = Hx[1,1] = Hx[2,2] = 1.
            K = self.Cov@Hx.T@np.linalg.inv(Hx@self.Cov@Hx.T + R)
            x_posterior = x_prior + K@(pose - self.pose) 
            self.pose, self.vel = x_posterior[0:3], x_posterior[3:6]

            # Update covariance
            self.Cov = (np.eye(6) - K@Hx)@self.Cov

        return np.hstack([self.pose, self.vel])