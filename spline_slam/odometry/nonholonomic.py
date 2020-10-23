import numpy as np

class Nonholonomic:
    def __init__(self, **kwargs):
        self.previous_pose = np.zeros(3)
        self.previous_timestamp = None

    def pose_to_odometry(self, timestamp, pose):
        w0, vf, wf, dt = 0., 0., 0., 0.
        if self.previous_timestamp is not None:
            theta0 = np.arctan2(pose[1] - self.previous_pose[1],pose[0]-self.previous_pose[0]) - self.previous_pose[2]
            translation = np.linalg.norm(pose[0:2]-self.previous_pose[0:2])
            thetaf = pose[2] - np.arctan2(pose[1] - self.previous_pose[1],pose[0]-self.previous_pose[0])
        
            dt = timestamp - self.previous_timestamp
            w0 = self.normalize_angle(theta0) / dt
            vf = translation / dt
            wf = self.normalize_angle(thetaf) / dt
        

        self.previous_pose = pose
        self.previous_timestamp = timestamp

        return np.array([w0, vf, wf]), dt

    def pose_to_discrete_odometry(self, timestamp, pose):
        theta0, translation, thetaf, dt = 0., 0., 0., 0.
        if self.previous_timestamp is not None:
            theta0 = np.arctan2(pose[1] - self.previous_pose[1],pose[0]-self.previous_pose[0]) - self.previous_pose[2]
            translation = np.linalg.norm(pose[0:2]-self.previous_pose[0:2])
            thetaf = pose[2] - np.arctan2(pose[1] - self.previous_pose[1],pose[0]-self.previous_pose[0])      
            
            theta0 = self.normalize_angle(theta0)
            thetaf = self.normalize_angle(thetaf)

            dt = timestamp - self.previous_timestamp
        self.previous_pose = np.copy(pose)
        self.previous_timestamp = timestamp

        return np.array([theta0, translation, thetaf]), dt

    def update(self, pose, odometry):
        pose_new = np.array([.0,.0,.0])
        pose_new[2] = pose[2] + odometry[0]
        pose_new[0:2] = pose[0:2] + odometry[1]*np.array([ np.cos(pose[2]), np.sin(pose[2])])
        pose_new[2] = pose_new[2] + odometry[2] 

        return pose_new

    def normalize_angle(self, theta):
        if theta > np.pi:
            theta -= 2*np.pi
        elif theta < -np.pi:
            theta += 2*np.pi
        return theta 
                    
