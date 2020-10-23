import numpy as np

class DiscreteTrajectory:
    def __init__(self, pose_init = np.array([.0,.0,.0])):
        self.traj = pose_init.reshape(3,1)

    def update(self, pose):
        self.traj = np.hstack([self.traj, pose.reshape(3,1)])

    def get_trajectory(self):
        return self.traj