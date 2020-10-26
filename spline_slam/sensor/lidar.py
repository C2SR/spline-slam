import numpy as np

class Lidar:
    def __init__(self, **kwargs):
        # Checking for missing parameters
        assert 'angle_min' in kwargs, "[sensor.lidar] Fail to initialize angle_min" 
        assert 'angle_max' in kwargs, "[sensor.lidar] Fail to initialize angle_max"
        assert 'number_beams' in kwargs, "[sensor.lidar] Fail to initialize number_beams"
        assert 'range_min' in kwargs, "[sensor.lidar] Fail to initialize range_min"
        assert 'range_max' in kwargs, "[sensor.lidar] Fail to initialize angle_max"
        assert 'free_detection_spacing' in kwargs, "[sensor.lidar] Fail to initialize free_detection_spacing"

        # Sensor scan parameters
        self.angle_min = kwargs['angle_min'] 
        self.angle_max = kwargs['angle_max']
        self.number_beams = kwargs['number_beams']
        self.range_min = kwargs['range_min']
        self.range_max = kwargs['range_max']
        self.free_detection_spacing = kwargs['free_detection_spacing']
 
        self.angles = np.linspace(self.angle_min, self.angle_max, self.number_beams)
        self.beam_samples = np.arange(self.range_min, self.range_max, self.free_detection_spacing).reshape([-1,1])       

        # Storing beam samples in memory for speed up 
        self.beam_matrix_x = self.beam_samples.reshape(-1,1) * np.cos(self.angles)
        self.beam_matrix_y = self.beam_samples.reshape(-1,1) * np.sin(self.angles)    

    def update(self, ranges, **kwargs):
        occupied_ranges, occupied_angles = self.filter_occupied_ranges(ranges)              
        self.occupied_pts = self.range_to_coordinate(occupied_ranges, occupied_angles)
        self.free_pts = self.compute_free_space(ranges)

    def filter_occupied_ranges(self, ranges):
        index = np.logical_and(ranges >= self.range_min, ranges < self.range_max)
        occupied_ranges = ranges[index]
        occupied_angles = self.angles[index]
        return occupied_ranges, occupied_angles

    def range_to_coordinate(self, ranges, angles):
        direction = np.array([np.cos(angles), np.sin(angles)]) 
        return  ranges * direction

    def compute_free_space(self, ranges):      
        index_free =  np.where((ranges >= self.range_min) & (ranges  <= self.range_max))[0]
        index_free_matrix = self.beam_samples  < (ranges[index_free]).reshape([1,-1])
        pts_free = np.vstack([self.beam_matrix_x[:, index_free][index_free_matrix],
                              self.beam_matrix_y[:, index_free][index_free_matrix]])    
        if pts_free.size == 0:
            pts_free = np.zeros([2,1])

        return pts_free  

    def get_occupied_pts(self):
        return self.occupied_pts

    def get_free_pts(self):
        return self.free_pts


