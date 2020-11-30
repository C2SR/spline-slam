import numpy as np

class Lidar:
    def __init__(self, **kwargs):
        # Checking for missing parameters
        assert 'angle_min' in kwargs, "[sensor.lidar] Fail to initialize angle_min" 
        assert 'angle_max' in kwargs, "[sensor.lidar] Fail to initialize angle_max"
        assert 'number_beams' in kwargs, "[sensor.lidar] Fail to initialize number_beams"
        assert 'range_min' in kwargs, "[sensor.lidar] Fail to initialize range_min"
        assert 'range_max' in kwargs, "[sensor.lidar] Fail to initialize angle_max"

        # Sensor scan parameters
        self.angle_min = kwargs['angle_min'] 
        self.angle_max = kwargs['angle_max']
        self.number_beams = kwargs['number_beams']
        self.range_min = kwargs['range_min']
        self.range_max = kwargs['range_max']
 
        self.angles = np.linspace(self.angle_min, self.angle_max, self.number_beams)

    def update(self, ranges, **kwargs):
        occupied_ranges, occupied_angles = self.filter_occupied_ranges(ranges)              
        self.occupied_pts = self.range_to_coordinate(occupied_ranges, occupied_angles)

    def filter_occupied_ranges(self, ranges):
        index = np.logical_and(ranges >= self.range_min, ranges < self.range_max)
        occupied_ranges = ranges[index]
        occupied_angles = self.angles[index]
        return occupied_ranges, occupied_angles

    def range_to_coordinate(self, ranges, angles):
        direction = np.array([np.cos(angles), np.sin(angles)]) 
        return  ranges * direction

    def get_occupied_pts(self):
        return self.occupied_pts



