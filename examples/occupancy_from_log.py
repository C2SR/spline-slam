import numpy as np
from matplotlib import pyplot as plt
from spline_slam.occupancy import OccupancyGridMap
import sys

def main():
    #TODO: Use better arg handling
    if len(sys.argv) < 2:
        print("You must provide a log file")
        sys.exit(-1)
    #TODO: Error handling for file not being there
    file_name = sys.argv[1]
    # Opening log file
    file_handle = open(file_name, "r")
    
    # Instantiating the grid map object
    kwargs_occupancy_map = {'resolution': .05, 'map_size': np.array([5.,5.])}
    map = OccupancyGridMap(**kwargs_occupancy_map)
    
    # Retrieving sensor parameters
    data = file_handle.readline()  
    data = np.fromstring( data, dtype=np.float, sep=' ' )
    
    # Read the data and build the map
    fig, ax = plt.subplots()
    plt.show(block=False)
    k = 0
    n = 0
    for data in file_handle:
        # Reading data from log
        data = np.fromstring( data, dtype=np.float, sep=' ' )
        pose = data[0:3]
        ranges = data[6:]
        # update the map
        map.update_map(pose, ranges)
        n = n + 1
        k += 1
        if k > 50:
            plt.imshow(map.occupancy_grid, 
                        interpolation='nearest',
                        cmap='gray_r', 
                        origin='upper',
                        vmax = map.logodd_max_occupied,
                        vmin= map.logodd_min_free)
            plt.pause(.001)
            k = 0
    
    total_time = np.sum(map.time)
    avg_time = np.sum(map.time/n)

    print('--------')
    print('Removing spurious measurements: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[0]/n * 1000, map.time[0]/total_time*100)) 
    print('Converting range to coordinate: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[1]/n * 1000, map.time[1]/total_time*100)) 
    print('Transforming local to global frame: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[2]/n * 1000, map.time[2]/total_time*100)) 
    print('Transforming metric to grid coordinates: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[3]/n * 1000, map.time[3]/total_time*100)) 
    print('Detecting free cells: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[4]/n * 1000, map.time[4]/total_time*100)) 
    print('Updating logodd map: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[5]/n * 1000, map.time[5]/total_time*100)) 
    print('--------')
    print('Average time: {:.2f} ms'.format(np.sum(map.time[0:6]/n) * 1000))
    print('Average frequency: {:.2f} Hz'.format(1/(np.sum(map.time[0:6]/n))))


    input("Press return key to exit")

if __name__ == '__main__':
    main()
