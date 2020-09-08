import numpy as np
from matplotlib import pyplot as plt
from spline_slam.spline import SplineMap
import sys
import time

def main():
    if len(sys.argv) < 2:
        print("You must enter a file name")
        sys.exit(-1)

    file_name = sys.argv[1]

    # Instantiating the grid map object
    kwargs_spline_map = {'knot_space': .05, 
                        'map_size': np.array([5.,5.]),
                        'logodd_occupied': .9,
                        'logodd_free': .6}
    map = SplineMap(**kwargs_spline_map)

    # Opening log file
    file_handle = open(file_name, "r")
    # Retrieving sensor parameters
    data = file_handle.readline()  
    data = np.fromstring( data, dtype=np.float, sep=' ' )
    
    # Read the data and build the map
    n = 0
    avg_time = 0
    # Plot
    fig, ax = plt.subplots()
    plt.show(block=False)
    
    k=1
    for data in file_handle:
        # collecting the data
        data = np.fromstring( data, dtype=np.float, sep=' ' )
        pose = np.array(data[0:3]) 
        ranges = data[6:]
        #update the map
        before = time.time()
        map.update_map(pose, ranges)
        avg_time += time.time() - before
        k += 1
        n += 1
        if k > 50:
            #print(pose[0:2])
            ax = plt.imshow(map.ctrl_pts.reshape([map.grid_size[0,0],map.grid_size[1,0]], order='F'),
                            interpolation='nearest',
                            cmap='gray_r',
                            origin='upper',
                            vmax = 100, 
                            vmin=-100)
            #ax.set_extent([map.xy_min,map.xy_max-map.knot_space,map.xy_min,map.xy_max-map.knot_space])
            plt.pause(.001)
            k = 0
    
    ## Computing/printing total and per task time
    total_time = np.sum(map.time[0:5])
    avg_time = np.sum(map.time[0:5]/n)
    print('--------')
    print('Removing spurious measurements: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[0]/n * 1000, map.time[0]/total_time*100)) 
    print('Converting range to coordinate: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[1]/n * 1000, map.time[1]/total_time*100)) 
    print('Transforming local to global frame: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[3]/n * 1000, map.time[3]/total_time*100)) 
    print('Detecting free cells: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[2]/n * 1000, map.time[2]/total_time*100)) 
    print('Updating logodd SPLINE map: {:.2f} ms. Relative time: {:.2f}%'.format(map.time[4]/n * 1000, map.time[4]/total_time*100)) 
    
    print('--------')
    print('Average time: {:.2f} ms'.format(np.sum(map.time[0:5]/n) * 1000))
    print('Average frequency: {:.2f} Hz'.format(1/(np.sum(map.time[0:5]/n))))
    
    input("Hit enter to continue")

if __name__ == '__main__':
    main()
