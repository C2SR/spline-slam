# B-spline SLAM
This package contains a python implementation of the B-spline Surface SLAM algorithm, a Simultaneous Localization and Mapping (SLAM) algorithm for range-based measurements using B-spline surfaces maps.

### Examples
Output using the RADISH repository (click on the image to load video)

| Intel Research Lab | ACES building | Freiburg 079 | MIT-CSAIL |
|-|-|-|-|
|  <a href="http://www.youtube.com/watch?feature=player_embedded&v=WaYcoHUqZWs"><img src=examples/images/intel.png alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>  | <a href="http://www.youtube.com/watch?feature=player_embedded&v=kn2fTP_VfbI" target="_blank"><img src=examples/images/aces.png alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>  |  <a href="http://www.youtube.com/watch?feature=player_embedded&v=AP1bM-Znl58" target="_blank"><img src=examples/images/freiburg.png alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>  |  <a href="http://www.youtube.com/watch?feature=player_embedded&v=N1IEwsYbmh0" target="_blank"><img src=examples/images/mit-csail.png alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>  |

### Installation
The algorithm is implemented in python3. For installing the requires python packages, type the following command:

`pip3 install numpy scipy matplotlib`

After that, clone this repository. Open a terminal and navigate until the root folder of the package and enter

`pip3 install -e .` 

### Usage
##### Running from log
The package has an example to run the algorithm from a log file. Each scan reading corresponds to a row in the log using the format described below: 
| timestamp | odom.x | odom.y | odom.theta | range[0] | range[1] | ... | range[n-1] |
|-|-|-|-|-|-|-|-|

**timestamp**: scan reading time in seconds (optional)

**odom.x**: x position estimation from odometry (optional)

**odom.y**: y position estimation from odometry (optional)

**odom.theta**: orientation estimation from odometry (optional)

**range[0]** .. **range[n-1]**: n-readings from the range sensor

Optional fields must be set to 0.0 if no information is available.

