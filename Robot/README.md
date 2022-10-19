# YOLOv5 installation
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install


# Robot Setup
pip install -r requirements.txt
source ~/LiveDemo/catkin_ws/devel/setup.bash // or whatever
roslaunch penguinpi_gazebo ECE4078.launch
rosrun penguinpi_gazebo scene_manager.py -l map.txt

python3 306.py --map example/map.txt --search example/search_list.txt

# Main parameters
`--map`, type=str                               Loads the map from the file
`--search`, type=str                            Loads the search list from the file

# Other parameters
`--fruit`                                       Enables fruit yolo for lvl3, e.g. `python3 306.py --map example/map.txt --search example/search_list.txt --fruit`
`--ip`, type=str. default=localhost             I don't think yolo will load properly if not connected to internet, who knows...
`--port`, type=int, default=40000               
`--speed`, type=int, default=2                  Change the robot speed, can only be in integers
`--obstacle_radius`, type=float, default=0.20   Radius to dodge obstacles
`--robot_radius`, type=float, default=0.1       Extra radius on top of obstacles to prevent robot getting stuck, requires guessing and checking to find good balance
`--stop`                                        Can be used alone to stop robot from moving, e.g. `python3 306.py --stop`)