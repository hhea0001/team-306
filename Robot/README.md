# YOLOv5 installation
From the robot directory:
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install


# Robot Setup
From the robot directory:
pip install -r requirements.txt
source ~/LiveDemo/catkin_ws/devel/setup.bash // or whatever
roslaunch penguinpi_gazebo ECE4078.launch
rosrun penguinpi_gazebo scene_manager.py -l map.txt

python3 306.py --search example/search_list.txt

# Main parameters
`--search`, type=str                            Loads the search list from the file

# Other parameters
`--map`, type=str                               Loads the map from the file
`--ip`, type=str. default=localhost             I don't think yolo will load properly if not connected to internet, who knows...
`--port`, type=int, default=40000               
`--speed`, type=int, default=2                  Change the robot speed, can only be in integers
`--radius`, type=float, default=0.20            Radius to dodge obstacles
`--stop`                                        Can be used alone to stop robot from moving, e.g. `python3 306.py --stop`)
`--confidence`, type=float, default=0.75        Changes the minimum confidence for a fruit to be detected
`--max_dist`, type=float, default=0.4           Maximum distance to define two fruit as the same
`--max_target_dist`, type=float, default=1      Maximum distance to record a new location of a target fruit

# For testing

python3 mapping_eval.py --true-map map.txt --slam-est slam_sim_1_306.txt --target-est targets_sim_1_306.txt