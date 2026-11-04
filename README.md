# RB2301 Final Project - Obstacle Course

The robot has to get through three sections of a "real-world" obstacle course: a zig-zag corridor with static obstacles, a small door, and dynamic swinging obstacles. You get the robot's pose and LiDAR scan. There is no occupancy grid, and the walls are rectilinear. Any approach is allowed.

Everything you write goes in one file, inside `ObstacleCourseNode.timer_callback()` between the two `###### INSERT CODE HERE ######` lines:

```
src/rb2301_fp/rb2301_fp/obstacle_course.py
```

What you have: `self.pose` (`[x, y, heading]`, metres and degrees, from Gazebo odometry or Optitrack), `self.last_scan` (LiDAR ranges, set once the first scan arrives), `self.move_2D(x, y, turn)` to drive, and `self.is_simulation`, which is set automatically from the LiDAR scan size.

## Set up (simulation)

Requirements: Ubuntu with ROS 2 Jazzy and Gazebo Harmonic, `ros-jazzy-ros-gz`, numpy, and your existing RB2301 workspace (the folder containing `src/`, e.g. `~/Documents/rb2301`).

1. Get this repo: `git clone https://github.com/rudra-8000/rb2301_final_project.git` (or Code > Download ZIP).
2. Copy `fp.sh` and `gz_fp.sh` into the workspace folder, next to `src/`.
3. Copy `src/rb2301_fp` and `src/rb2301_gz` into your workspace's `src/`. The new `rb2301_gz` replaces your old one and has the worlds and launch files for CA1, CA2 and the final project.
4. Build:

```bash
cd ~/Documents/rb2301
chmod +x fp.sh gz_fp.sh
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

`--symlink-install` means edits to `obstacle_course.py` take effect without rebuilding.

## Run

Two terminals, both in the workspace folder:

```bash
./gz_fp.sh     # terminal 1: Gazebo. Wait until the course and robot appear
./fp.sh        # terminal 2: your node
```

The goal is to get past the moving obstacles. To start the robot somewhere else (for example past the gate), change `x:=` and `y:=` in `gz_fp.sh`; a commented example is in the file. Stop with `Ctrl+C`.

To disable both static and moving obstacles while keeping the maze walls and gate:

```bash
./gz_fp.sh disable_obstacles:=true
```

The launch argument defaults to `false`, so `./gz_fp.sh` still generates obstacles.

The unchanged starter code just drives forward at 0.1 m/s.

## RViz visualization (including a node started over SSH)

Visualization is built at 10 Hz in the adjacent `init_rviz_visualization()`,
`_rviz_marker()` and `publish_rviz_visualization()` methods. The controller needs
neither a Matplotlib window nor `DISPLAY`. It supports two transports:

- `visualization_transport:=ros` (default): publish `/obstaclecourse/visualization`
  directly as `visualization_msgs/msg/MarkerArray`. Use this for local simulation
  or compatible ROS installations.
- `visualization_transport:=ssh`: cache plain JSON snapshots at
  `http://127.0.0.1:8765/snapshot`. The local `visualization_relay` reads them through
  an SSH tunnel and publishes the same MarkerArray topic locally. No ROS CDR or
  DDS discovery traffic crosses the tunnel. This is the mode for a Foxy robot
  and a Jazzy laptop.

Do not join the Foxy robot's domain with native Jazzy RViz, `ros2 topic echo`, or
a ROS CLI daemon. Cross-distribution graph messages can crash older middleware;
changing visualization message size or QoS does not isolate discovery.

### Foxy robot + Jazzy laptop: SSH mode

Sync the updated `src/rb2301_fp` package to the robot, including
`visualization_bridge.py` and the updated `setup.py`. The example robot workspace
below matches robot 14; substitute your robot's actual path/address.

On the robot's SSH terminal, keep its existing `ROS_DOMAIN_ID` and driver setup:

```bash
cd /home/bingda/Downloads/rb2301_fp_25
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --packages-select rb2301_fp
source install/setup.bash
bash fp.sh --ros-args -p visualization_transport:=ssh
```

On the laptop, build the updated package once, then use the SSH launcher:

```bash
cd /home/fro4ten/Documents/rb2301_final_project
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select rb2301_fp
bash rviz_fp.sh --ssh bingda@192.168.1.214
```

The launcher authenticates SSH normally, creates a localhost-only `ssh -L`
tunnel, and starts the relay and RViz with `ROS_DOMAIN_ID=166`,
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and `ROS_LOCALHOST_ONLY=1`. It clears
`ROS_STATIC_PEERS`, `ROS_DISCOVERY_SERVER` and `ROS_SUPER_CLIENT` for those child
processes. Exiting RViz or pressing Ctrl+C closes the relay and this launcher's
own tunnel. Existing terminal settings and other SSH sessions are unaffected.
Before switching to this mode, close old Jazzy RViz/echo processes connected to
the robot's domain and stop any CLI daemon started in that environment with
`ros2 daemon stop`; the launcher cannot isolate already running processes.

To inspect the **local relay** topic, use an isolated terminal:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=166
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOCALHOST_ONLY=1
unset ROS_STATIC_PEERS ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT
ros2 topic echo /obstaclecourse/visualization \
  visualization_msgs/msg/MarkerArray --once --timeout 10 --no-arr \
  --qos-reliability reliable --qos-durability transient_local
```

`curl http://127.0.0.1:8765/snapshot` on the robot checks the snapshot server
without creating any ROS participant. HTTP 503 means the node is waiting for
its first pose/frame. The server binds only to loopback and exposes telemetry
only. Slow or disconnected viewers never block the controller on network I/O.
The relay downloads in a separate worker with a 3-second socket timeout and at
most 5 requests per second. Its ROS timer refreshes the most recent valid scene
at 10 Hz, so a slow SSH request cannot block RViz updates. Short gaps retain the
last frame; after 3 seconds without a **new snapshot sequence**, the relay sends
DELETEALL and reports stale data. Repeated HTTP responses carrying the same
sequence do not extend that freshness deadline.

The updated robot server supports HTTP keep-alive and cached gzip payloads,
reducing repeated SSH channel setup and bandwidth. The new relay also works
with the original HTTP/1.0, uncompressed robot server: restarting only the local
launcher already enables the longer timeout, worker and short-gap buffering.
Sync the new `visualization_bridge.py` and restart the robot node to enable
keep-alive/gzip as well.

For a busy local port, launch with `FP_RVIZ_PORT=8766 bash rviz_fp.sh --ssh ...`.
The robot port can be set with `-p visualization_port:=8766`; set
`FP_RVIZ_REMOTE_PORT=8766` on the laptop launcher to match it. An optional
`FP_RVIZ_DOMAIN_ID` changes the launcher's private local domain (default 166).
The launcher also accepts `FP_RVIZ_TIMEOUT` (seconds, default 3),
`FP_RVIZ_STALE_AFTER` (seconds, default 3), and `FP_RVIZ_POLL_RATE` (Hz, default 5).
For example, to tolerate a slower connection while retaining a frame for up to
5 seconds after its last distinct update:

```bash
FP_RVIZ_TIMEOUT=5 FP_RVIZ_STALE_AFTER=5 bash rviz_fp.sh --ssh bingda@192.168.1.214
```

### Local simulation: direct ROS mode

After adding the RViz config, rebuild once and open RViz on the machine with a display:

```bash
cd ~/Documents/rb2301_final_project  # or your workspace folder
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select rb2301_fp
bash rviz_fp.sh
```

The config is also installed with the package, so you can run:

```bash
source install/setup.bash
rviz2 -d "$(ros2 pkg prefix rb2301_fp)/share/rb2301_fp/rviz/obstacle_course.rviz"
```

Colors follow `dynamic_window_approach.animate()`: green predicted trajectory,
red robot position/heading, blue reference goal, dark LiDAR points/footprint,
and colored segmentation points. The cyan arrow is the active `_sub_goal`
heading. The blue cross shows `_goal.goal_node` when a positional goal is active.
Both rectangular and circular footprints use the active DWA config.

The bundled view uses `Fixed Frame: fp_visualization` and the RViz `Identity`
transformation, because every marker is already in the same coordinates as
`self.pose` and `_ob`: Gazebo odom in simulation, CA2 maze coordinates on the
real robot. It needs neither TF nor a robot model. This view does not establish
an alignment between Optitrack maze coordinates and an existing `map` frame.
Do not add topics from other frames to this Identity view without transforming
them first. If the node is renamed or namespaced, update the MarkerArray topic
in RViz to match its private `~/visualization` topic.

Direct ROS viewing between separate machines is for compatible distributions:
use the same domain and working DDS connectivity. For Foxy/Jazzy use the SSH
mode above. See the official [ROS 2 discovery documentation](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Tutorials/Advanced/Improved-Dynamic-Discovery.rst)
and [OpenSSH forwarding documentation](https://man.openbsd.org/ssh.1).

The publisher and bundled RViz display use reliable, transient-local QoS with
depth 1, so RViz can join after the node starts. Empty layers are explicitly
deleted, and markers expire after one second without updates. Calibration mode
also publishes the available pose and scan without requiring a DWA prediction.
To disable visualization:

```bash
./fp.sh --ros-args -p visualization_enabled:=false
```

## The simulation map

- Each time you launch, the map is randomised from `obstacle_modifier.py`: gate position, zig-zag size, obstacle speed, and the position of the static obstacles. Switch each one on or off with the `randomise_...` variables at the top of that file.
- The untouched map is `obstacle_course_world_fp_original.sdf`. Do not edit it; the launched map is generated from it.
- The moving obstacles have no collision in the simulation.

## Problems

- `Permission denied` on a `.sh` file: `chmod +x fp.sh gz_fp.sh`.
- Gazebo hangs on "Requesting list of world names": an old Gazebo is still running. `pkill -9 ruby`, then run `./gz_fp.sh` again.
- Nothing happens when you run `./fp.sh`: Gazebo has to be fully loaded first (check terminal 1 is still running). The node waits for the LiDAR before it starts.
- `ros2: command not found`: `source /opt/ros/jazzy/setup.bash`. Run the scripts from the workspace folder (the one with `install/`).
- Gazebo window black, slow or crashing (virtual machine, no GPU): in `src/rb2301_gz/launch/fp_gazebo.launch.py` set `LIBGL_ALWAYS_SOFTWARE` to `"1"`.

## Real robot

Same flow as the slides. `NN` is your robot's number in two digits (robot 7 is `07`, address `192.168.1.207`), `<group>` is your group number. You edit on your laptop and copy the package to the robot.

**1. Make your workspace on the robot** (ssh in as usual):

```bash
ssh bingda@192.168.1.2NN
mkdir -p ~/Downloads/rb2301_fp_<group>/src
```

**2. Copy the package and the script** from your laptop's own terminal (not the ssh one). Repeat after every edit:

```bash
rsync -auvx --delete ~/Documents/rb2301/src/rb2301_fp bingda@192.168.1.2NN:~/Downloads/rb2301_fp_<group>/src
rsync -auvx --delete ~/Documents/rb2301/fp.sh bingda@192.168.1.2NN:~/Downloads/rb2301_fp_<group>
```

**3. Build on the robot** (ssh terminal):

```bash
cd ~/Downloads/rb2301_fp_<group>
colb                    # alias for colcon build --symlink-install
```

**4. Run each of these in its own ssh terminal** (or use `tmux` on the robot to split one ssh into several terminals):

```bash
basecontrol             # alias for ros2 launch base_control_ros2 base_control.launch.py
rplidar                 # alias for ros2 launch rplidar_ros rplidar.launch.py
vrpn                    # alias for ros2 launch vrpn_mocap client.launch.yaml server:=192.168.1.199 port:=3883
cd ~/Downloads/rb2301_fp_<group> && ./fp.sh
```

Add `--calibrate` to the last command (`./fp.sh --calibrate`) to only print your pose and never drive. Stay out of the Optitrack area while the robot runs.

What is different from the simulation:

- **Pose frame.** On the real robot `self.pose` is in the **CA2 maze frame**: metres from the corner of the CA2 maze, heading 0 along its +x axis and +90 along +y (counter-clockwise positive). The mapping from Optitrack to that frame is in `src/rb2301_fp/rb2301_fp/optitrack_variables.config`, already filled in. Don't edit it. If the calibration changes you will get a new file to drop in.
- **Course coordinates.** In the simulation the robot starts at `(0, 0)` facing +x. If you want the same numbers on the real robot, `self.to_course()` converts a pose from the CA2 maze frame to the course frame. It needs the position of the course start point in the maze frame (`[course]` in the config), which is not set yet, so for now it returns the pose unchanged.
- **Robot number.** The node reads the Optitrack topic of its own robot from `ROS_DOMAIN_ID` (already set on each robot) and prints it at start-up (`/vrpn_mocap/bingda_0NN/pose`).
- **Rigid bodies** are created in Motive with the robot facing +y of the CA2 maze. If you reset one, face the robot +y first.
- **Checking the pose.** Run the node with `--calibrate`, for example `./fp.sh --calibrate` on the robot. It prints the raw Optitrack pose and the maze-frame pose twice a second and never sends a drive command, so you can move the robot by hand and read off coordinates (for example the position of the gate). Start `rplidar` first, otherwise the node waits for the LiDAR.
