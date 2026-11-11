import argparse
import configparser
import math
import os
import time
import numpy as np
import rclpy
from typing import List, Tuple
from rb2301_fp import dynamic_window_approach

from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> CA2 maze-frame pose (see optitrack_variables.config).'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))

def maze_to_course(x:float, y:float, heading_deg:float, course:dict) -> tuple:
    '''CA2 maze-frame pose -> final-project course frame (origin/rotation of the course in the maze frame, [course] in the config).'''
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_deg(heading_deg - course["rotation_deg"]))

def load_frame(robot_number:int) -> tuple:
    '''Read optitrack_variables.config: returns (frame, course) for this robot.'''
    parser = configparser.ConfigParser()
    path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read {path}")
    frame = {k: parser.getfloat("frame", k) for k in ("origin_x", "origin_y", "rotation_deg")}
    raw = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    # Rigid bodies are created facing +y of the maze frame, so Motive's zero heading is maze +y (90 deg)
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw == "auto" else float(raw)
    if parser.has_section("heading_offsets") and parser.has_option("heading_offsets", str(robot_number)):
        frame["heading_offset_deg"] = parser.getfloat("heading_offsets", str(robot_number))
    course = {k: parser.getfloat("course", k, fallback=0.0) for k in ("origin_x", "origin_y", "rotation_deg")}
    return frame, course


class LidarNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("LidarNode")
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, qos_profile_sensor_data) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360:
            self.is_simulation = True
        else:
            self.is_simulation = False
class Wall():
    def __init__(self,para,center=None,edge = None):
        self.para=para#para = (A,B,C)
        self.center=center
        self.edge = edge#edge = [(first_dot),(last_dot)]

class Goal():
    def __init__(self,goal_heading=None,goal_node=None):
        self._goal_heading = goal_heading
        self._goal_node = goal_node

    @property
    def goal_heading(self):
        return self._goal_heading
    
    @goal_heading.setter
    def goal_heading(self, value):
        self._goal_heading = value
        self._goal_node = None

    @property
    def goal_node(self):
        return self._goal_node
    
    @goal_node.setter
    def goal_node(self, value):
        self._goal_node = value
        self._goal_heading = None
#============================================node==================================================================================
class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=True, calibrate:bool=False, *, parameter_overrides=None):
        super().__init__('obstaclecourse', parameter_overrides=parameter_overrides)
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation
        self.dwa_config = dynamic_window_approach.Config(is_simulation=self.is_simulation)
        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0

        self.actual_velocity = None
        self.last_timestamp = None

        #DWA
        self._state = np.array([0.0, 0.0, 0.0 , 0.0, 0.0])  #state representation:
        #                      [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        self._goal = Goal()


        # Temporary single-goal mode; disable the parameter to restore course stages.
        self.single_point_goal = self.declare_parameter('single_point_goal', True).value
        if is_simulation:
            self.point_goal_distance = self.declare_parameter('point_goal_distance', 8.0).value
            self.point_goal_left_angle_deg = self.declare_parameter(
                'point_goal_left_angle_deg', -35.0).value
        else:
            self.point_goal_distance = self.declare_parameter('point_goal_distance', 8.0).value
            self.point_goal_left_angle_deg = self.declare_parameter(
                'point_goal_left_angle_deg', 155.0).value


        
        if (not math.isfinite(self.point_goal_distance) or self.point_goal_distance <= 0
                or not math.isfinite(self.point_goal_left_angle_deg)):
            raise ValueError('point_goal_distance must be positive and goal settings must be finite')
        self._point_goal_reached = False

        if self.is_simulation == False:
            self._sub_goal = ['-y','-x','+y']# sub_goal_heading, in reverse order
            self._sub_goal_2 = ['-y']
        else:
            self._sub_goal = ['+x','-y','+x']
            self._sub_goal_2 = ['+x']
        self._initial_goal_heading = self._sub_goal[-1]
        if not self.single_point_goal:
            self._goal.goal_heading = self._initial_goal_heading
        self._start_stage_started = False
        self._gate_stage_started = False
        self._final_stage_started = False

        self._trajectory = np.array(self._state)
        self._ob = np.array([]) #obstacle array
        self.scan_step = 2 # LiDAR downsampling: keep every Nth beam (1 = all, 2 = half, 4 = quarter)
        self._is_dwa_initialized = False


        #======Deadend checker and goal switch params======#
        self.density_threshold = 0.25#the obstacle dot density threshhold for detecting deadend
        self.check_list_threshold = 15 # how many dot needed to start determine if it is a wall
        self.outlier_dot_threshold = 0.2 # The distance from the point with the outlier dot to the fitted line
        self.outlier_offset_threshold = 1
        self.tilt_threshold = 60 # in angular
        self.close_threshold = 1# how close to a deadend triggers transformation to the next _sub_goal
        self.deadend_confirm_checks = 3 # consecutive checks of the same wall
        self.deadend_wall_tolerance = 0.15 # fitted wall offset tolerance, metres
        self._deadend_candidate = None
        self._deadend_hits = 0
        self.curve_threshold = 85# beyond the angle will not be seen as a single wall
        self.sub_goal_cd =3.0# sub_goal switch cooldown,in seconds
        self.timestamp = 0.0#
        self._segmentation = []
        self.deadend_check_gap = 10# how many times the timercallback run required for deadend check once
        self.timer_callback_counter = 1
        self.farthest_dot = None
        self.wall_list = []
        self.gate_mse_threshold = 0.1
        self.min_gate_width_threshold=0.6
        self.max_gate_width_threshold=0.8
        self.stage = 'point_goal' if self.single_point_goal else 'start'
        
        if is_simulation:
            self.max_translate_velocity = 0.5
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, qos_profile_sensor_data) # Subscribe to LiDAR scan data

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, qos_profile_sensor_data)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            # On the Bingda robots ROS_DOMAIN_ID is the robot's number, so the Optitrack topic is picked automatically
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            self.robot_number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.frame, self.course = load_frame(self.robot_number)
            self.get_logger().info(f"Reading pose from /vrpn_mocap/bingda_{self.robot_number:03d}/pose")

            self.map_sub = self.create_subscription(
                PoseStamped,
                f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose',
                self.optitrack_callback,
                qos_profile
                )

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.pose = None
        self.init_rviz_visualization()
#==============================================================================================================
    # RViz visualization
    def init_rviz_visualization(self):
        """Choose direct ROS visualization or JSON snapshots for SSH viewing."""
        self._rviz_predicted_trajectory = None
        self._rviz_server = None
        enabled = self.declare_parameter('visualization_enabled', True).value
        self._rviz_frame = self.declare_parameter(
            'visualization_frame', 'fp_visualization').value
        transport = self.declare_parameter('visualization_transport', 'ros').value
        if not enabled:
            return
        if transport == 'ssh':
            from rb2301_fp.visualization_bridge import VisualizationSnapshotServer
            port = self.declare_parameter('visualization_port', 8765).value
            self._rviz_server = VisualizationSnapshotServer(port)
            self.get_logger().info(
                'SSH visualization: http://127.0.0.1:{}/snapshot'.format(self._rviz_server.port))
        elif transport == 'ros':
            # Keep the latest complete scene for RViz instances that join later.
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self._rviz_publisher = self.create_publisher(
                MarkerArray, '~/visualization', qos)
        else:
            raise ValueError('visualization_transport must be ros or ssh')
        self._rviz_timer = self.create_timer(0.1, self.publish_rviz_visualization)

    def _rviz_marker(self, name, marker_type, points, color, size, stamp, z=0.03):
        """Build one marker in controller coordinates; empty points delete it."""
        marker = Marker()
        marker.header.frame_id = self._rviz_frame
        marker.header.stamp = stamp
        marker.ns = name
        marker.id = 0
        marker.type = marker_type
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = float(size)
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
        marker.lifetime.sec = 1
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=z) for p in points]
        marker.action = Marker.ADD if marker.points else Marker.DELETE
        return marker

    def publish_rviz_visualization(self):
        """RViz equivalent of DWA animate(), plus the active sub-goal heading.

        All data use self.pose's coordinates (Gazebo odom or Optitrack maze).
        The standalone fp_visualization frame needs no TF when used as RViz's
        Fixed Frame. This timer also runs while calibration/control is waiting.
        """
        if self.pose is None:
            return
        stamp = self.get_clock().now().to_msg()
        position = np.asarray(self.pose[:2])
        yaw = math.radians(self.pose[2])
        scene = MarkerArray()

        predicted = self._rviz_predicted_trajectory
        if (self.calibrate or (not self.single_point_goal and not self._sub_goal) or self._ob.size == 0
                or (predicted is not None and len(predicted) < 2)):
            predicted = None
        scene.markers.append(self._rviz_marker(
            'predicted_trajectory', Marker.LINE_STRIP,
            [] if predicted is None else predicted[:, :2],
            (0.0, 0.7, 0.0), 0.025, stamp))
        scene.markers.append(self._rviz_marker(
            'robot_position', Marker.SPHERE_LIST, [position],
            (1.0, 0.0, 0.0), 0.07, stamp, z=0.08))
        scene.markers.append(self._rviz_marker(
            'obstacles', Marker.POINTS, np.asarray(self._ob).reshape(-1, 2),
            (0.1, 0.1, 0.1), 0.035, stamp))

        # Blue cross marks the active point goal.
        goal = [] if self._goal.goal_node is None else np.asarray(self._goal.goal_node) + np.array(
            [[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1], [0.1, -0.1]])
        scene.markers.append(self._rviz_marker(
            'reference_goal', Marker.LINE_LIST, goal,
            (0.0, 0.2, 1.0), 0.035, stamp))

        config = self.dwa_config
        if config.robot_type == dynamic_window_approach.RobotType.rectangle:
            outline = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1], [-1, 1]])
            outline = outline * [config.robot_length / 2, config.robot_width / 2]
        else:
            angles = np.linspace(0.0, 2 * math.pi, 49)
            outline = config.robot_radius * np.column_stack(
                (np.cos(angles), np.sin(angles)))
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                             [math.sin(yaw), math.cos(yaw)]])
        scene.markers.append(self._rviz_marker(
            'robot_footprint', Marker.LINE_STRIP, outline @ rotation.T + position,
            (0.1, 0.1, 0.1), 0.02, stamp, z=0.06))
        heading = [position, position + 0.5 * np.array([math.cos(yaw), math.sin(yaw)])]
        arrow = self._rviz_marker('robot_heading', Marker.ARROW, heading,
                                  (1.0, 0.0, 0.0), 0.04, stamp, z=0.1)
        arrow.scale.y, arrow.scale.z = 0.08, 0.12
        scene.markers.append(arrow)

        direction = {'+x': (1, 0), '-x': (-1, 0), '+y': (0, 1), '-y': (0, -1)}
        sub_goal = []
        if self._goal.goal_heading:
            sub_goal = [position, position + 0.8 * np.array(direction[self._goal.goal_heading])]
        arrow = self._rviz_marker('sub_goal_heading', Marker.ARROW, sub_goal,
                                  (0.0, 0.65, 0.9), 0.035, stamp, z=0.12)
        arrow.scale.y, arrow.scale.z = 0.09, 0.12
        scene.markers.append(arrow)

        # One POINTS marker with per-point colors avoids stale segment IDs.
        palette = [(1.0, 0.4, 0.0), (0.6, 0.0, 0.8), (0.0, 0.65, 0.65),
                   (0.8, 0.55, 0.0), (0.9, 0.0, 0.5), (0.3, 0.4, 1.0)]
        points, colors = [], []
        for i, segment in enumerate(self._segmentation):
            points.extend(segment)
            r, g, b = palette[i % len(palette)]
            colors.extend([ColorRGBA(r=r, g=g, b=b, a=1.0) for _ in segment])
        segments = self._rviz_marker('segmentation', Marker.POINTS, points,
                                     (1.0, 1.0, 1.0), 0.065, stamp, z=0.05)
        segments.colors = colors
        scene.markers.append(segments)
        if self._rviz_server is not None:
            try:
                self._rviz_server.publish(scene)
            except (ValueError, TypeError) as error:
                self.get_logger().warning('Visualization frame skipped: {}'.format(error))
        else:
            self._rviz_publisher.publish(scene)

    def destroy_node(self):
        if self._rviz_server is not None:
            self._rviz_server.close()
        return super().destroy_node()


#========Utilities=======#
    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        self.last_scan = np.array(msg.ranges)[::self.scan_step]
        self.update_ob(msg)

    def update_ob(self,msg):
        if self.pose is None:
            return
        r = self.last_scan
        yaw = np.deg2rad(self.pose[2])
        # LiDAR mounting from nanocar_description.urdf: x=0.11423 m, yaw=3.1416 rad.
        # Each kept beam is scan_step raw beams apart, so the angle step scales with it.
        a = msg.angle_min + np.arange(r.size) * msg.angle_increment * self.scan_step + 3.1416 + yaw
        valid = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        x = self.pose[0] + 0.11423 * np.cos(yaw)
        y = self.pose[1] + 0.11423 * np.sin(yaw)
        self._ob = np.column_stack((x + r[valid] * np.cos(a[valid]), y + r[valid] * np.sin(a[valid])))

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        self.pose = np.array((x,y,heading))
        self._initialize_point_goal()
        self.update_actual_velocity(msg.header.stamp)
        return self.pose

    def to_course(self, pose=None):
        '''Convert a pose [x, y, heading] (default: self.pose) from the CA2 maze frame to the final-project course frame. Optional; see [course] in optitrack_variables.config'''
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(pose[0], pose[1], pose[2], self.course)) if self.course else np.array(pose)

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        self._initialize_point_goal()
        self.update_actual_velocity(msg.header.stamp)
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def _initialize_point_goal(self):
        """Freeze one target relative to the first pose and initial course direction."""
        if (not self.single_point_goal or self._goal.goal_node is not None
                or self.pose is None or not np.isfinite(self.pose).all()):
            return
        forward_yaw = {'+x': 0.0, '-x': math.pi,
                       '+y': math.pi / 2, '-y': -math.pi / 2}[self._initial_goal_heading]
        angle = forward_yaw + math.radians(self.point_goal_left_angle_deg)
        offset = self.point_goal_distance * np.array([math.cos(angle), math.sin(angle)])
        self._goal.goal_node = self.pose[:2] + offset
        self.get_logger().info(
            f'Fixed point goal: {self._goal.goal_node}, '
            f'{self.point_goal_distance} m at {self.point_goal_left_angle_deg} deg left '
            f'of initial {self._initial_goal_heading}; course switching disabled')

    def is_goal_reached(self,threshold = 0.3)->bool:
        if self._goal.goal_node is None:
            return False
        dist_to_goal = math.hypot(self._state[0] - self._goal.goal_node[0], self._state[1] - self._goal.goal_node[1])
        if dist_to_goal <= threshold:
            print("Goal!!")
            return True
        return False

    def update_state(self)->np.ndarray:
        """update speed and orientation using pose[], keep vel and omega"""
        return np.array([self.pose[0], self.pose[1], np.deg2rad(self.pose[2]), self.actual_velocity[0], self.actual_velocity[1] ])

    def update_actual_velocity(self, stamp):
        """use differentiation of position to estimate the forward speed(m/s) and angular speed(rad/s)"""
        timestamp = stamp.sec + stamp.nanosec * 1e-9
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt == 0:
                return  
            if dt < 0:
                self.actual_velocity = None
                self.filtered_forward_velocity = None
            else:
                dx, dy = self.pose[:2] - self.last_pose[:2]
                yaw = math.radians(self.pose[2])
                previous_yaw = math.radians(self.last_pose[2])
                delta_yaw = math.atan2(math.sin(yaw - previous_yaw),
                                       math.cos(yaw - previous_yaw))
                midpoint_yaw = previous_yaw + delta_yaw / 2
                v = (dx * math.cos(midpoint_yaw) + dy * math.sin(midpoint_yaw)) / dt
                self.actual_velocity = (v, delta_yaw / dt)
        self.last_pose = self.pose.copy()
        self.last_timestamp = timestamp

    def set_dwa_param(self,to_goal_heading=None, to_goal_node=None,speed=0.5, ob=0.01, pre_time=1.5, v_res=0.01):  #Update DWA param online
        self.dwa_config.speed_cost_gain = speed
        self.dwa_config.obstacle_cost_gain = ob
        self.dwa_config.to_goal_heading_cost_gain = to_goal_heading
        self.dwa_config.to_goal_cost_gain = to_goal_node
        self.dwa_config.predict_time = pre_time
        self.dwa_config.v_resolution = v_res
#=====================================stage_para_initialization==========================================
    def start_stage(self):
        if self._start_stage_started:
            return
        self._start_stage_started = True
        self.set_dwa_param(to_goal_heading=0.1,speed=1.2,ob=0.1,pre_time=1.5,v_res=0.02)
        self.get_logger().info('enter start stage!dwa param changed')

    def final_stage(self):
        if self._final_stage_started:
            return
        self._final_stage_started = True
        self.close_threshold /= 2
        self.set_dwa_param(to_goal_heading=0.5,speed=1.5,ob=0.5,pre_time=1.5,v_res=0.01)
        self.get_logger().info('enter final stage!dwa param changed')

    def gate_stage(self):
        if self._gate_stage_started:
            return
        self._gate_stage_started = True
        self.close_threshold /= 2
        self.set_dwa_param(to_goal_node=0.5,speed=1,ob=0.2,pre_time=1.5,v_res=0.01)
        self.get_logger().info('enter gate stage!dwa param changed')
#===============================timer_callback=====================================================
    def timer_callback(self):   #timer callback framework for performance evaluation
        callback_started = time.perf_counter()
        timings = {'DWA': None, 'deadend_detect': None}
        try:
            self._run_timer_callback(timings)
        finally:
            callback_ms = (time.perf_counter() - callback_started) * 1000.0
            phases = ', '.join(
                f'{name}={elapsed * 1000.0:.3f} ms' if elapsed is not None
                else f'{name}=skipped'
                for name, elapsed in timings.items())
            self.get_logger().info(
                f'Performance: {phases}, timer_callback={callback_ms:.3f} ms')

    def _run_timer_callback(self, timings): #real timer_callback
        """Controller loop; deadend timing includes segmentation and wall fitting."""
        if self.pose is None:
            print("No pose detected")
            return # Does not run if no pose received from Odom or Optitrack

        if self.calibrate and self.raw_pose is not None:
            self._ticks += 1
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                #self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)")
            return # Calibration mode: look, don't drive

        #self.get_logger().info(f"Pose: {self.pose}")

        self._initialize_point_goal()
        
        if self.single_point_goal:
            if self._goal.goal_node is None:
                self.move_2D()
                return
            if self._point_goal_reached or self.is_goal_reached():
                self._point_goal_reached = True
                self.stage = 'done'
                self._rviz_predicted_trajectory = None
                self._state[3:] = 0.0
                self.move_2D()
                return

        #DWA integration
        if self.pose is None or self._ob.size == 0 or self.actual_velocity is None:
            return
        if not self.single_point_goal and not self._sub_goal:
            self.move_2D()
            return
        
        self._state = self.update_state()   #Update the state

        if not self.single_point_goal and self.stage == 'start':
            self.start_stage()
            if self.timer_callback_counter % self.deadend_check_gap == 0:
                self.timer_callback_counter = 1
                deadend_started = time.perf_counter()
                try:
                    self._segmentation = []
                    self.segment_chipper()#genrate _segmentation
                    self.wall_list=[]
                    self.wall_generator()
                    #Detect deadend_wall and change subgoal
                    deadend_wall = self.deadend_detector()
                finally:
                    timings['deadend_detect'] = time.perf_counter() - deadend_started

                # Only the final sub-goal searches for the exit gate.
                gate = self.gate_detector("+x")
                if gate is not None:
                    self._goal.goal_node = gate
                    self.stage = 'gate'

                else:
                    if self.confirm_deadend(deadend_wall):
                        self._sub_goal.pop()
                        self.timestamp = time.monotonic()
                        if not self._sub_goal:
                            self._goal.goal_heading = self._sub_goal_2[-1]

                        self._goal.goal_heading = self._sub_goal[-1]
                        self.get_logger().info(
                            f'switch to the next _sub_goal!{self._sub_goal[-1]}'
                        )

            else:
                self.timer_callback_counter+=1


        if not self.single_point_goal and self.stage == 'gate':
            self.gate_stage()
            if self.is_goal_reached():
                self._goal.goal_node = None
                self.get_logger().info('gate reached')
                self.stage = 'final'


        if not self.single_point_goal and self.stage == 'final':
            self._goal.goal_heading = self._sub_goal_2[-1]
            self.final_stage()



#===========================DWA======================================================================
        self._trajectory = np.vstack((self._trajectory, self._state))
        dwa_started = time.perf_counter()
        try:
            u, predicted_trajectory = dynamic_window_approach.dwa_control(
                self,
                x=self._state,
                goal=self._goal.goal_node,
                goal_heading=None if self.single_point_goal else self._goal.goal_heading,
                ob=self._ob,
                config=self.dwa_config,
            )
        finally:
            timings['DWA'] = time.perf_counter() - dwa_started
        self._rviz_predicted_trajectory = predicted_trajectory
        self.move_2D(u[0],0.0,u[1])
        #dynamic_window_approach.animate(predicted_trajectory,self._state,self._goal,self._ob, self._segmentation)


#===========================MATH=====================================================================
    @staticmethod
    def measure_distance_dot_and_line(dot,line):
        A = line[0]
        B = line[1]
        C = line[2]
        _dot = np.asarray(dot)[:2]
        norm = np.hypot(A, B)
        if norm == 0:
            raise ValueError('A and B cannot both be zero')
        return float(np.abs(_dot @ np.array([A, B]) + C) / norm)

    @staticmethod
    def measure_angle(vector_1,vector_2):
            norm_product = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
            cos_theta = np.dot(vector_1, vector_2) / norm_product
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            return angle_deg
    
#
    def wall_generator(self):
        for segment in self._segmentation:
            if len(segment) < self.check_list_threshold:
                continue

            points = np.asarray(segment, dtype=float)
            center = points.mean(axis=0)
            centered = points - center

            _, s, vh = np.linalg.svd(centered, full_matrices=False)
            # 第一主方向沿着墙，第二主方向垂直于墙
            # vh[0]点云延伸最明显的方向,vh[1]法线方向
            #s[0]点云沿第1个方向的分散程度
            if s[0] < 1e-12:
                continue

            A, B = vh[1]  # 单位法向量，A²+B²=1
            C = -np.dot([A, B], center)
            this_wall = Wall((A,B,C),center=center,edge=((segment[0]),(segment[-1])))
            self.wall_list.append(this_wall)
    def confirm_deadend(self, wall):
        """Require repeated observations of one nearby wall."""
        distance = (self.measure_distance_dot_and_line(self.pose, wall)
                    if wall is not None else float('inf'))
        if (distance > self.close_threshold
                or time.monotonic() < self.timestamp + self.sub_goal_cd):
            self._deadend_candidate = None
            self._deadend_hits = 0
            return False

        candidate = np.asarray(wall, dtype=float)
        candidate = candidate / np.linalg.norm(candidate[:2])
        previous = self._deadend_candidate
        if previous is not None and np.dot(candidate[:2], previous[:2]) < 0:
            candidate = -candidate
        same_wall = (previous is not None
                     and np.dot(candidate[:2], previous[:2]) >= math.cos(math.radians(self.tilt_threshold))
                     and abs(candidate[2] - previous[2]) <= self.deadend_wall_tolerance)
        if same_wall:
            self._deadend_hits += 1
        else:
            self._deadend_candidate = candidate
            self._deadend_hits = 1
        if self._deadend_hits < self.deadend_confirm_checks:
            return False
        self._deadend_candidate = None
        self._deadend_hits = 0
        return True

    def check_segment(self,wall,farthest_dot):# for each segments, detect if it is a deadend wall
        '''
        check if it is a deadend
        '''
        A,B,C = wall.para
        center = wall.center

        x_distance = self.measure_distance_dot_and_line(farthest_dot,(A,B,C))
        y_distance = r = np.hypot(farthest_dot[0]-center[0], farthest_dot[1]-center[1])

        if x_distance > self.outlier_dot_threshold and y_distance < self.outlier_offset_threshold:
            self.get_logger().warn(f'infeasible wall! its not the farthest one:distance:{x_distance}')
            return None

        goal_direction = {
            '+x': (1, 0), '-x': (-1, 0),
            '+y': (0, 1), '-y': (0, -1),
        }[self._sub_goal[-1]]
        normal = np.array([A, B])
        # Reject a wall behind the robot along the current goal direction.
        robot_side = normal @ self.pose[:2] + C
        if robot_side * np.dot(normal, goal_direction) > 0:
            self.get_logger().warn(f'infeasible wall!it is behind the robot')
            return None

        vector_1 = np.array([-B, A])  # 沿墙方向，与直线法向量 (A, B) 垂直
        if self._sub_goal[-1] == '+x' or self._sub_goal[-1]=='-x':
            vector_2 = np.array([1,0])
        else:
            vector_2 = np.array([0,1])

        angle_deg = self.measure_angle(vector_1,vector_2)
        # A blocking wall is perpendicular to travel; SVD signs are arbitrary.
        angle = abs(angle_deg-90.0)
        if angle > self.tilt_threshold:
            self.get_logger().warn(f'infeasible wall!it is not in good angle{angle}')
            return None

        # The finite wall must span the robot's forward path, not merely have
        # an infinite extension in front of it. Require coverage of its width.
        direction = np.asarray(goal_direction, dtype=float)
        lateral = np.array([-direction[1], direction[0]])
        edges = np.asarray(wall.edge) - self.pose[:2]
        side = edges @ lateral
        half_width = self.dwa_config.robot_width / 2
        #if side.min() > -half_width or side.max() < half_width:
        #    return None

        return (A,B,C)
    
    def segment_chipper(self):
        if self.pose is None or self._ob.size == 0 or not self._sub_goal:
            return None

        # Find the global extreme before accepting any candidate segment.
        self.farthest_dot=self.pose[:2]
        last_dot = None
        
        for dot in self._ob:
            if self._sub_goal[-1] == '+x':# update the farthest dot
                if dot[0]>self.farthest_dot[0]:
                    self.farthest_dot = dot
            elif self._sub_goal[-1] == '-x':
                if dot[0]<self.farthest_dot[0]:
                    self.farthest_dot = dot
            elif self._sub_goal[-1] == '+y':
                if dot[1]>self.farthest_dot[1]:
                    self.farthest_dot = dot
            elif self._sub_goal[-1] == '-y':
                if dot[1]<self.farthest_dot[1]:
                    self.farthest_dot = dot
            else:
                raise ValueError('_sub_goal should only contain "+x","-x","+y"or"-y"')

        continuous_list=[]
        aver_hist_vector = None
        fracture = False
        cache_segment=[]
        fractured_flag = False
        for dot in self._ob:
            if last_dot is not None:
                distance = abs(last_dot[0]-dot[0]) + abs(last_dot[1]-dot[1])

                new_vector = np.asarray(last_dot)-np.asarray(dot)
                if aver_hist_vector is None and last_dot is not None:
                    aver_hist_vector = np.asarray(last_dot)-np.asarray(dot)
                if aver_hist_vector is not None:
                    angle = self.measure_angle(new_vector,aver_hist_vector)
                    if angle < self.curve_threshold:
                        aver_hist_vector= aver_hist_vector/5+new_vector*4/5
                        fracture = False
                    else:
                        fracture = True
                        aver_hist_vector = new_vector
            if (last_dot is not None and distance > self.density_threshold) or (aver_hist_vector is not None and fracture == True):  #once the continuous list fracture or reaches the end
                if len(continuous_list) >= self.check_list_threshold:
                    self._segmentation.append(continuous_list)
                continuous_list=[dot]
                fractured_flag = True
            else:
                if fractured_flag == False:
                    cache_segment.append(dot)
                else:
                    continuous_list.append(dot)
            last_dot = dot
        continuous_list =continuous_list + cache_segment
        self._segmentation.append(continuous_list)

    def deadend_detector(self):
        '''
        for each segment,check if it it a deadend
        '''
        for wall in self.wall_list:
            result = self.check_segment(wall=wall,farthest_dot=self.farthest_dot)
            if result is not None:
                return result
        return None
    
    def gate_detector(self,self_goal_heading):
        candidate_walls = []

        for wall in self.wall_list:
            A,B,C = wall.para
            goal_direction = {
                '+x': (1, 0), '-x': (-1, 0),
                '+y': (0, 1), '-y': (0, -1),
            }[self_goal_heading]
            normal = np.array([A, B])
            #filter walls that perpendicular to travel or in the other side
            robot_side = normal @ self.pose[:2] + C
            if robot_side * np.dot(normal, goal_direction) > 0:
                continue

            vector_1 = np.array([-B, A])  
            if self_goal_heading == '+x' or self_goal_heading=='-x':
                vector_2 = np.array([1,0])
            else:
                vector_2 = np.array([0,1])
            angle_deg = self.measure_angle(vector_1,vector_2)
            if abs(angle_deg - 90.0) > self.tilt_threshold:
                continue
            candidate_walls.append(wall)

        if len(candidate_walls) < 2:
            self.get_logger().debug(
                f'Gate detection: segments={len(self._segmentation)}, '
                f'walls={len(self.wall_list)}, candidates={len(candidate_walls)}, '
                f'segment_sizes={[len(s) for s in self._segmentation]}'
            )
            return None
        min_mse = float('inf')
        best_pair = None
        for i, wall_1 in enumerate(candidate_walls):
            for wall_2 in candidate_walls[i + 1:]:
                p1 = np.asarray(wall_1.para)
                p2 = np.asarray(wall_2.para)
                # SVD may give opposite normals for the same line.
                # Keep the squared-error sum to preserve the threshold scale.
                mse = min(np.sum((p1 - p2)**2), np.sum((p1 + p2)**2))
                if mse < min_mse:
                    min_mse = mse
                    best_pair = (wall_1, wall_2)

        if best_pair is not None and min_mse < self.gate_mse_threshold:
            wall_1, wall_2 = best_pair
            gate_width = abs(np.asarray(wall_1.edge[1][1])-np.asarray(wall_2.edge[0][1]))
            if gate_width < self.min_gate_width_threshold:
                
                return None
            if gate_width > self.max_gate_width_threshold:
                return None
            gate_center = (np.asarray(wall_1.edge[1])+np.asarray(wall_2.edge[0]))/2
            self.get_logger().error(f'detected a gate!{gate_center}')
            gate_center[0] += 0.15
            return gate_center
        self.get_logger().debug(
            f'Gate detection: candidates={len(candidate_walls)}, '
            f'min_mse={min_mse:.6f}, threshold={self.gate_mse_threshold}'
        )
        return None

def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course")
    rclpy.init(args=ros_args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate)
    try:
        rclpy.spin(obstacle)
    except SystemExit:
        print("Shutting down")
    obstacle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
