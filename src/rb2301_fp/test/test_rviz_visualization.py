"""Headless ROS message/transport checks; uses an isolated ROS domain."""
import math
import os
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp import dynamic_window_approach as dwa
from rb2301_fp.obstacle_course import LidarNode, ObstacleCourseNode


class RvizVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Override inherited robot domain; never connect the test to a robot.
        rclpy.init(args=[], domain_id=int(os.environ.get('FP_TEST_DOMAIN_ID', '167')))

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = ObstacleCourseNode(is_simulation=True, calibrate=True, parameter_overrides=[
            Parameter('single_point_goal', value=False)])
        self.node.timer.cancel()
        self.node._rviz_timer.cancel()
        self.reader = Node('fp_visualization_test_reader')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.reader)
        self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def tearDown(self):
        self.executor.shutdown()
        self.reader.destroy_node()
        self.node.destroy_node()

    def receive_snapshot(self):
        """Join after publication: verify a real DDS late-joining subscriber."""
        self.node.publish_rviz_visualization()
        received = []
        subscription = self.reader.create_subscription(
            MarkerArray, '/obstaclecourse/visualization', received.append, self.qos)
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
        self.reader.destroy_subscription(subscription)
        self.assertTrue(received, 'No transient-local snapshot received over DDS')
        return {marker.ns: marker for marker in received[-1].markers}

    def test_late_join_without_scan_or_prediction(self):
        self.node.pose = np.array([2.0, -1.0, 90.0])
        scene = self.receive_snapshot()
        self.assertEqual(scene['obstacles'].action, Marker.DELETE)
        self.assertEqual(scene['predicted_trajectory'].action, Marker.DELETE)
        for marker in scene.values():
            self.assertEqual(marker.header.frame_id, 'fp_visualization')
            self.assertEqual(marker.pose.orientation.w, 1.0)
            self.assertEqual(marker.lifetime.sec, 1)
        arrow = scene['robot_heading']
        np.testing.assert_allclose(
            [(p.x, p.y) for p in arrow.points], [[2.0, -1.0], [2.0, -0.5]])
        self.assertNotIn('matplotlib.backends.backend_qtagg', sys.modules)

    def test_timer_publishes_while_controller_has_no_scan(self):
        self.node.pose = np.array([1.0, 2.0, 0.0])
        received = []
        self.reader.create_subscription(
            MarkerArray, '/obstaclecourse/visualization', received.append, self.qos)
        self.node._rviz_timer.reset()
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
        self.assertTrue(received)
        self.assertIsNone(self.node.actual_velocity)
        self.assertEqual(self.node._ob.size, 0)

    def test_empty_layers_and_completed_subgoals_delete_old_markers(self):
        self.node.calibrate = False
        self.node.pose = np.array([0.0, 0.0, 0.0])
        self.node._ob = np.array([[1.0, 2.0], [1.5, 2.0]])
        self.node._segmentation = [self.node._ob.copy()]
        self.node._rviz_predicted_trajectory = np.array([
            [0.0, 0.0, 0.0, 0.1, 0.0], [0.1, 0.0, 0.0, 0.1, 0.0]])
        scene = self.receive_snapshot()
        self.assertEqual(scene['predicted_trajectory'].action, Marker.ADD)
        self.assertEqual(scene['segmentation'].action, Marker.ADD)
        self.assertEqual(len(scene['segmentation'].colors), 2)
        self.node._ob = np.empty((0, 2))
        self.node._segmentation = []
        self.node._sub_goal = []
        scene = self.receive_snapshot()
        for name in ('obstacles', 'segmentation', 'predicted_trajectory', 'sub_goal_heading'):
            self.assertEqual(scene[name].action, Marker.DELETE)

    def test_footprint_rotation_matches_pose_and_dwa_dimensions(self):
        self.node.pose = np.array([2.0, 3.0, 90.0])
        self.node.dwa_config.robot_length = 0.46
        with mock.patch.object(self.node.dwa_config, '_robot_type', dwa.RobotType.rectangle):
            footprint = self.receive_snapshot()['robot_footprint']
        point = footprint.points[0]
        self.assertAlmostEqual(point.x, 2.0 - self.node.dwa_config.robot_width / 2)
        self.assertAlmostEqual(point.y, 3.0 - self.node.dwa_config.robot_length / 2)
        self.assertEqual(footprint.points[0], footprint.points[-1])

    def test_circle_and_infeasible_prediction(self):
        self.node.calibrate = False
        self.node.pose = np.array([-2.0, 1.0, 45.0])
        self.node._ob = np.array([[5.0, 5.0]])
        self.node._rviz_predicted_trajectory = np.zeros((1, 5))
        with mock.patch.object(self.node.dwa_config, '_robot_type', dwa.RobotType.circle):
            scene = self.receive_snapshot()
        for point in scene['robot_footprint'].points:
            self.assertAlmostEqual(math.hypot(point.x + 2.0, point.y - 1.0),
                                   self.node.dwa_config.robot_radius)
        self.assertEqual(scene['predicted_trajectory'].action, Marker.DELETE)

    def test_rejected_wall_does_not_block_controller_and_visualization(self):
        self.node.calibrate = False
        self.node.pose = np.zeros(3)
        self.node.actual_velocity = (0.0, 0.0)
        # A wall parallel to +x must be rejected without re-appending it.
        self.node._ob = np.column_stack((np.linspace(0.5, 1.5, 51), np.full(51, 0.6)))
        self.node.timer_callback_counter = self.node.deadend_check_gap
        original_check = self.node.check_segment
        calls = []

        def bounded_check(segment, farthest_dot):
            calls.append(segment)
            if len(calls) > len(self.node._ob):
                self.fail('Wall detection repeatedly visits appended segments and blocks RViz')
            return original_check(segment, farthest_dot)

        with mock.patch.object(self.node, 'check_segment', side_effect=bounded_check), \
                mock.patch.object(dwa, 'dwa_control', return_value=([0.0, 0.0], np.zeros((2, 5)))), \
                mock.patch.object(self.node, 'move_2D'):
            self.node.timer_callback()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.node._segmentation), 1)
        scene = self.receive_snapshot()
        for name in ('obstacles', 'segmentation', 'robot_position', 'robot_heading',
                     'robot_footprint', 'sub_goal_heading', 'predicted_trajectory'):
            self.assertEqual(scene[name].action, Marker.ADD)

    def test_best_effort_sensors_feed_startup_pose_and_obstacles(self):
        lidar = LidarNode()
        self.executor.add_node(lidar)
        scan_writer = self.reader.create_publisher(LaserScan, 'scan', qos_profile_sensor_data)
        odom_writer = self.reader.create_publisher(Odometry, 'odom', qos_profile_sensor_data)
        scan = LaserScan(angle_min=-1.0, angle_increment=0.01,
                         range_min=0.1, range_max=10.0, ranges=[1.0] * 360)
        odom = Odometry()
        odom.pose.pose.orientation.w = 1.0
        odom.pose.pose.position.x = 2.0
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                odom_writer.publish(odom)
                scan_writer.publish(scan)
                self.executor.spin_once(timeout_sec=0.02)
                if lidar.is_simulation is True and self.node._ob.size:
                    break
            self.assertIs(lidar.is_simulation, True, 'Startup did not receive best-effort scan')
            self.assertIsNotNone(self.node.pose, 'No best-effort odometry received')
            self.assertEqual(self.node._ob.shape, (360, 2))
            scene = self.receive_snapshot()
            self.assertEqual(len(scene['obstacles'].points), 360)
            self.assertEqual(scene['robot_position'].points[0].x, 2.0)
        finally:
            self.executor.remove_node(lidar)
            lidar.destroy_node()
            self.reader.destroy_publisher(scan_writer)
            self.reader.destroy_publisher(odom_writer)


if __name__ == '__main__':
    unittest.main()
