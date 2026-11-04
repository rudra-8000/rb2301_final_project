"""Single-goal navigation without sending motion commands to a robot."""
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp.obstacle_course import ObstacleCourseNode


class SinglePointGoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[], domain_id=169)

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def make_node(self, is_simulation=True, overrides=None):
        node = ObstacleCourseNode(is_simulation=is_simulation, parameter_overrides=overrides)
        node.timer.cancel()
        node._rviz_timer.cancel()
        self.addCleanup(node.destroy_node)
        node.move_2D = mock.Mock()
        node._ob = np.array([[100., 100.]])
        node.actual_velocity = (0., 0.)
        return node

    def mock_dwa(self):
        patcher = mock.patch(
            'rb2301_fp.obstacle_course.dynamic_window_approach.dwa_control',
            return_value=([.1, .05], np.zeros((2, 5))))
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_simulation_goal_is_frozen_at_first_pose_using_course_direction(self):
        node = self.make_node()
        first = Odometry()
        first.header.stamp.sec = 1
        first.pose.pose.position.x, first.pose.pose.position.y = 1.2, -.8
        # Course forward is +x, even when the initial robot yaw is +y.
        first.pose.pose.orientation.z = math.sin(math.pi / 4)
        first.pose.pose.orientation.w = math.cos(math.pi / 4)
        node.odometer_callback(first)
        expected = np.array([1.2, -.8]) + 5 * np.array([
            math.cos(math.radians(35)), math.sin(math.radians(35))])
        np.testing.assert_allclose(node._goal.goal_node, expected)
        first.header.stamp.sec = 2
        first.pose.pose.position.x = 2.5
        node.odometer_callback(first)
        np.testing.assert_allclose(node._goal.goal_node, expected)
        self.assertIsNone(node._goal.goal_heading)
        node.move_2D.assert_not_called()

    def test_real_goal_uses_initial_minus_y_in_maze_frame(self):
        node = self.make_node(is_simulation=False)
        node.frame = dict(origin_x=10., origin_y=20., rotation_deg=0., heading_offset_deg=0.)
        first = PoseStamped()
        first.header.stamp.sec = 1
        first.pose.position.x, first.pose.position.y = 11., 22.
        first.pose.orientation.w = 1.
        node.optitrack_callback(first)
        expected = np.array([1., 2.]) + 5 * np.array([
            math.sin(math.radians(35)), -math.cos(math.radians(35))])
        np.testing.assert_allclose(node._goal.goal_node, expected)
        first.header.stamp.sec = 2
        first.pose.position.y = 21.
        node.optitrack_callback(first)
        np.testing.assert_allclose(node._goal.goal_node, expected)

    def test_startup_parameters_control_goal_distance_and_left_angle(self):
        node = self.make_node(overrides=[
            Parameter('point_goal_distance', value=3.),
            Parameter('point_goal_left_angle_deg', value=40.)])
        node.pose = np.zeros(3)
        self.mock_dwa()
        node.timer_callback()
        np.testing.assert_allclose(node._goal.goal_node, 3 * np.array([
            math.cos(math.radians(40)), math.sin(math.radians(40))]))

    def test_all_course_stages_and_detectors_are_bypassed(self):
        node = self.make_node()
        node.pose = np.zeros(3)
        dwa = self.mock_dwa()
        methods = ('segment_chipper', 'wall_generator', 'deadend_detector',
                   'confirm_deadend', 'gate_detector', 'start_stage', 'gate_stage', 'final_stage')
        for name in methods:
            setattr(node, name, mock.Mock())
        initial_sub_goals = node._sub_goal.copy()
        for stage in ('start', 'gate', 'final'):
            node.stage = stage
            node.timer_callback_counter = node.deadend_check_gap
            node.timer_callback()
            self.assertIsNone(dwa.call_args.kwargs['goal_heading'])
            np.testing.assert_allclose(dwa.call_args.kwargs['goal'], node._goal.goal_node)
        self.assertEqual(node._sub_goal, initial_sub_goals)
        for name in methods:
            getattr(node, name).assert_not_called()
        # The old sub-goal list is not a prerequisite for point-goal control.
        node._sub_goal = []
        node.timer_callback()
        self.assertEqual(dwa.call_count, 4)
        node.move_2D.assert_called_with(.1, 0., .05)

    def test_arrival_stops_and_stays_stopped_without_scan_or_velocity(self):
        node = self.make_node()
        node.pose = np.zeros(3)
        dwa = self.mock_dwa()
        node.timer_callback()
        goal = node._goal.goal_node.copy()
        node.pose[:2] = goal + np.array([.1, 0.])
        node._ob = np.empty((0, 2))
        node.actual_velocity = None
        node.timer_callback()
        self.assertEqual(node.stage, 'done')
        self.assertIsNone(node._rviz_predicted_trajectory)
        np.testing.assert_array_equal(node._state[3:], [0., 0.])
        node.move_2D.assert_called_with()
        node.pose[:2] = [0., 0.]
        node.timer_callback()
        self.assertEqual(dwa.call_count, 1)
        np.testing.assert_array_equal(node._goal.goal_node, goal)
        node.move_2D.assert_called_with()

    def test_visualization_shows_fixed_goal_without_sub_goal_heading(self):
        node = self.make_node()
        node.pose = np.zeros(3)
        self.mock_dwa()
        node.timer_callback()
        node._sub_goal = []
        node._rviz_publisher = mock.Mock()
        node.publish_rviz_visualization()
        scene = {marker.ns: marker for marker in node._rviz_publisher.publish.call_args.args[0].markers}
        self.assertEqual(scene['sub_goal_heading'].action, Marker.DELETE)
        self.assertEqual(scene['predicted_trajectory'].action, Marker.ADD)
        goal_points = np.array([(p.x, p.y) for p in scene['reference_goal'].points])
        np.testing.assert_allclose(goal_points.mean(axis=0), node._goal.goal_node)

    def test_real_dwa_uses_point_costs_and_produces_finite_command(self):
        node = self.make_node()
        node.pose = np.zeros(3)
        with mock.patch(
                'rb2301_fp.obstacle_course.dynamic_window_approach.calc_to_goal_heading_cost',
                side_effect=AssertionError('Heading mode must not run')):
            node.timer_callback()
        self.assertTrue(node.move_2D.called)
        self.assertTrue(np.isfinite(node.move_2D.call_args.args).all())
        self.assertIsNone(node._goal.goal_heading)


if __name__ == '__main__':
    unittest.main()
