"""Controller goal transitions without publishing commands to a robot."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp.obstacle_course import ObstacleCourseNode


class GoalSwitchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[], domain_id=168)

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = ObstacleCourseNode(is_simulation=True)
        self.node.timer.cancel()
        self.node._rviz_timer.cancel()
        self.addCleanup(self.node.destroy_node)
        self.node.pose = np.zeros(3)
        self.node.actual_velocity = (0., 0.)
        self.node._ob = np.array([[2., 0.]])
        self.node.segment_chipper = mock.Mock()
        self.node.wall_generator = mock.Mock()
        self.node.deadend_detector = mock.Mock(return_value=(1., 0., -.75))
        self.node.gate_detector = mock.Mock(return_value=None)
        self.node.move_2D = mock.Mock()
        patcher = mock.patch(
            'rb2301_fp.obstacle_course.dynamic_window_approach.dwa_control',
            return_value=([0., 0.], np.zeros((2, 5))))
        self.dwa = patcher.start()
        self.addCleanup(patcher.stop)

    def check_walls(self):
        self.node.timer_callback_counter = self.node.deadend_check_gap
        self.node.timer_callback()

    def test_initial_heading_is_used_before_first_wall_check(self):
        self.node.timer_callback()
        self.assertEqual(self.dwa.call_args.kwargs['goal_heading'], '+x')

    def test_deadend_switch_preserves_heading_and_cooldown(self):
        with mock.patch('rb2301_fp.obstacle_course.time.monotonic', return_value=10.):
            self.check_walls()
            self.assertEqual(self.node._sub_goal, ['+x', '+x', '-y'])
            self.assertEqual(self.dwa.call_args.kwargs['goal_heading'], '-y')
            self.check_walls()
            self.assertEqual(len(self.node._sub_goal), 3)
        self.node.gate_detector.assert_not_called()
        self.assertEqual(self.node.close_threshold, 1.)

    def test_failed_gate_detection_keeps_heading_and_threshold(self):
        self.node._sub_goal = ['+x']
        for _ in range(3):
            self.check_walls()
        self.assertEqual(self.node.close_threshold, .5)
        self.assertEqual(self.node._sub_goal, ['+x'])
        self.assertEqual(self.dwa.call_args.kwargs['goal_heading'], '+x')
        self.assertEqual(self.node.dwa_config.to_goal_cost_gain, 1.)

    def test_gate_target_takes_priority_over_deadend(self):
        self.node._sub_goal = ['+x']
        self.node.gate_detector.return_value = np.array([2., 0.])
        self.node.deadend_detector.return_value = (1., 0., -.1)
        self.check_walls()
        self.assertEqual(self.node._sub_goal, ['+x'])
        np.testing.assert_array_equal(self.dwa.call_args.kwargs['goal'], [2., 0.])
        self.assertIsNone(self.dwa.call_args.kwargs['goal_heading'])

    def test_last_deadend_stops_without_indexing_empty_list(self):
        self.node._sub_goal = ['+x']
        self.node.deadend_detector.return_value = (1., 0., -.1)
        self.check_walls()
        self.assertEqual(self.node._sub_goal, [])
        self.node.move_2D.assert_called_once_with()
        self.dwa.assert_not_called()

    def test_reached_gate_stops_and_does_not_reacquire_target(self):
        self.node._sub_goal = ['+x']
        self.node._goal.goal_node = np.array([.1, 0.])
        self.node.timer_callback()
        self.node.timer_callback()
        self.assertEqual(self.node._sub_goal, [])
        self.dwa.assert_not_called()
        self.node.gate_detector.assert_not_called()
        self.assertEqual(self.node.move_2D.call_count, 2)


if __name__ == '__main__':
    unittest.main()
