"""Gate/final transitions without publishing commands to a robot."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import rclpy
from rclpy.parameter import Parameter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp.obstacle_course import ObstacleCourseNode


class FinalStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[], domain_id=168)

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = ObstacleCourseNode(is_simulation=True, parameter_overrides=[
            Parameter('single_point_goal', value=False)])
        self.node.timer.cancel()
        self.node._rviz_timer.cancel()
        self.addCleanup(self.node.destroy_node)
        self.node.pose = np.zeros(3)
        self.node.actual_velocity = (0., 0.)
        self.node._ob = np.array([[2., 0.]])
        self.node.move_2D = mock.Mock()
        patcher = mock.patch(
            'rb2301_fp.obstacle_course.dynamic_window_approach.dwa_control',
            return_value=([0., 0.], np.zeros((2, 5))))
        self.dwa = patcher.start()
        self.addCleanup(patcher.stop)

    def test_detected_gate_uses_gate_parameters_in_same_callback(self):
        self.node.timer_callback_counter = self.node.deadend_check_gap
        self.node.segment_chipper = mock.Mock()
        self.node.wall_generator = mock.Mock()
        self.node.deadend_detector = mock.Mock(return_value=None)
        self.node.gate_detector = mock.Mock(return_value=np.array([2., 0.]))
        self.node.timer_callback()
        self.assertEqual(self.node.stage, 'gate')
        self.assertIsNone(self.dwa.call_args.kwargs['goal_heading'])
        np.testing.assert_array_equal(self.dwa.call_args.kwargs['goal'], [2., 0.])
        self.assertEqual(self.node.dwa_config.to_goal_cost_gain, 1.)
        self.assertEqual(self.node.dwa_config.speed_cost_gain, 1.)
        self.assertFalse(self.node._final_stage_started)

    def test_reached_gate_uses_final_heading_and_parameters_immediately(self):
        for heading in ('+x', '+y'):
            with self.subTest(heading=heading):
                self.node.stage = 'gate'
                self.node._final_stage_started = False
                self.node._sub_goal_2 = [heading]
                self.node._goal.goal_node = np.array([.1, 0.])
                self.node.gate_stage()
                self.node.timer_callback()
                self.assertEqual(self.node.stage, 'final')
                self.assertIsNone(self.dwa.call_args.kwargs['goal'])
                self.assertEqual(self.dwa.call_args.kwargs['goal_heading'], heading)
                self.assertEqual(self.node.dwa_config.to_goal_heading_cost_gain, .5)
                self.assertEqual(self.node.dwa_config.speed_cost_gain, .5)
                self.assertTrue(self.node._final_stage_started)
                threshold = self.node.close_threshold
                self.node.timer_callback()
                self.assertEqual(self.node.close_threshold, threshold)
                self.assertEqual(self.dwa.call_args.kwargs['goal_heading'], heading)


if __name__ == '__main__':
    unittest.main()
