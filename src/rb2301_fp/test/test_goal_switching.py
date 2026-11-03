"""Controller goal transitions without publishing commands to a robot."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp.obstacle_course import ObstacleCourseNode, Wall


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
        self.node.deadend_detector.return_value = (1., 0., -.4)
        with mock.patch('rb2301_fp.obstacle_course.time.monotonic', return_value=10.):
            for _ in range(2):
                self.check_walls()
                self.assertEqual(self.node._sub_goal, ['+x', '-y', '+x'])
            self.check_walls()
            self.assertEqual(self.node._sub_goal, ['+x', '-y'])
            self.assertEqual(self.dwa.call_args.kwargs['goal_heading'], '-y')
            self.node.pose[2] = -90.
            self.node.deadend_detector.return_value = (0., 1., .4)
            for _ in range(3):
                self.check_walls()
            self.assertEqual(len(self.node._sub_goal), 2)
        self.assertEqual(self.node.close_threshold, 1.)

    def test_failed_gate_detection_keeps_heading_and_threshold(self):
        self.node._sub_goal = ['+x']
        self.node.deadend_detector.return_value = None
        for _ in range(3):
            self.check_walls()
        self.assertEqual(self.node.close_threshold, 1.)
        self.assertEqual(self.node._sub_goal, ['+x'])
        self.assertEqual(self.dwa.call_args.kwargs['goal_heading'], '+x')
        self.assertIsNone(self.node.dwa_config.to_goal_cost_gain)

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
        for _ in range(self.node.deadend_confirm_checks - 1):
            self.check_walls()
        self.node.move_2D.reset_mock()
        self.dwa.reset_mock()
        self.check_walls()
        self.assertEqual(self.node._sub_goal, [])
        self.node.move_2D.assert_not_called()
        self.dwa.assert_not_called()
        self.node.timer_callback()
        self.node.move_2D.assert_called_once_with()

    def test_reached_gate_enters_final_and_does_not_reacquire_target(self):
        self.node._sub_goal = ['+x']
        self.node.stage = 'gate'
        self.node._goal.goal_node = np.array([.1, 0.])
        self.node.timer_callback()
        self.node.timer_callback()
        self.assertEqual(self.node.stage, 'final')
        self.assertEqual(self.node._sub_goal, ['+x'])
        self.assertEqual(self.dwa.call_args.kwargs['goal_heading'], '+x')
        self.node.gate_detector.assert_not_called()
        self.assertEqual(self.node.move_2D.call_count, 2)

    def test_lost_or_changed_wall_resets_confirmation(self):
        self.assertFalse(self.node.confirm_deadend((1., 0., -.4)))
        self.assertFalse(self.node.confirm_deadend(None))
        self.assertFalse(self.node.confirm_deadend((1., 0., -.4)))
        self.assertFalse(self.node.confirm_deadend((1., 0., -.15)))
        self.assertEqual(self.node._deadend_hits, 1)
        self.assertFalse(self.node.confirm_deadend((-1., 0., .15)))
        self.assertTrue(self.node.confirm_deadend((1., 0., -.15)))

    def test_confirmation_does_not_require_heading_alignment(self):
        self.node._sub_goal = ['+x', '-y']
        self.node.pose[2] = 90.
        for _ in range(2):
            self.assertFalse(self.node.confirm_deadend((0., 1., .4)))
        self.assertTrue(self.node.confirm_deadend((0., 1., .4)))

    def test_finite_wall_coverage_in_all_directions(self):
        for heading, direction in (('+x', (1, 0)), ('-x', (-1, 0)),
                                   ('+y', (0, 1)), ('-y', (0, -1))):
            with self.subTest(heading=heading):
                self.node._sub_goal = [heading]
                d = np.array(direction)
                lateral = np.array([-d[1], d[0]])
                def wall(low, high, sign=1):
                    return Wall(tuple(sign * np.array([d[0], d[1], -.4])),
                                center=.4*d + (low+high)/2*lateral,
                                edge=(.4*d + low*lateral, .4*d + high*lateral))
                self.node._ob = np.array([.4*d])
                for sign in (1, -1):
                    self.assertIsNotNone(self.node.check_segment(wall(-.5, .5, sign), .4*d))
                self.assertIsNone(self.node.check_segment(wall(1.4, 1.6), .4*d))

    def test_scan_segmentation_accepts_end_wall_but_rejects_offset_wall(self):
        # Exercise the actual segmentation, SVD fit and detector together.
        for offset, expected in ((0., True), (1.5, False)):
            with self.subTest(offset=offset):
                self.node._ob = np.column_stack((np.full(51, .4),
                                                np.linspace(-.5, .5, 51) + offset))
                self.node._segmentation = []
                self.node.wall_list = []
                ObstacleCourseNode.segment_chipper(self.node)
                ObstacleCourseNode.wall_generator(self.node)
                result = ObstacleCourseNode.deadend_detector(self.node)
                self.assertEqual(result is not None, expected)


if __name__ == '__main__':
    unittest.main()
