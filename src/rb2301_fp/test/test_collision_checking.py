"""Collision regression checks and a ROS-free benchmark.

Run tests with pytest, or run this file with --benchmark on the target robot.
"""

import argparse
import importlib.util
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "rb2301_fp" / "dynamic_window_approach.py"
SPEC = importlib.util.spec_from_file_location("dwa_collision_checks", str(MODULE_PATH))
dwa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dwa)
RobotType = dwa.RobotType


# Frozen pre-optimization implementation; independent of the new broad phase.
def reference_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    if np.size(ob) == 0:
        return 0.0
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        # Rotate each relative point using the heading at the same time step.
        local_ob = np.einsum('oti,tij->otj', local_ob, rot)
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK



# Frozen per-candidate DWA loop from before batching; used as the reference
# for the batched calc_control_and_trajectory and in --benchmark.
def reference_control_and_trajectory(x, dw, config, goal, ob, goal_heading):
    if goal is not None and goal_heading is not None:
        raise RuntimeError('cannot implement goal and goal_heading at the same time!')
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    _vy_candidates = dwa.sample_trajectory_from_dw(dw, config)

    for v in _vy_candidates[0]:
        for y in _vy_candidates[1]:
            trajectory = dwa.predict_trajectory(x_init, v, y, config)
            to_goal_cost = 0
            if goal_heading is not None:
                to_goal_cost = config.to_goal_heading_cost_gain * dwa.calc_to_goal_heading_cost(trajectory, goal_heading)
                to_goal_cost += config.to_goal_heading_dist_cost_gain * dwa.calc_to_goal_heading_dist_cost(trajectory, goal_heading)
            if goal is not None:
                to_goal_cost = config.to_goal_node_cost_gain * dwa.calc_to_goal_cost(trajectory, goal)
                to_goal_cost += config.to_goal_node_dist_cost_gain * dwa.calc_to_goal_node_dist_cost(trajectory, goal)

            diff = config.max_speed - max(0, abs(trajectory[-1, 3]))
            speed_cost = config.speed_cost_gain * diff

            ob_cost = config.obstacle_cost_gain * dwa.calc_obstacle_cost(trajectory, ob, config)
            ob_cost += config.obstacle_turn_cost_gain * dwa.calc_obstacle_turn_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost
            if np.isfinite(final_cost) and min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory

    if abs(best_u[0]) < config.robot_stuck_flag_cons \
            and abs(x[3]) < config.robot_stuck_flag_cons \
            and best_u[1] < config.robot_stuck_flag_cons:
        best_u[1] = (-1) ** np.random.randint(1, 2) * config.max_delta_yaw_rate * 0.5
    return best_u, best_trajectory


def reference_dwa_control(x, ob, goal=None, goal_heading=None, config=None):
    dw = dwa.calc_dynamic_window(x, config)
    return reference_control_and_trajectory(x, dw, config, goal, ob, goal_heading)


def room_obstacles(count, half_width=1.0):
    """Raycast a rectangular room around the origin."""
    angles = np.linspace(-np.pi, np.pi, count, endpoint=False)
    cosine, sine = np.cos(angles), np.sin(angles)
    ranges = np.minimum(
        1.5 / np.maximum(np.abs(cosine), 1e-12),
        half_width / np.maximum(np.abs(sine), 1e-12),
    )
    return np.column_stack((ranges * cosine, ranges * sine))


class CollisionCheckingTests(unittest.TestCase):
    def setUp(self):
        self.config = dwa.Config()

    def assert_reference_cost(self, trajectory, obstacles):
        expected = reference_obstacle_cost(trajectory, obstacles, self.config)
        actual = dwa.calc_obstacle_cost(trajectory, obstacles, self.config)
        np.testing.assert_equal(actual, expected)

    def test_empty_obstacles(self):
        trajectory = np.zeros((1, 5))
        for obstacles in (np.empty((0, 2)), np.array([])):
            self.assertEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), 0.0)

    def test_rectangle_includes_edges_and_corners(self):
        trajectory = np.zeros((1, 5))
        half_length = self.config.robot_length / 2
        half_width = self.config.robot_width / 2
        for x in (-half_length, 0.0, half_length):
            for y in (-half_width, 0.0, half_width):
                obstacles = np.array([[x, y]])
                self.assertEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), np.inf)
        for x, y in (
                (np.nextafter(half_length, np.inf), 0.0),
                (0.0, np.nextafter(half_width, np.inf))):
            self.assertTrue(np.isfinite(dwa.calc_obstacle_cost(
                trajectory, np.array([[x, y]]), self.config)))

    def test_circle_broad_phase_does_not_enlarge_rectangle(self):
        trajectory = np.zeros((1, 5))
        obstacles = np.array([[0.12, 0.0]])
        self.assertAlmostEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), 1 / 0.12)

    def test_clearance_still_uses_distant_points(self):
        trajectory = np.zeros((2, 5))
        trajectory[1, 0] = 0.2
        obstacles = np.array([[4.0, 0.0], [0.0, 3.0]])
        self.assertEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), 1 / 3.0)
        self.assert_reference_cost(trajectory, obstacles)

    def test_collision_at_middle_pose_with_different_heading(self):
        self.config.robot_length = 0.3
        self.config.robot_width = 0.1
        trajectory = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, np.pi / 2, 0.0, 0.0],
            [2.0, 0.0, np.pi, 0.0, 0.0],
        ])
        obstacles = np.array([[1.0, 0.11], [10.0, 10.0]])
        self.assertEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), np.inf)
        self.assert_reference_cost(trajectory, obstacles)

    def test_rotated_boundary_points_match_reference(self):
        half_length = self.config.robot_length / 2
        half_width = self.config.robot_width / 2
        local_points = np.array([
            [half_length, half_width], [-half_length, half_width],
            [half_length, -half_width], [-half_length, -half_width],
            [half_length, 0.0], [-half_length, 0.0],
            [0.0, half_width], [0.0, -half_width],
            [half_length * (1 + 1e-5), 0.0],
            [0.0, half_width * (1 + 1e-5)],
        ])
        for dtype in (np.float32, np.float64):
            for yaw in (0.0, np.pi / 6, np.pi / 2, np.pi, -2.7):
                trajectory = np.array([[1.25, -0.75, yaw, 0.0, 0.0]], dtype=dtype)
                rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                                     [np.sin(yaw), np.cos(yaw)]], dtype=dtype)
                points = local_points.astype(dtype) @ rotation.T + trajectory[0, :2]
                for point in points:
                    with self.subTest(dtype=dtype, yaw=yaw, point=point):
                        self.assert_reference_cost(trajectory, point[None, :])

    def test_random_costs_match_reference(self):
        rng = np.random.RandomState(2301)
        for robot_type in (RobotType.rectangle, RobotType.circle):
            self.config.robot_type = robot_type
            for count in (1, 17, 360, 720, 1440):
                for _ in range(16):
                    trajectory = rng.normal(size=(26, 5))
                    obstacles = rng.normal(size=(count, 2)) * 5
                    self.assert_reference_cost(trajectory, obstacles)

    def test_circle_radius_boundary(self):
        self.config.robot_type = RobotType.circle
        trajectory = np.zeros((1, 5))
        radius = self.config.robot_radius
        self.assertEqual(dwa.calc_obstacle_cost(
            trajectory, np.array([[radius, 0.0]]), self.config), np.inf)
        self.assertTrue(np.isfinite(dwa.calc_obstacle_cost(
            trajectory, np.array([[np.nextafter(radius, np.inf), 0.0]]), self.config)))

    def test_inputs_are_unchanged(self):
        trajectory = dwa.predict_trajectory(np.zeros(5), 0.12, 0.1, self.config)
        obstacles = room_obstacles(360)
        original_trajectory, original_obstacles = trajectory.copy(), obstacles.copy()
        dwa.calc_obstacle_cost(trajectory, obstacles, self.config)
        np.testing.assert_array_equal(trajectory, original_trajectory)
        np.testing.assert_array_equal(obstacles, original_obstacles)

    def test_dwa_commands_and_trajectories_match_reference(self):
        state = np.array([0.0, 0.0, 0.0, 0.12, 0.05])
        for robot_type in (RobotType.rectangle, RobotType.circle):
            self.config.robot_type = robot_type
            for half_width in (1.0, 0.18):
                obstacles = room_obstacles(720, half_width)
                for heading in ("+x", "-x", "+y", "-y"):
                    with mock.patch.object(dwa, "calc_obstacle_cost", reference_obstacle_cost):
                        expected = reference_dwa_control(state.copy(), obstacles, goal_heading=heading,
                                                         config=self.config)
                    actual = dwa.dwa_control(None, state.copy(), obstacles, goal_heading=heading,
                                             config=self.config)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-9)


class BatchedDwaEquivalenceTests(unittest.TestCase):
    """Batched calc_control_and_trajectory must pick what the old loop picked."""

    STATES = (
        np.array([0.0, 0.0, 0.0, 0.12, 0.05]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.1, -0.05, 0.6, 0.25, -0.3]),
        np.array([-0.2, 0.08, -2.8, 0.05, 0.4]),
        np.array([0.3, 0.0, np.pi / 2, 0.2, 0.0]),
    )
    GOALS = (np.array([5.0, 0.0]), np.array([-3.0, 2.0]), np.array([0.5, -4.0]),
             np.array([0.05, 0.02]))
    HEADINGS = ("+x", "-x", "+y", "-y")

    def assert_same(self, state, obstacles, config, **goal):
        expected = reference_dwa_control(state.copy(), obstacles, config=config, **goal)
        actual = dwa.dwa_control(None, state.copy(), obstacles, config=config, **goal)
        self.assertEqual(len(actual[0]), 2)
        self.assertEqual(float(actual[0][0]), float(expected[0][0]))
        self.assertEqual(float(actual[0][1]), float(expected[0][1]))
        self.assertEqual(actual[1].shape, expected[1].shape)
        np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-9)

    def test_matches_reference_over_scenes(self):
        for is_simulation in (True, False):
            for robot_type in (RobotType.rectangle, RobotType.circle):
                config = dwa.Config(is_simulation=is_simulation)
                config.robot_type = robot_type
                for scene, half_width in (("room", 1.0), ("corridor", 0.18)):
                    for count in (360, 720):
                        obstacles = room_obstacles(count, half_width)
                        for index, state in enumerate(self.STATES):
                            for heading in self.HEADINGS:
                                with self.subTest(sim=is_simulation, robot=robot_type.name,
                                                  scene=scene, points=count, state=index,
                                                  heading=heading):
                                    self.assert_same(state, obstacles, config, goal_heading=heading)
                            for goal in self.GOALS:
                                with self.subTest(sim=is_simulation, robot=robot_type.name,
                                                  scene=scene, points=count, state=index,
                                                  goal=goal.tolist()):
                                    self.assert_same(state, obstacles, config, goal=goal)

    def test_matches_reference_with_random_obstacles(self):
        rng = np.random.RandomState(2301)
        for is_simulation in (True, False):
            config = dwa.Config(is_simulation=is_simulation)
            for trial in range(40):
                obstacles = rng.uniform(-1.5, 1.5, size=(rng.randint(1, 400), 2))
                state = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                                  rng.uniform(-np.pi, np.pi), rng.uniform(0, 0.25),
                                  rng.uniform(-0.5, 0.5)])
                with self.subTest(sim=is_simulation, trial=trial):
                    if trial % 2:
                        self.assert_same(state, obstacles, config, goal=rng.uniform(-5, 5, 2))
                    else:
                        self.assert_same(state, obstacles, config,
                                         goal_heading=self.HEADINGS[trial % 4])

    def test_no_obstacles_and_no_goal(self):
        config = dwa.Config()
        state = np.array([0.0, 0.0, 0.3, 0.1, 0.0])
        for obstacles in (np.empty((0, 2)), np.array([])):
            self.assert_same(state, obstacles, config, goal=np.array([2.0, 1.0]))
            self.assert_same(state, obstacles, config)

    def test_all_candidates_collide(self):
        config = dwa.Config()
        state = np.array([0.0, 0.0, 0.0, 0.1, 0.0])
        obstacles = np.array([[0.0, 0.0], [0.05, 0.0]])
        self.assert_same(state, obstacles, config, goal_heading="+x")

    def test_batch_obstacle_cost_matches_per_candidate(self):
        rng = np.random.RandomState(7)
        for is_simulation in (True, False):
            for robot_type in (RobotType.rectangle, RobotType.circle):
                config = dwa.Config(is_simulation=is_simulation)
                config.robot_type = robot_type
                for trial in range(30):
                    offset = rng.uniform(-6, 6, 2)
                    state = np.array([offset[0], offset[1], rng.uniform(-np.pi, np.pi),
                                      rng.uniform(0, 0.25), rng.uniform(-0.5, 0.5)])
                    v = rng.uniform(-0.1, 0.3, 40)
                    y = rng.uniform(-1.0, 1.0, 40)
                    trajectories = dwa.predict_trajectories(state, v, y, config)
                    # Points scattered around, plus points placed just inside
                    # and just outside the footprint of random poses.
                    obstacles = offset + rng.uniform(-1.2, 1.2, size=(rng.randint(1, 500), 2))
                    poses = trajectories[rng.randint(40, size=20), rng.randint(trajectories.shape[1], size=20)]
                    for scale in (0.999, 1.001):
                        local = np.column_stack((
                            rng.choice([-1, 1], 20) * config.robot_length / 2 * scale,
                            rng.uniform(-1, 1, 20) * config.robot_width / 2))
                        cosine, sine = np.cos(poses[:, 2]), np.sin(poses[:, 2])
                        world = np.column_stack((
                            poses[:, 0] + local[:, 0] * cosine - local[:, 1] * sine,
                            poses[:, 1] + local[:, 0] * sine + local[:, 1] * cosine))
                        obstacles = np.vstack((obstacles, world[: rng.randint(0, 3)]))
                    if trial % 3 == 0:
                        obstacles = obstacles.astype(np.float32)
                    vectors = np.asarray(obstacles) - trajectories[0, 0, :2]
                    nearest = float(np.min(np.linalg.norm(vectors, axis=1)))
                    actual = dwa._batch_obstacle_cost(trajectories, obstacles, config, nearest)
                    expected = [dwa.calc_obstacle_cost(t, obstacles, config) for t in trajectories]
                    with self.subTest(sim=is_simulation, robot=robot_type.name, trial=trial):
                        np.testing.assert_array_equal(actual, expected)

    def test_rollout_matches_predict_trajectory(self):
        config = dwa.Config()
        state = np.array([0.3, -0.2, 1.1, 0.1, 0.2])
        v = np.array([0.0, 0.1, 0.25, -0.05])
        y = np.array([0.0, -0.4, 0.7, 1.0])
        batch = dwa.predict_trajectories(state, v, y, config)
        for index in range(v.size):
            np.testing.assert_allclose(
                batch[index], dwa.predict_trajectory(state, v[index], y[index], config),
                rtol=0, atol=1e-9)


def benchmark():
    parser = argparse.ArgumentParser(description="Compare the frozen per-candidate DWA with the batched one, without ROS.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--points", nargs="+", type=int, default=[360, 720, 1440])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sim", action="store_true", help="use Config(is_simulation=True) instead of the real-robot config")
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 1 for count in args.points):
        parser.error("points and repeats must be positive")

    config = dwa.Config(is_simulation=args.sim)
    config.robot_type = RobotType.rectangle
    state = np.array([0.0, 0.0, 0.0, 0.12, 0.05])
    print("Python {}; NumPy {}; synthetic scans, no ROS or GUI; {} config".format(
        sys.version.split()[0], np.__version__, "simulation" if args.sim else "real-robot"))
    print("scene,points,dwa_old_ms,dwa_new_ms,speedup")
    for scene, half_width in (("room", 1.0), ("corridor", 0.18)):
        for count in args.points:
            obstacles = room_obstacles(count, half_width)
            runners = (
                lambda: reference_dwa_control(state.copy(), obstacles, goal_heading="+x", config=config),
                lambda: dwa.dwa_control(None, state.copy(), obstacles, goal_heading="+x", config=config),
            )
            timings, outputs = [], []
            for run in runners:
                run()  # warm-up
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    output = run()
                    samples.append((time.perf_counter() - start) * 1000)
                outputs.append(output)
                timings.append(float(np.median(samples)))
            np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
            np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=0, atol=1e-9)
            print("{},{},{:.3f},{:.3f},{:.2f}x".format(
                scene, count, timings[0], timings[1], timings[0] / timings[1]))


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
