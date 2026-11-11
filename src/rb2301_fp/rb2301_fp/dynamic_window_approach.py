"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

Modified by Wu Haolun, Xue Shengbo, with the help of LLM tools: Chatgpt & Claude

"""
from typing import List, Tuple
import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

show_animation = False
is_simulation = True


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self, is_simulation=True):
        if is_simulation:
            # robot parameter
            self.max_speed = 0.4  # [m/s]
            self.min_speed = -0.1  # [m/s]
            #self.max_yaw_rate = 90.0 * math.pi / 180.0  # [rad/s]

            self.max_yaw_rate = 1.0
            self.max_accel = 0.2  # [m/ss]
            self.max_delta_yaw_rate = 90.0 * math.pi / 180.0  # [rad/ss]
            
            self.v_resolution = 0.02  # [m/s]
            self.yaw_rate_resolution = 2.0 * math.pi / 180.0  # [rad/s]
            
            self.v_sample_size = 5
            self.yaw_sample_size = 20
            
            self.dt = 0.1  # [s] Time tick for motion prediction
            self.predict_time = 2.5  # [s]

            self.to_goal_node_cost_gain = 0.25      # in angle
            self.to_goal_node_dist_cost_gain = 2 # in distance

            self.to_goal_heading_cost_gain = 0.1        #in angle
            self.to_goal_heading_dist_cost_gain = 0.5   #in distance

            self.speed_cost_gain = 1.3 #was 1.2
            self.speed_back_gain = 0.7

            self.obstacle_cost_gain = 0.1
            self.obstacle_turn_cost_gain = 1.2

            self.robot_stuck_flag_cons = 0.012  #was 0.001 constant to prevent robot stucked
            self.robot_type = RobotType.rectangle

            # if robot_type == RobotType.circle
            # Also used to check if goal is reached in both types
            self.robot_radius = 1.0  # [m] for collision check

            # if robot_type == RobotType.rectangle
            self.robot_width = 0.24  # [m] for collision check
            self.robot_length = 0.24  # [m] for collision check

            # obstacles [x(m) y(m), ....]
        
        else:   #IRL param
                        # robot parameter
            self.max_speed = 0.25         # [m/s]
            self.min_speed = -0.0  # [m/s]
            self.max_yaw_rate = 0.6 #Same as the move2d output limit #90.0 * math.pi / 180.0  # [rad/s]
            self.max_accel = 0.25  # [m/ss]
            self.max_delta_yaw_rate = 90.0 * math.pi / 180.0  # [rad/ss]
            
            self.v_resolution = 0.02  # [m/s] NO LONGER USED
            self.yaw_rate_resolution = 1.0 * math.pi / 180.0  # [rad/s] NO LONGER USED

            self.v_sample_size = 4
            self.yaw_sample_size = 15

            self.dt = 0.1  # [s] Time tick for motion prediction
            self.predict_time = 2.5  # [s]

            self.to_goal_node_cost_gain = 0.125      # in angle
            self.to_goal_node_dist_cost_gain = 1.25  # in distance

            self.to_goal_heading_cost_gain = 0.1        #in angle
            self.to_goal_heading_dist_cost_gain = 0.5   #in distance

            self.speed_cost_gain = 2.5
            self.speed_back_gain = 0.2

            self.obstacle_cost_gain = 0.3
            self.obstacle_turn_cost_gain = 1.5

            self.robot_stuck_flag_cons = 0.001  #was 0.001 constant to prevent robot stucked
            self.robot_type = RobotType.rectangle

            # if robot_type == RobotType.circle
            # Also used to check if goal is reached in both types
            self.robot_radius = 1.0  # [m] for collision check

            # if robot_type == RobotType.rectangle
            self.robot_width = 0.20  # [m] for collision check
            self.robot_length = 0.22  # [m] for collision check
            # obstacles [x(m) y(m), ....]



    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()

def dwa_control(self,x,ob, goal=None,  goal_heading=None,config = config):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(self,x, dw, config, goal, ob,goal_heading)

    return u, trajectory

def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt   #heading
    x[0] += u[0] * math.cos(x[2]) * dt  #position x
    x[1] += u[0] * math.sin(x[2]) * dt  #position y
    x[3] = u[0] #velocity chosen
    x[4] = u[1] #turning chosen

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init, dtype=float)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory

def sample_trajectory_from_dw(dw,config):
    _vy_candidates= np.empty(2, dtype=object)

    def sample(lo,hi,sample_size):
        crosses_zero = lo < 0.0 < hi
        samples = np.linspace(lo, hi, sample_size)
        samples = np.append(samples,0.0) if crosses_zero else samples
        return samples

    _vy_candidates[0], _vy_candidates[1] = (
        sample(dw[0], dw[1], config.v_sample_size),
        sample(dw[2], dw[3], config.yaw_sample_size),
    )
    return _vy_candidates

def _prediction_step_count(config):
    """Number of motion() steps the `while time <= predict_time` loop makes."""
    steps = 0
    time = 0
    while time <= config.predict_time:
        steps += 1
        time += config.dt
    return steps


def predict_trajectories(x_init, v, y, config):
    """
    Roll out every (v[i], y[i]) candidate at once.

    Reproduces predict_trajectory() exactly: same step count, yaw updated
    before position, and running sums accumulated in the same order.
    Returns an array of shape (candidates, steps + 1, 5).
    """
    x0 = np.array(x_init, dtype=float)
    v = np.asarray(v, dtype=float)
    y = np.asarray(y, dtype=float)
    count = v.shape[0]
    steps = _prediction_step_count(config)

    trajectories = np.empty((count, steps + 1, x0.shape[0]))
    trajectories[:, 0, :] = x0
    if steps == 0:
        return trajectories

    yaw_increments = np.empty((count, steps + 1))
    yaw_increments[:, 0] = x0[2]
    yaw_increments[:, 1:] = (y * config.dt)[:, None]
    yaw = np.cumsum(yaw_increments, axis=1)  # sequential, like x[2] += ...

    increments = np.empty((count, steps + 1))
    increments[:, 0] = x0[0]
    increments[:, 1:] = v[:, None] * np.cos(yaw[:, 1:]) * config.dt
    trajectories[:, :, 0] = np.cumsum(increments, axis=1)
    increments[:, 0] = x0[1]
    increments[:, 1:] = v[:, None] * np.sin(yaw[:, 1:]) * config.dt
    trajectories[:, :, 1] = np.cumsum(increments, axis=1)

    trajectories[:, :, 2] = yaw
    trajectories[:, 1:, 3] = v[:, None]
    trajectories[:, 1:, 4] = y[:, None]
    return trajectories


# _batch_obstacle_cost splits each trajectory into segments of this many steps.
_SEGMENT_STEPS = 5
# Rough number of (candidate, coarse pose, point) elements per chunk, to keep
# temporaries in cache.
_CHUNK_ELEMENTS = 65536


def _obstacle_cost_setup(config):
    if config.robot_type == RobotType.rectangle:
        half_length = config.robot_length / 2
        half_width = config.robot_width / 2
        collision_radius = np.hypot(half_length, half_width)
        # Same broad-phase radius as calc_obstacle_cost (trajectories are float64).
        collision_radius += 8 * np.finfo(np.float64).eps * max(1.0, collision_radius)
        return collision_radius, half_length, half_width
    if config.robot_type == RobotType.circle:
        return config.robot_radius, None, None
    return 0.0, None, None


def _batch_obstacle_cost(trajectories, ob, config, nearest_distance):  
    """
    calc_obstacle_cost() for every trajectory at once, with identical results.

    calc_obstacle_cost returns inf if any (pose, point) pair collides, else
    1 / (minimum hypot distance over all pairs). Only pairs that can be that
    minimum or a collision affect the result, so the work is cut in exact
    steps. Steps 2 and 3 are filters only: they run in float32 on coordinates
    centred at the start pose and every bound carries a margin far above the
    float32 error, so they can only keep extra entries, never drop a needed
    one. The result is computed in step 4 with the reference arithmetic.

    1. Crop: with d_near the distance from the start pose to its nearest
       point and L the largest displacement any candidate reaches, a point
       farther than max(d_near + 2L, L + collision radius) from the start is
       farther from every pose than that nearest point, and cannot collide.
    2. Coarse pass: each trajectory is cut into segments of _SEGMENT_STEPS
       steps. Distances from the segment end poses give each candidate an
       upper bound U on its clearance, and, by the triangle inequality, a
       lower bound min(end distances) - h for every pose in a segment, where
       h is the farthest any pose of that segment lies from the nearer end.
       (candidate, segment, point) triples whose lower bound exceeds
       max(U, collision radius) cannot hold the minimum or a collision.
    3. Fine pass over every pose of the kept triples: keep entries within
       margin of the candidate's minimum distance, or within the collision
       radius.
    4. For the kept entries only, r = hypot(ob - pose) in float64 exactly as
       calc_obstacle_cost computes it; minimum clearance and the rectangle
       or circle collision test follow from those values.
    """
    count = trajectories.shape[0]
    if np.size(ob) == 0:
        return np.zeros(count)
    ob = np.asarray(ob)
    if not (np.isfinite(ob).all() and np.isfinite(trajectories[:, :, :3]).all()):
        # Keep the exact NaN/inf semantics of the per-candidate function.
        return np.array([calc_obstacle_cost(trajectory, ob, config)
                         for trajectory in trajectories])

    collision_radius, half_length, half_width = _obstacle_cost_setup(config)
    traj_x = trajectories[:, :, 0]
    traj_y = trajectories[:, :, 1]
    yaw = trajectories[:, :, 2]
    steps = traj_x.shape[1]
    start_x, start_y = trajectories[0, 0, 0], trajectories[0, 0, 1]
    rel_x = traj_x - start_x
    rel_y = traj_y - start_y
    reach = float(np.max(np.hypot(rel_x, rel_y)))

    # 1. Crop.
    keep_radius = max(nearest_distance + 2 * reach, reach + collision_radius)
    keep_radius += 1e-9 * max(1.0, keep_radius)
    ob_rel_x = ob[:, 0] - start_x
    ob_rel_y = ob[:, 1] - start_y
    kept = np.flatnonzero(np.hypot(ob_rel_x, ob_rel_y) <= keep_radius)
    # Filter margin in metres; float32 error on these coordinates is ~1e-7
    # of their magnitude.
    margin = 1e-5 * max(1.0, keep_radius + reach)
    ox = ob_rel_x[kept].astype(np.float32)
    oy = ob_rel_y[kept].astype(np.float32)
    tx = rel_x.astype(np.float32)
    ty = rel_y.astype(np.float32)
    points = ox.shape[0]

    # Segment layout: ends[j] .. ends[j + 1] inclusive, padded by repeating
    # the last step (a repeated pose changes neither a minimum nor a test).
    ends = np.unique(np.append(np.arange(0, steps, _SEGMENT_STEPS), steps - 1))
    if ends.size == 1:
        ends = np.array([0, 0])
    segment_steps = ends[:-1, None] + np.arange(_SEGMENT_STEPS + 1)
    segment_steps = np.minimum(segment_steps, ends[1:, None])
    segments, width = segment_steps.shape
    # h[c, j]: farthest a pose in segment j lies from the nearer segment end.
    seg_x = rel_x[:, segment_steps]
    seg_y = rel_y[:, segment_steps]
    h = np.max(np.sqrt(np.minimum(
        (seg_x - seg_x[:, :, :1]) ** 2 + (seg_y - seg_y[:, :, :1]) ** 2,
        (seg_x - seg_x[:, :, -1:]) ** 2 + (seg_y - seg_y[:, :, -1:]) ** 2)), axis=2)
    seg_rows_x = seg_x.astype(np.float32).reshape(count * segments, width)
    seg_rows_y = seg_y.astype(np.float32).reshape(count * segments, width)
    collision_keep = (collision_radius + margin) ** 2

    cost = np.empty(count)
    chunk = max(1, _CHUNK_ELEMENTS // max(1, ends.size * points))
    for first in range(0, count, chunk):
        last = min(count, first + chunk)
        size = last - first

        # 2. Coarse pass, layout (candidate, segment end, point).
        end_x = tx[first:last, ends]
        end_y = ty[first:last, ends]
        end_d2 = np.subtract(ox, end_x[:, :, None])
        np.multiply(end_d2, end_d2, out=end_d2)
        dy = np.subtract(oy, end_y[:, :, None])
        np.multiply(dy, dy, out=dy)
        end_d2 += dy
        near_end = np.minimum(end_d2[:, :-1], end_d2[:, 1:])
        upper = np.sqrt(np.min(near_end.reshape(size, -1), axis=1), dtype=np.float64)
        limit = np.maximum(upper, collision_radius)[:, None] + h[first:last] + margin
        tri_c, tri_seg, tri_p = np.nonzero(near_end <= (limit * limit)[:, :, None])
        # Each candidate keeps at least the triple achieving its upper bound;
        # nonzero returns triples grouped by candidate in order.
        tri_c += first

        # 3. Fine pass, layout (triple, pose in segment).
        rows = tri_c * segments + tri_seg
        d2 = ox[tri_p, None] - np.take(seg_rows_x, rows, axis=0)
        np.multiply(d2, d2, out=d2)
        dy = oy[tri_p, None] - np.take(seg_rows_y, rows, axis=0)
        np.multiply(dy, dy, out=dy)
        d2 += dy
        group_start = np.flatnonzero(np.concatenate(([True], tri_c[1:] != tri_c[:-1])))
        group_size = np.diff(np.append(group_start, tri_c.size))
        candidate_min = np.minimum.reduceat(np.min(d2, axis=1), group_start)
        band = np.sqrt(candidate_min, dtype=np.float64) + 2 * margin
        band = np.maximum(band * band, collision_keep)
        sel_tri, sel_col = np.nonzero(d2 <= np.repeat(band, group_size)[:, None])

        # 4. Exact values, same arithmetic as calc_obstacle_cost.
        sel_c = tri_c[sel_tri]
        sel_step = segment_steps[tri_seg[sel_tri], sel_col]
        sel_p = kept[tri_p[sel_tri]]
        sel_dx = ob[sel_p, 0] - traj_x[sel_c, sel_step]
        sel_dy = ob[sel_p, 1] - traj_y[sel_c, sel_step]
        r = np.hypot(sel_dx, sel_dy)
        sel_start = np.flatnonzero(np.concatenate(([True], sel_c[1:] != sel_c[:-1])))
        with np.errstate(divide='ignore'):
            cost[first:last] = 1.0 / np.minimum.reduceat(r, sel_start)

        near = r <= collision_radius
        if near.any():
            near_c = sel_c[near]
            if config.robot_type == RobotType.rectangle:
                near_dx, near_dy = sel_dx[near], sel_dy[near]
                heading = yaw[near_c, sel_step[near]]
                cosine = np.cos(heading)
                sine = np.sin(heading)
                local_x = near_dx * cosine + near_dy * sine
                local_y = -near_dx * sine + near_dy * cosine
                inside = (np.abs(local_x) <= half_length) & (np.abs(local_y) <= half_width)
                cost[near_c[inside]] = float("Inf")
            elif config.robot_type == RobotType.circle:
                cost[near_c] = float("Inf")
    return cost


def _wrapped_abs(angle):
    return np.abs(np.arctan2(np.sin(angle), np.cos(angle)))


def _heading_target(goal_heading):
    if goal_heading == '+x':
        return 0.0
    if goal_heading == '-x':
        return math.pi
    if goal_heading == '+y':
        return math.pi / 2
    if goal_heading == '-y':
        return -math.pi / 2
    raise ValueError('goal_heading must be +x, -x, +y or -y')


def _batch_heading_dist_cost(final, goal_heading):
    if goal_heading == '+x':
        return - final[:, 0]
    if goal_heading == '-x':
        return + final[:, 0]
    if goal_heading == '+y':
        return - final[:, 1]
    if goal_heading == '-y':
        return + final[:, 1]
    raise ValueError('goal_heading must be +x, -x, +y or -y')


def calc_control_and_trajectory(self,x, dw, config, goal, ob,goal_heading):
    """
    calculation final input with dynamic window

    All candidates are rolled out and scored at once; the result matches the
    original per-candidate loop (v outer, yaw rate inner, last minimum wins).
    """
    if goal is not None and goal_heading is not None:
        raise RuntimeError('cannot implement goal and goal_heading at the same time!')
    x_init = x[:]
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    _vy_candidates = sample_trajectory_from_dw(dw,config)
    v_samples = np.asarray(_vy_candidates[0], dtype=float)
    y_samples = np.asarray(_vy_candidates[1], dtype=float)
    # v is the outer loop, yaw rate the inner one.
    v_grid, y_grid = np.meshgrid(v_samples, y_samples, indexing='ij')
    v_flat, y_flat = v_grid.ravel(), y_grid.ravel()
    if v_flat.size == 0:
        return best_u, best_trajectory

    trajectories = predict_trajectories(x_init, v_flat, y_flat, config)
    final = trajectories[:, -1, :]

    to_goal_cost = 0
    if goal_heading is not None:
        target_yaw = _heading_target(goal_heading)
        to_goal_cost = config.to_goal_heading_cost_gain * _wrapped_abs(target_yaw - final[:, 2])
        to_goal_cost = to_goal_cost + config.to_goal_heading_dist_cost_gain * _batch_heading_dist_cost(final, goal_heading)
    if goal is not None:
        dx = goal[0] - final[:, 0]
        dy = goal[1] - final[:, 1]
        to_goal_cost = config.to_goal_node_cost_gain * _wrapped_abs(np.arctan2(dy, dx) - final[:, 2])
        to_goal_cost = to_goal_cost + config.to_goal_node_dist_cost_gain * np.hypot(dx, dy)

    diff = config.max_speed - np.maximum(0, np.abs(final[:, 3]))
    speed_cost = config.speed_cost_gain * diff

    if np.size(ob) == 0:
        obstacle_cost = np.zeros(v_flat.size)
        turn_cost = np.zeros(v_flat.size)
    else:
        # Nearest obstacle to the start pose is shared by every candidate.
        vectors = np.asarray(ob) - trajectories[0, 0, :2]
        norms = np.linalg.norm(vectors, axis=1)
        nearest = int(np.argmin(norms))
        a = vectors[nearest]
        angle_error = math.atan2(a[1], a[0]) - final[:, 2]
        turn_cost = np.maximum(-_wrapped_abs(angle_error), -np.pi/3)
        nearest_distance = float(np.hypot(a[0], a[1]))
        obstacle_cost = _batch_obstacle_cost(trajectories, ob, config, nearest_distance)

    ob_cost = config.obstacle_cost_gain * obstacle_cost
    ob_cost = ob_cost + config.obstacle_turn_cost_gain * turn_cost

    final_cost = to_goal_cost + speed_cost + ob_cost
    final_cost = np.broadcast_to(final_cost, v_flat.shape)

    finite = np.isfinite(final_cost)
    if finite.any():
        min_cost = np.min(final_cost[finite])
        best_index = int(np.flatnonzero(finite & (final_cost == min_cost))[-1])
        best_u = [v_flat[best_index], y_flat[best_index]]
        best_trajectory = trajectories[best_index]

    if abs(best_u[0]) < config.robot_stuck_flag_cons \
            and abs(x[3]) < config.robot_stuck_flag_cons \
            and best_u[1] <config.robot_stuck_flag_cons:
    #if no_choice:
        # to ensure the robot do not get stuck in
        # best v=0 m/s (in front of an obstacle) and
        # best omega=0 rad/s (heading to the goal with
        # angle difference of 0)
        best_u[1] = (-1) ** np.random.randint(1, 2) * config.max_delta_yaw_rate * 0.5

        #self.get_logger().warn('stucked!turning...')
    # else:
    #     if best_cost_list:
    #         self.get_logger().info(f'best cost config:')
    #         self.get_logger().info(f'to_goal:{best_cost_list[0]}')
    #         self.get_logger().info(f'speed_cost:{best_cost_list[1]}')
    #         self.get_logger().info(f'ob_cost:{best_cost_list[2]}')
    #         #self.get_logger().info(f'to_goal_dist_cost:{best_cost_list[3]}')   
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """Return inverse minimum clearance, or inf for a footprint collision."""
    if np.size(ob) == 0:
        return 0.0
    dx = ob[:, 0, None] - trajectory[:, 0]
    dy = ob[:, 1, None] - trajectory[:, 1]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        half_length = config.robot_length / 2
        half_width = config.robot_width / 2
        collision_radius = np.hypot(half_length, half_width)
        # Keep rotated corners in the broad phase despite floating-point rounding.
        collision_radius += 8 * np.finfo(r.dtype).eps * max(1.0, collision_radius)
        obstacle_indices, step_indices = np.nonzero(r <= collision_radius)

        if obstacle_indices.size:
            # Only nearby point/pose pairs need an exact rectangle check.
            yaw = trajectory[:, 2]
            cosine = np.cos(yaw)[step_indices]
            sine = np.sin(yaw)[step_indices]
            nearby_dx = dx[obstacle_indices, step_indices]
            nearby_dy = dy[obstacle_indices, step_indices]
            local_x = nearby_dx * cosine + nearby_dy * sine
            local_y = -nearby_dx * sine + nearby_dy * cosine
            if ((np.abs(local_x) <= half_length)
                    & (np.abs(local_y) <= half_width)).any():
                return float("Inf")
    elif config.robot_type == RobotType.circle:
        if (r <= config.robot_radius).any():
            return float("Inf")

    # All points still contribute to clearance, including broad-phase rejects.
    return 1.0 / np.min(r)

def calc_obstacle_turn_cost(trajectory,ob,config):
    """Return the negative angle between robot-to-obstacle and final heading."""
    if np.size(ob) == 0:
        return 0.0

    vectors = np.asarray(ob) - trajectory[0, :2]
    a = vectors[np.argmin(np.linalg.norm(vectors, axis=1))]
    angle_error = math.atan2(a[1], a[0]) - trajectory[-1, 2]
    return max(   -abs(math.atan2(math.sin(angle_error), math.cos(angle_error)))  ,  -np.pi/3)

def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost

def calc_to_goal_node_dist_cost(trajectory,goal):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    d = np.hypot(dx, dy)
    return d

def calc_to_goal_heading_cost(trajectory,goal_heading):
    """
        Absolute wrapped angle error to the target heading, in radians.
    """
    if goal_heading == '+x':
        target_yaw = 0.0
    elif goal_heading == '-x':
        target_yaw = math.pi
    elif goal_heading == '+y':
        target_yaw = math.pi / 2
    elif goal_heading == '-y':
        target_yaw = -math.pi / 2
    else:
        raise ValueError('goal_heading must be +x, -x, +y or -y')
    angle_error = target_yaw - trajectory[-1, 2]
    return abs(math.atan2(math.sin(angle_error), math.cos(angle_error)))

def calc_to_goal_heading_dist_cost(trajectory,goal_heading):

    if goal_heading == '+x':
        d = - trajectory[-1, 0]
    elif goal_heading == '-x':
        d = + trajectory[-1, 0]
    elif goal_heading == '+y':
        d = - trajectory[-1, 1]
    elif goal_heading == '-y':
        d = + trajectory[-1, 1]
    else:
        raise ValueError('goal_heading must be +x, -x, +y or -y')
    cost = d

    return cost
def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")

def plot_seg(seg):
    rng = np.random.default_rng()

    for segment in seg:
        if len(segment) == 0:
            continue

        points = np.asarray(segment)  
        color = rng.random(3)         

        plt.scatter(points[:, 0], points[:, 1], color=color,s=20, zorder=3)
        #print("ppp")


def animate(predicted_trajectory,x,goal,ob,seg,config=config):

    if show_animation:
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
        plt.plot(x[0], x[1], "xr")
        if goal is not None:
            plt.plot(goal[0], goal[1], "xb")
        plt.plot(ob[:, 0], ob[:, 1], "ok",zorder=1)
        plot_seg(seg)
        plot_robot(x[0], x[1], x[2], config)
        plot_arrow(x[0], x[1], x[2])
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)


def main(gx=10.0, gy=10.0):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi , 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    
    trajectory = np.array(x)
    ob = config.ob
    while True:
        u, predicted_trajectory = dwa_control(x, ob, goal=goal, config=config)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
    # main(robot_type=RobotType.circle)
