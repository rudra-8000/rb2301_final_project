"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

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
            self.dt = 0.1  # [s] Time tick for motion prediction
            self.predict_time = 2.5  # [s]

            self.to_goal_node_cost_gain = 0.5      # in angle
            self.to_goal_node_dist_cost_gain = 2 # in distance

            self.to_goal_heading_cost_gain = 0.1        #in angle
            self.to_goal_heading_dist_cost_gain = 0.5   #in distance

            self.speed_cost_gain = 1.2
            self.speed_back_gain = 0.7

            self.obstacle_cost_gain = 0.1
            self.obstacle_turn_cost_gain = 1.0

            self.robot_stuck_flag_cons = 0.01  #was 0.001 constant to prevent robot stucked
            self.robot_type = RobotType.rectangle

            # if robot_type == RobotType.circle
            # Also used to check if goal is reached in both types
            self.robot_radius = 1.0  # [m] for collision check

            # if robot_type == RobotType.rectangle
            self.robot_width = 0.23  # [m] for collision check
            self.robot_length = 0.23  # [m] for collision check

            # obstacles [x(m) y(m), ....]
        
        else:   #IRL param
                        # robot parameter
            self.max_speed = 0.25         # [m/s]
            self.min_speed = -0.1  # [m/s]
            self.max_yaw_rate = 90.0 * math.pi / 180.0  # [rad/s]
            self.max_accel = 0.2  # [m/ss]
            self.max_delta_yaw_rate = 90.0 * math.pi / 180.0  # [rad/ss]
            self.v_resolution = 0.02  # [m/s]
            self.yaw_rate_resolution = 1.0 * math.pi / 180.0  # [rad/s]
            self.dt = 0.1  # [s] Time tick for motion prediction
            self.predict_time = 2.0  # [s]

            self.to_goal_node_cost_gain = 0.5      # in angle
            self.to_goal_node_dist_cost_gain = 0.5 # in distance

            self.to_goal_heading_cost_gain = 0.1        #in angle
            self.to_goal_heading_dist_cost_gain = 0.5   #in distance

            self.speed_cost_gain = 0.3
            self.speed_back_gain = 0.2

            self.obstacle_cost_gain = 0.08
            self.obstacle_turn_cost_gain = 0.5

            self.robot_stuck_flag_cons = 0.0001  #was 0.001 constant to prevent robot stucked
            self.robot_type = RobotType.rectangle

            # if robot_type == RobotType.circle
            # Also used to check if goal is reached in both types
            self.robot_radius = 1.0  # [m] for collision check

            # if robot_type == RobotType.rectangle
            self.robot_width = 0.2  # [m] for collision check
            self.robot_length = 0.2  # [m] for collision check
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


def calc_control_and_trajectory(self,x, dw, config, goal, ob,goal_heading):
    """
    calculation final input with dynamic window
    """
    if goal is not None and goal_heading is not None:
        raise RuntimeError('cannot implement goal and goal_heading at the same time!')
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    best_cost_list = []
    no_choice = True
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = 0
            
            if goal_heading is not None:
                to_goal_cost = config.to_goal_heading_cost_gain * calc_to_goal_heading_cost(trajectory,goal_heading)
                to_goal_cost += config.to_goal_heading_dist_cost_gain * calc_to_goal_heading_dist_cost(trajectory,goal_heading)
            if goal is not None:
                to_goal_cost = config.to_goal_node_cost_gain * calc_to_goal_cost(trajectory, goal)
                to_goal_cost += config.to_goal_node_dist_cost_gain * calc_to_goal_node_dist_cost(trajectory,goal)

            diff = config.max_speed - max(0,trajectory[-1, 3])
            if diff > 0:
                speed_cost = config.speed_cost_gain * diff
            else:
                speed_cost = config.speed_back_cost_gain * diff
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)
            ob_cost += config.obstacle_turn_cost_gain * calc_obstacle_turn_cost(trajectory,ob,config)

            final_cost = to_goal_cost + speed_cost + ob_cost
            cost_list = [to_goal_cost,speed_cost , ob_cost]
            # search minimum trajectory
            if np.isfinite(final_cost) and min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                best_cost_list = cost_list
                no_choice = False
    if abs(best_u[0]) < config.robot_stuck_flag_cons \
            and abs(x[3]) < config.robot_stuck_flag_cons:
    #if no_choice:
        # to ensure the robot do not get stuck in
        # best v=0 m/s (in front of an obstacle) and
        # best omega=0 rad/s (heading to the goal with
        # angle difference of 0)
        best_u[1] = (-1) ** np.random.randint(1, 2) * config.max_delta_yaw_rate * 0.5

        #self.get_logger().warn('stucked!turning...')
    #else:
        #if best_cost_list:
            #self.get_logger().info(f'best cost config:')
            #self.get_logger().info(f'to_goal:{best_cost_list[0]}')
            #self.get_logger().info(f'speed_cost:{best_cost_list[1]}')
            #self.get_logger().info(f'ob_cost:{best_cost_list[2]}')
            #self.get_logger().info(f'to_goal_dist_cost:{best_cost_list[3]}')   
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
