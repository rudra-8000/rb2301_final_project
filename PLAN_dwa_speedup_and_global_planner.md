# Plan: faster DWA + separate global planner node

## Context

The single-point-goal DWA on branch `Switch-back-to-point-goal-approach` gets trapped in local minima in the zig-zag corridor, because its goal cost is straight-line distance over a ~1 m horizon. The fix is a global planner that feeds DWA a nearby waypoint ("carrot"). Two things block that today:

- The control callback takes ~125 ms on the Jetson Nano (target is 50 ms). Profiling on the laptop shows ~70% is the per-candidate obstacle cost and ~20% the Python-loop trajectory rollout.
- There is no map or planner. It must run as its own process so it lands on a different core from the controller, and must run on the robot's Foxy / Python 3.8.

Two subagents run in parallel in this checkout. Their files do not overlap. Neither commits, switches branch, or touches the other's files.

| | Agent A: DWA speed-up | Agent B: global planner |
|---|---|---|
| Edits | `rb2301_fp/dynamic_window_approach.py`, `test/test_collision_checking.py` | `rb2301_fp/obstacle_course.py`, `setup.py`, `README.md` |
| Creates | optionally `test/test_dwa_batch.py` | `rb2301_fp/global_planner_core.py`, `rb2301_fp/global_planner.py`, `gp.sh`, `test/test_global_planner_core.py`, `test/test_path_following.py` |

(All paths under `src/rb2301_fp/` except `gp.sh` and `README.md` at the repo root.)

---

## Brief for Agent A — speed up DWA without changing its decisions

**Goal.** Make `dwa_control` in `src/rb2301_fp/rb2301_fp/dynamic_window_approach.py` several times faster while choosing the same command for the same inputs. This is a pure performance change.

**Background.** Each cycle evaluates up to (v_sample_size+1) × (yaw_sample_size+1) candidates (64–80). For each one, `predict_trajectory` runs a Python loop with `np.vstack` per step, and `calc_obstacle_cost` builds a points × steps distance matrix. Laptop baseline with the real-robot config: 8.5 ms at 360 points, 11.4 ms at 575, 27.6 ms at 1150.

**What to do.**
1. Run the existing tests first and record which pass. `test_dwa_commands_and_trajectories_match_reference` calls `dwa_control(state, obstacles, ...)` without the leading `self` argument the function now takes, so expect it to be stale; fix the call, do not delete the test.
2. Roll out all candidates at once inside `calc_control_and_trajectory`. Keep the exact discrete model of `motion()` (yaw updated first, then position) and the exact step count the `while time <= predict_time` loop produces, using cumulative sums rather than a different closed form.
3. Compute the obstacle cost for all candidates in one batched operation. Crop obstacle points first, but only with a crop that provably cannot change the result: with `d_near` = distance from the start pose to its nearest point and `L` = the largest displacement any candidate can reach, keep points within `max(d_near + 2L, L + collision_radius)` of the start pose. Points beyond that can never be the minimum clearance or a collision.
4. `calc_obstacle_turn_cost` finds the nearest obstacle to the start pose, which is the same for every candidate; compute it once.
5. Keep the tie rule: the current loop uses `min_cost >= final_cost` with v as the outer loop and yaw rate as the inner loop, so the last minimum wins.

**Constraints.**
- Keep the public functions and their signatures (`dwa_control`, `predict_trajectory`, `calc_obstacle_cost`, `calc_obstacle_turn_cost`, the goal-cost functions, `Config`). Other tests and `obstacle_course.py` use them.
- `calc_obstacle_cost` called directly must still behave as today, including `test_clearance_still_uses_distant_points`.
- No behaviour fixes. In particular leave the stuck-recovery block as it is, even though `np.random.randint(1, 2)` always returns 1.
- No new dependencies (numpy only, no scipy). Code must parse and run on Python 3.8.
- Do not edit `obstacle_course.py`.

**Tests.** Freeze the current `calc_control_and_trajectory` as a reference inside the test file and compare old vs new over the room and corridor scenes already in `test_collision_checking.py`, both robot types, both `Config(is_simulation=True/False)`, heading goals and point goals: chosen `[v, yaw_rate]` exactly equal, trajectories equal to 1e-9. Update the file's `--benchmark` mode so it times the frozen reference against the new code.

**Done when.** All tests in `src/rb2301_fp/test/` that passed before still pass; the equivalence test passes; the benchmark shows old vs new at 360 / 720 / 1440 points for room and corridor. Aim for at least 4× on the corridor scene; report the real numbers even if lower. Say explicitly that nothing was measured on the Jetson.

---

## Brief for Agent B — global planner node and path following

**Goal.** Add a second ROS node, run as its own process, that builds an occupancy grid from LiDAR and pose, plans a path to the controller's fixed goal about once a second, and publishes it. Make the controller follow that path by handing DWA a waypoint on it instead of the far goal.

**Background.** Read `src/rb2301_fp/rb2301_fp/obstacle_course.py` first. In single-point-goal mode the controller freezes one goal at the first pose (`_initialize_point_goal`) and passes it to DWA every cycle. Pose is `[x, y, heading_deg]`: Gazebo odom in simulation, CA2 maze frame on the real robot (`optitrack_to_maze`, `load_frame`). `update_ob` shows the LiDAR mounting (0.11423 m forward, yaw offset π) and scan-to-world conversion. Robot footprint is about 0.20–0.24 m wide; the gate is 0.6–0.8 m wide; the last section has swinging obstacles.

**Structure.**
- `global_planner_core.py` — no ROS imports. Grid, map update, inflation, planner, path post-processing. This is what the unit tests exercise.
- `global_planner.py` — thin `rclpy` node around the core, with `main()`; add a `global_planner` console script in `setup.py` and a `gp.sh` beside `fp.sh` that runs it.

**Map (core).**
- Fixed grid in the pose frame, centred on the first pose. Defaults: 0.05 m cells, 10 m half-size; both parameters.
- Per scan: mark hit cells occupied and clear cells along each beam up to the hit, using hit/miss evidence so a swinging obstacle that has moved away gets cleared. Ignore invalid/infinite beams. Limit mapping range (default 3.5 m) and skip scans taken while turning fast, to limit smearing from pose/scan timing skew. Vectorise with numpy; throttle map updates if needed (about 5 Hz is enough).
- Unknown cells count as free.
- Inflate occupied cells by robot half-width plus a small margin (default 0.12 + 0.04 m) as impassable, plus a decaying cost out to about 0.3 m so paths keep to the middle of corridors. A 0.6 m gate must stay passable with the defaults. No scipy.

**Planner (core).**
- A* on the 8-connected grid from the robot cell to the goal cell, at about 1 Hz.
- If the robot's own cell is inside inflation, allow it to plan out. If the goal cell is blocked or unreachable, return the path to the reachable cell closest to the goal.
- Output world-frame points at roughly 0.1 m spacing.

**Node.**
- Detect simulation vs real the same way the controller does and get the pose the same way; import `LidarNode`, `optitrack_to_maze`, `load_frame` from `obstacle_course.py` rather than copying them.
- Subscribe to `scan` and to the goal published by the controller (below). Publish `nav_msgs/Path` on `global_path`, frame id matching the controller's `visualization_frame`. Publish an empty path when no path exists.
- Log map-update and planning time at a low rate.

**Controller changes (`obstacle_course.py`, keep them small).**
- Publish the frozen goal once it exists on a reliable, transient-local topic (`geometry_msgs/PointStamped`, `global_goal`).
- New parameters: `use_global_path` (default true), `path_lookahead` (default 0.8 m), `path_timeout` (default 3.0 s).
- Subscribe to `global_path`. Each cycle: find the closest path point to the robot, walk `path_lookahead` further along, and pass that point to `dwa_control` as `goal`. Near the end of the path the waypoint is the final goal.
- Keep `_goal.goal_node` as the final goal; `is_goal_reached` and stopping logic stay as they are.
- If there is no path, the path is empty, or it is older than `path_timeout`, pass the final goal to DWA exactly as today. With `use_global_path:=false` behaviour must be identical to today.
- Add the path (line strip) and the current waypoint to `publish_rviz_visualization` so they appear through both the `ros` and `ssh` transports.
- Do not touch the staged heading mode, and do not edit `dynamic_window_approach.py`.

**Constraints.** Must run on Foxy with Python 3.8: only `rclpy` features the controller already uses, no newer syntax (`test_visualization_bridge.py` has a Python 3.8 parse check to copy for the new modules). numpy only.

**Tests.**
- Core (no ROS): hits become occupied; a previously occupied cell is cleared after beams pass through it; a goal behind a U-shaped wall gives a path that goes around it; a 0.6 m gap is crossed with default inflation; goal inside a wall returns a path ending at the nearest reachable cell; print planning time for a 400 × 400 grid.
- Controller (pattern of `test_single_point_goal.py`): waypoint selection along a path; fallback to the final goal when the path is missing, empty or stale; `use_global_path:=false` leaves the DWA call unchanged.
- Existing tests still pass.

**End-to-end.** Try a simulation run: `./gz_fp.sh`, then `./fp.sh` and `./gp.sh` in separate shells, and record whether the robot reaches the goal with obstacles enabled. If Gazebo cannot be run in the agent's environment, say so plainly instead of claiming it works.

**Done when.** Tests pass; README has a short section on starting the planner (simulation and robot) and the new parameters; the report lists laptop planning and map-update times, the end-to-end result or why it was not run, and anything left unverified on the real robot.

---

## Verification after both finish

1. `source /opt/ros/jazzy/setup.bash`, `colcon build --symlink-install --packages-select rb2301_fp`, then `python3 -m pytest src/rb2301_fp/test -q`.
2. `python3 src/rb2301_fp/test/test_collision_checking.py --benchmark` for the DWA numbers.
3. Simulation: `./gz_fp.sh`, `./fp.sh`, `./gp.sh`, `bash rviz_fp.sh`; check the path and waypoint in RViz and that the robot clears the zig-zag, gate and moving obstacles over several randomised launches.
4. On the robot (yours to run): sync, build, check `htop` for a free core, compare the `Performance:` log line before and after, then run with `./gp.sh` in an extra tmux pane.

## Defaults I chose that you may want to change

- `use_global_path` defaults to on, with automatic fallback to today's behaviour when no fresh path arrives.
- 0.05 m grid, 10 m half-size, 0.16 m hard inflation, 0.8 m lookahead.
- The small bugs noted earlier (stuck-recovery direction, README default goal values) are left out of both briefs.
