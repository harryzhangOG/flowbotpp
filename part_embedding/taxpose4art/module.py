import random

import numpy as np
import theseus as th
import torch

from .cost_function import ObstacleCost


def point_diff_func(optim_vars, aux_vars):
    current_pos = optim_vars[0]
    target_pos, weight, bias = aux_vars
    error = current_pos.tensor - (target_pos.tensor + bias.tensor)
    error = error * weight.tensor
    return error


class TheseusMotionPlanner:
    def __init__(
        self,
        traj_len,
        num_obstacle,
        optimization_steps,
        total_time=10.0,
        env_name=None,
        straight_init=False,
        max_distance=0.5,
    ) -> None:
        torch.set_default_dtype(torch.float32)

        seed = 0
        torch.random.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.prev_traj = None  # Note that it is in the global frame
        self.prev_goal = None
        # self.prev_agent_pose = None
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        "Set up planning parameters"
        self.traj_len = traj_len
        num_time_steps = traj_len - 1
        self.total_time = total_time
        self.env_name = env_name
        self.straight_init = straight_init
        dt_val = total_time / num_time_steps
        Qc_inv = [[1.0, 0.0], [0.0, 1.0]]
        boundary_w = 100.0

        poses = []
        velocities = []
        for i in range(traj_len):
            poses.append(th.Point2(name=f"pose_{i}", dtype=torch.float))
            velocities.append(th.Point2(name=f"vel_{i}", dtype=torch.float))
        start_point = th.Point2(name="start", dtype=torch.float)
        goal_point = th.Point2(name="goal", dtype=torch.float)

        dt = th.Variable(torch.tensor(dt_val).float().view(1, 1), name="dt")
        # Cost weight to use for all GP-dynamics cost functions
        gp_cost_weight = th.eb.GPCostWeight(torch.tensor(Qc_inv).float(), dt)
        # boundary_w = th.Vector(1, name="boundary_w")
        boundary_cost_weight = th.ScaleCostWeight(boundary_w)

        objective = th.Objective(dtype=torch.float)

        # Fixed starting position
        objective.add(
            th.Difference(poses[0], start_point, boundary_cost_weight, name="pose_0")
        )
        objective.add(
            th.Difference(poses[-1], goal_point, boundary_cost_weight, name="pose_N")
        )

        obstacle_pos_list = []
        for i in range(num_obstacle):
            obstacle_pos_list.append(th.Point2(name=f"obstacle_{i}"))

        obstacle_cost_weight = th.ScaleCostWeight(1.0)
        for i, obstacle_pos in enumerate(obstacle_pos_list):
            for j, pose in enumerate(poses):
                objective.add(
                    ObstacleCost(
                        cost_weight=obstacle_cost_weight,
                        agent_pos=pose,
                        obstacle_pos=obstacle_pos,
                        max_distance=max_distance,
                        name=f"agent_{j}_obstacle_{i}",
                    )
                )
        for i in range(1, traj_len):
            objective.add(
                (
                    th.eb.GPMotionModel(
                        poses[i - 1],
                        velocities[i - 1],
                        poses[i],
                        velocities[i],
                        dt,
                        gp_cost_weight,
                        name=f"gp_{i}",
                    )
                )
            )

        optimizer = th.LevenbergMarquardt(
            objective,
            th.CholeskyDenseSolver,
            max_iterations=optimization_steps,
            step_size=0.5,
        )
        self.motion_planner = th.TheseusLayer(optimizer)
        self.motion_planner.to(self.device)

    def plan(self, start, goal, obstacles):
        "Note that only works for single batch right now"
        start = torch.tensor(start).unsqueeze(0).float()
        goal = torch.tensor(goal).unsqueeze(0).float()
        input_dict = {"start": start.to(self.device), "goal": goal.to(self.device)}
        input_dict.update(self.get_straight_line_inputs(start, goal))
        for i, pos in enumerate(obstacles):
            input_dict.update(
                {
                    f"obstacle_{i}": torch.tensor(pos)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                }
            )
        # input specific to each environment
        final_values, info = self.motion_planner.forward(
            input_dict,
            optimizer_kwargs={
                "track_best_solution": True,
                "verbose": True,
                "damping": 0.1,
            },
        )
        global_traj = info.best_solution
        # all the trajectory output outside of the class are within global coordinate
        self.prev_traj = global_traj
        return global_traj

    def get_straight_line_inputs(self, start, goal):
        # Returns a dictionary with pose and velocity variable names associated to a
        # straight line trajectory between start and goal
        start_goal_dist = goal - start
        avg_vel = start_goal_dist / self.total_time
        unit_trajectory_len = start_goal_dist / (self.traj_len - 1)
        input_dict = {}
        for i in range(self.traj_len):
            input_dict[f"pose_{i}"] = (
                (start + unit_trajectory_len * i + 1e-4).to(self.device).float()
            )
            if i == 0 or i == self.traj_len - 1:
                input_dict[f"vel_{i}"] = torch.zeros_like(avg_vel).to(self.device)
            else:
                input_dict[f"vel_{i}"] = avg_vel.to(self.device)
        return input_dict
