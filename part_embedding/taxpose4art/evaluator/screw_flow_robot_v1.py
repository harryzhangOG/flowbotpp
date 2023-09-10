import os
import pickle

import numpy as np
import pybullet as p
import torch
from scipy.spatial.transform import Rotation as R

from part_embedding.datasets.calc_art import compute_new_points
from part_embedding.datasets.pm.pm_raw import PMRawData
from part_embedding.taxpose4art.generate_art_training_data import transform_pcd
from part_embedding.taxpose4art.run_goalflow_robot import RobotEvalualtor
from part_embedding.taxpose4art.suction_env import PMSuctionEnv
from part_embedding.taxpose4art.train_utils import record_actuation_result

"""
Improved FlowBot V2. Generic MPC.

Every k steps:

- Given flow estimation
- Given local screw estimation
- GS
    - Then use Gram-Schmidt to “correct” the local screw estimation
    - —> flow cross screw gives us axis
    - trajectory
- Execute the new flow
"""


def rotate_on_the_fly(
    axis: torch.Tensor, origin: torch.Tensor, pts: np.ndarray, angle: float
):
    """
    Take as input tensor axis, origin, and pts

    Perform 3D rotation using the axis and angle.
    """
    with torch.no_grad():
        r_mat = R.from_rotvec(axis.cpu().numpy() * angle / 180 * np.pi).as_matrix()
        rotated_pts = r_mat @ (
            pts.reshape((-1, 3)).T - origin.cpu().numpy().reshape(3, 1)
        ) + origin.cpu().numpy().reshape((3, 1))
    return rotated_pts.T


class RobotEvaluatorMPC(RobotEvalualtor):
    def __init__(
        self,
        obj_id,
        model,
        screw_model,
        sem_class_model,
        results_act,
        split_name,
        offline=False,
        action=None,
        anchor=None,
        flowbot=False,
        render=False,
    ):
        self.goalflownet = model
        self.screwnet = screw_model
        self.offline = offline
        self.classifier = sem_class_model
        if self.offline:
            self.anchor = anchor
            self.action = action
        self.obj_id = obj_id
        self.generated_metadata = pickle.load(
            open(f"part_embedding/taxpose4art/flowbot_data/{split_name}.pkl", "rb")
        )
        tmp = obj_id.split("_")[0] if "_" in obj_id else obj_id
        self.env = PMSuctionEnv(
            tmp,
            os.path.expanduser("~/partnet-mobility/raw"),
            camera_pos=[-2.5, 0, 2.5],
            gui=False,
        )
        self.flowbot = flowbot  # If set to True, will open to the max
        self.full_sem_dset = pickle.load(
            open(f"part_embedding/taxpose4art/flowbot_split/{split_name}.pkl", "rb")
        )
        self.render = render
        self.results_act = results_act
        self.pm_raw_data = PMRawData(
            os.path.join(os.path.expanduser("~/partnet-mobility/raw"), tmp)
        )

    def proj_uv(self, u, v):
        proj = (
            np.dot(u.cpu(), v.cpu().T).diagonal()
            / np.dot(u.cpu(), u.cpu().T).diagonal()
        ).reshape(-1, 1) * u.cpu().numpy()
        return torch.from_numpy(proj).cuda()

    def naive_cross_product(self, screw, art_flow):
        axis_agg = torch.cross(art_flow, screw).mean(dim=0)
        axis_agg = axis_agg / axis_agg.norm()
        return axis_agg

    def gram_schmidt(self, screw, art_flow):
        with torch.no_grad():
            screw_corrected = screw - self.proj_uv(art_flow, screw)
            axis_agg = torch.cross(art_flow, screw_corrected).mean(dim=0)
            axis_agg = axis_agg / axis_agg.norm()
        return axis_agg

    def infer_screw_flow(self, action, anchor, gs=True):
        """
        pred_flow: This is GOAL FLOW
        art_flow: This is articulated flow (FlowBot v1)
        screw_un: This is local screw projection displacement (need to coin a term for this)
        """
        with torch.no_grad():
            pred_flow = self.goalflownet(
                action,
                anchor,
            )
            R_pred, t_pred = self.goalflownet.svd(action.pos, pred_flow, anchor.flow)
            pred_pose = (
                (
                    torch.bmm(action.pos.reshape(-1, 500, 3), R_pred.transpose(-1, -2))
                    + t_pred
                )
                .reshape(-1, 3)
                .cuda()
            )

            pred_flow = pred_flow.cpu().numpy()
            pred_flow[500:] = 0
            (
                origin,
                projection,
                pred_flow_screw,
                axis_agg,
                max_flow_idx,
            ) = self.infer_screw(action, anchor, gs, pred_flow)
            return (
                pred_flow,
                origin,
                projection,
                pred_pose,
                pred_flow_screw,
                axis_agg,
                max_flow_idx,
            )

    def infer_screw(self, action, anchor, gs, pred_flow, max_flow_idx=None):
        with torch.no_grad():
            pred_flow_screw = self.screwnet(
                action,
                anchor,
            )
        screw_un = pred_flow_screw[:500][:, 3:]
        screw = screw_un / screw_un.norm(dim=1).reshape(-1, 1)
        art_flow = pred_flow_screw[:500][:, :3]
        art_flow = art_flow / art_flow.norm(dim=1).reshape(-1, 1)

        # Infer axis from screw estimation and art flow
        try:
            origin = screw_un + action.pos
        except:
            breakpoint()
        if max_flow_idx is None:
            max_change_pos = np.linalg.norm(pred_flow, axis=1)
            max_flow_idx = np.argpartition(max_change_pos, -20)[-20:]

        if self.sem == "hinge":
            if gs:
                axis_agg = self.gram_schmidt(
                    screw[max_flow_idx], art_flow[max_flow_idx]
                )
            else:
                axis_agg = self.naive_cross_product(screw, art_flow)
        else:
            axis_agg = art_flow[max_flow_idx].mean(dim=0)
            axis_agg = axis_agg / axis_agg.norm()
        origin = torch.mean(origin[max_flow_idx], axis=0)
        projection = lambda x: (
            axis_agg.reshape(3, 1) @ axis_agg.reshape(3, 1).T
        ) @ x.T / torch.dot(axis_agg, axis_agg) + (
            (
                torch.eye(3).cuda()
                - (axis_agg.reshape(3, 1) @ axis_agg.reshape(3, 1).T)
                / torch.dot(axis_agg, axis_agg)
            )
            @ origin
        ).reshape(
            3, -1
        )

        return origin, projection, pred_flow_screw, axis_agg, max_flow_idx

    def check_dir(self, projection, max_flow_pt, pred_flow, max_flow_idx, temp):
        """
        4 points A, B, C, D
        A is proj point on the axis
        B is max_flow_pt (contact)
        C is goal flow pt
        D is ambiguous rotation point

        Check if (AB x AC) . (AB x AD) > 0 if so, same direction
        """
        with torch.no_grad():
            pointA = (
                projection(torch.from_numpy(max_flow_pt.reshape(1, 3)).cuda())
                .cpu()
                .numpy()
                .squeeze()
            )
            pointB = max_flow_pt
            pointC = pred_flow[max_flow_idx].mean(axis=0) + max_flow_pt
            pointD = temp[max_flow_idx].mean(axis=0)

            disp0 = np.cross(pointB - pointA, pointC - pointA)
            disp0 = disp0 / np.linalg.norm(disp0)
            disp1 = np.cross(pointB - pointA, pointD - pointA)
            disp1 = disp1 / np.linalg.norm(disp1)
        return np.dot(disp0, disp1) < 0

    def get_metric_quantities(self, action):
        chain = self.pm_raw_data.obj.get_chain(self.move_joints)
        current_ja = np.zeros(len(chain))
        target_ja = np.zeros(len(chain))
        if self.sem == "slider":
            target_ja[-1] = self.end_ang - self.init_ang
        else:
            target_ja[-1] = (self.end_ang - self.init_ang) / 180 * np.pi
        gt_tf = compute_new_points(
            action.pos.cpu().numpy(),
            self.env.T_world_base,
            chain,
            current_ja=current_ja,
            target_ja=target_ja,
            return_transform=True,
        )
        gt_pcd = transform_pcd(action.pos.cpu().numpy(), gt_tf)

        chain = self.pm_raw_data.obj.get_chain(self.move_joints)
        current_ja = np.zeros(len(chain))
        target_ja = np.zeros(len(chain))
        if self.sem == "slider":
            target_ja[-1] = self.res_angle - self.init_ang
        else:
            target_ja[-1] = (self.res_angle - self.init_ang) / 180 * np.pi
        achieved_tf = compute_new_points(
            action.pos.cpu().numpy(),
            self.env.T_world_base,
            chain,
            current_ja=current_ja,
            target_ja=target_ja,
            return_transform=True,
        )
        achieved_pcd = transform_pcd(action.pos.cpu().numpy(), achieved_tf)
        return gt_pcd, achieved_pcd

    def project_to_plane(self, point, axis, origin):
        v = point - origin
        dist = np.dot(v, axis)
        projected = point - dist * axis
        return projected

    def get_target_angle(
        self,
        axis_agg,
        max_flow_pt,
        action,
        goal_flowed,
        projection,
        gt_ang=True,
    ):
        if gt_ang:
            target_angle = action.loc.item()
        else:
            if self.sem == "hinge":
                plane_start = (
                    projection(torch.from_numpy(max_flow_pt.reshape(1, 3)).cuda())
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                projected_goal_flowed = self.project_to_plane(
                    goal_flowed, axis_agg, plane_start
                )
                projected_vec = projected_goal_flowed - plane_start
                projected_vec /= np.linalg.norm(projected_vec)

                max_flow_pt_vec = max_flow_pt - plane_start
                max_flow_pt_vec /= np.linalg.norm(max_flow_pt_vec)

                target_angle = np.dot(projected_vec, max_flow_pt_vec) / np.pi * 180
            else:
                end = (
                    projection(torch.from_numpy(goal_flowed.reshape(1, 3)).cuda())
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                target_angle = np.linalg.norm(end - max_flow_pt)

        return target_angle

    def get_curr_angle(self):
        joint_angles = self.env.get_joint_angles()
        for k in joint_angles:
            if k == self.target_joint:
                if self.sem == "hinge":
                    self.res_angle = joint_angles[k] / np.pi * 180
                else:
                    self.res_angle = joint_angles[k]
        return self.res_angle

    def get_execution_trajectory(
        self,
        axis_agg,
        origin,
        max_flow_pt,
        trajectory,
        target_angle,
    ):
        for interv in range(10):
            if self.sem == "hinge":
                trajectory = np.vstack(
                    [
                        trajectory.reshape(-1, 3),
                        rotate_on_the_fly(
                            axis_agg,
                            origin,
                            max_flow_pt.reshape(-1, 3),
                            target_angle / 10 * (interv + 1),
                        ).reshape(-1, 3),
                    ]
                )
            else:
                trajectory = np.vstack(
                    [
                        trajectory.reshape(-1, 3),
                        (
                            max_flow_pt
                            + axis_agg.cpu().numpy() * target_angle / 10 * (interv + 1)
                        ).reshape(-1, 3),
                    ]
                )
        trajectory = np.array(trajectory).squeeze()
        return trajectory

    def debug_vis(
        self,
        axis_agg,
        origin,
        projection,
        anchor,
        action,
        pred_flow_screw,
        temp,
        trajectory,
        max_flow_pt,
        max_flow_idx,
    ):
        """
        Debug
        """
        import trimesh

        points = axis_agg * torch.linspace(-1, 1, 100).reshape(100, 1).cuda() + origin
        line = trimesh.points.PointCloud(points.cpu(), colors=(255, 0, 0))
        act = trimesh.points.PointCloud(anchor.pos.cpu())
        proj = trimesh.points.PointCloud(
            projection(action.pos).cpu().T, colors=(0, 255, 0)
        )
        add = trimesh.points.PointCloud(
            (pred_flow_screw[:500][:, 3:] + action.pos).cpu(), colors=(0, 0, 255)
        )
        inst_flow = trimesh.points.PointCloud(
            pred_flow_screw[:500][:, :3]
            .cpu()[max_flow_idx]
            .mean(axis=0)
            .reshape((-1, 3))
            + max_flow_pt,
            colors=(0, 255, 255),
        )
        if self.sem == "hinge":
            rot_pts = trimesh.points.PointCloud(temp, colors=(255, 255, 20))
        traj = trimesh.points.PointCloud(trajectory, colors=(255, 0, 255))
        if self.sem == "hinge":
            scene = trimesh.Scene([act, line, proj, add, rot_pts, traj, inst_flow])
        else:
            scene = trimesh.Scene([act, line, proj, add, traj, inst_flow])
        scene.show()
        """"""

    def mpc_step(self, loc, goal_flowed, exec_step=3, gt_ang=True):
        try:
            action, anchor = self.obs()
        except:
            return None
        pos, ori = p.getBasePositionAndOrientation(
            self.env.gripper.body_id, physicsClientId=self.env.client_id
        )
        # Add synthetic point
        # action.pos[499, 0] = pos[0]
        # action.pos[499, 1] = pos[1]
        # action.pos[499, 2] = pos[2]
        # anchor.pos[499, 0] = pos[0]
        # anchor.pos[499, 1] = pos[1]
        # anchor.pos[499, 2] = pos[2]

        action.loc = loc
        origin, projection, pred_flow_screw, axis_agg, max_flow_idx = self.infer_screw(
            action, anchor, True, self.pred_flow
        )

        if self.sem_aux == "hinge":
            temp = rotate_on_the_fly(
                axis_agg, origin, action.pos.cpu().numpy(), action.loc.item()
            )
            # Check direction with predicted flow
            dir_swap = self.check_dir(
                projection,
                self.max_flow_pt,
                # pred_flow_screw[:500][:, :3].cpu(),
                self.pred_flow,
                self.max_flow_idx,
                temp,
            )

            if dir_swap:
                axis_agg = -axis_agg
                temp = rotate_on_the_fly(
                    axis_agg, origin, action.pos.cpu().numpy(), action.loc.item()
                )
                print("Disambiguated the direction")
        else:
            temp = self.pred_flow.mean(axis=0)
            temp = temp / np.linalg.norm(temp)
            dir_swap = np.dot(temp, axis_agg.cpu().numpy()) < 0
            if dir_swap:
                axis_agg = -axis_agg
                print("Disambiguated the direction")

        trajectory = np.array(pos)
        goal_flowed = self.max_flow_pt + np.mean(
            self.pred_flow[self.max_flow_idx], axis=0
        )
        target_angle = self.get_target_angle(
            axis_agg, pos, action, goal_flowed, projection, gt_ang
        )
        trajectory = self.get_execution_trajectory(
            axis_agg,
            origin,
            np.array(pos),
            trajectory,
            target_angle,
        )
        # self.debug_vis(
        #     axis_agg,
        #     origin,
        #     projection,
        #     anchor,
        #     action,
        #     pred_flow_screw,
        #     temp,
        #     trajectory,
        #     trajectory[0],
        #     max_flow_idx,
        # )
        return trajectory[:exec_step]

    def move_gripper_traj_with_feedback(self, traj, target_angle):
        imgs = self.env.move_gripper_vel_to(traj[0])
        for i in range(1, len(traj)):
            vel = traj[i] - traj[i - 1]
            rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.env.render(
                True
            )
            imgs.append(rgb)
            for j in range(1, 10):
                self.env.gripper.set_pose(traj[i - 1] + 0.1 * j * vel)
                p.stepSimulation(self.env.client_id)
                if self.sem == "hinge":
                    early_stop = np.abs(self.get_curr_angle() - target_angle) < 5
                else:
                    early_stop = np.abs(self.get_curr_angle() - target_angle) < 0.05
                if early_stop:
                    (
                        rgb,
                        depth,
                        seg,
                        P_cam,
                        P_world,
                        P_rgb,
                        pc_seg,
                        segmap,
                    ) = self.env.render(True)
                    imgs.append(rgb)
                    break

        return imgs, early_stop

    def run_eval(self, gs=True, mpc=True, exec_step=3, gt_ang=True):
        self.get_joint_metadata()
        self.target_joint = [
            x for x in self.pm_raw_data.obj.joints if x.child == self.move_joints
        ][0].name

        # if self.sem == "hinge":
        #     return None

        if self.offline:
            self.actuate_obj_in_env()
        else:
            self.randomize_obj_in_env()
        if self.offline:
            action, anchor = self.obs_offline()
        else:
            action, anchor = self.obs()
        self.sem_aux = (
            "hinge" if self.classifier(action, anchor).mean() < 0.5 else "slider"
        )
        if self.flowbot:
            if self.sem == "hinge":
                action.loc = torch.Tensor(
                    [self.end_ang / np.pi * 180 - self.init_ang]
                ).cuda()
            else:
                action.loc = torch.Tensor([self.end_ang - self.init_ang]).cuda()

        # Run inference
        (
            self.pred_flow,
            origin,
            projection,
            pred_pose,
            pred_flow_screw,
            axis_agg,
            self.max_flow_idx,
        ) = self.infer_screw_flow(action, anchor, gs=gs)

        # Add the robot to the env
        self.env.add_gripper_to_env()

        # Teleport the gripper to max change point (MAX "flow")
        top_k = 10

        # max_change_pos = np.linalg.norm(self.pred_flow, axis=1)
        # max_change_pos = np.linalg.norm(pred_flow_screw[:500][:, :3].cpu(), axis=1)
        # self.max_flow_idx = np.argpartition(max_change_pos, -top_k)[-top_k:]
        anchor_pcd = anchor.pos.cpu().numpy()
        anchor_pcd_transform = anchor_pcd + self.pred_flow
        self.max_flow_pt = np.mean(anchor_pcd[self.max_flow_idx], axis=0)

        if self.sem_aux == "hinge":
            temp = rotate_on_the_fly(
                axis_agg, origin, action.pos.cpu().numpy(), action.loc.item()
            )
            # Check direction with predicted flow
            dir_swap = self.check_dir(
                projection,
                self.max_flow_pt,
                # pred_flow_screw[:500][:, :3].cpu(),
                self.pred_flow,
                self.max_flow_idx,
                temp,
            )

            if dir_swap:
                axis_agg = -axis_agg
                temp = rotate_on_the_fly(
                    axis_agg, origin, action.pos.cpu().numpy(), action.loc.item()
                )
                print("Disambiguated the direction")
        else:
            temp = self.pred_flow.mean(axis=0)
            temp = temp / np.linalg.norm(temp)
            dir_swap = np.dot(temp, axis_agg.cpu().numpy()) < 0
            if dir_swap:
                axis_agg = -axis_agg
                print("Disambiguated the direction")

        trajectory = self.max_flow_pt.reshape(
            -1,
        )
        goal_flowed = self.max_flow_pt + np.mean(
            self.pred_flow[self.max_flow_idx], axis=0
        )
        target_angle = self.get_target_angle(
            axis_agg, self.max_flow_pt, action, goal_flowed, projection, gt_ang
        )
        trajectory = self.get_execution_trajectory(
            axis_agg,
            origin,
            self.max_flow_pt,
            trajectory,
            target_angle,
        )

        # self.debug_vis(
        #     axis_agg,
        #     origin,
        #     projection,
        #     anchor,
        #     action,
        #     pred_flow_screw,
        #     temp,
        #     trajectory,
        #     trajectory[0],
        #     self.max_flow_idx,
        # )
        # breakpoint()
        assert trajectory.shape == (11, 3)

        # Move the robot. For now, just move the translation.
        link_ = self.env.link_name_to_index[self.move_joints]
        start_imgs = self.env.begin_suction(self.max_flow_pt, link_, render=self.render)

        if mpc:
            trajectory = trajectory[:exec_step]
        # Execute trajectory
        inp_temp_ang = self.hi / np.pi * 180 if self.sem == "hinge" else self.hi
        if mpc:
            tel_imgs = self.env.move_gripper_traj(trajectory, render=self.render)
        else:
            tel_imgs, early_stop = self.move_gripper_traj_with_feedback(
                trajectory, target_angle=inp_temp_ang
            )

        if mpc:
            for _ in range(20):
                mpc_traj = self.mpc_step(
                    action.loc, goal_flowed, exec_step=exec_step, gt_ang=True
                )
                if mpc_traj is None:
                    return None
                exec_imgs, early_stop = self.move_gripper_traj_with_feedback(
                    mpc_traj, target_angle=inp_temp_ang
                )
                tel_imgs += exec_imgs
                if early_stop:
                    break
        print(f"Residual: {np.abs(self.get_curr_angle() - inp_temp_ang)}")
        # breakpoint()

        # Get joint angle
        self.res_angle = self.get_curr_angle()

        gt_pcd = None
        achieved_pcd = action.pos.cpu().numpy()

        self.results_act = record_actuation_result(
            self.results_act,
            anchor.obj_id[0],
            self.init_ang,
            self.res_angle,
            self.end_ang,
            self.sem,
            gt_pcd,
            achieved_pcd,
            pred_pose.cpu().numpy(),
            action.pos.cpu().numpy(),
        )

        return start_imgs + tel_imgs, self.results_act
