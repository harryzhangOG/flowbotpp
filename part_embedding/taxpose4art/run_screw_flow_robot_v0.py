import os
import pickle

import imageio
import numpy as np
import torch
import torch_geometric.loader as tgl
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from part_embedding.datasets.calc_art import compute_new_points
from part_embedding.datasets.pm.pm_raw import PMRawData
from part_embedding.taxpose4art.generate_art_training_data import transform_pcd
from part_embedding.taxpose4art.generate_art_training_data_flowbot import get_category
from part_embedding.taxpose4art.run_goalflow_robot import RobotEvalualtor
from part_embedding.taxpose4art.suction_env import PMSuctionEnv
from part_embedding.taxpose4art.train_goalflow import Model
from part_embedding.taxpose4art.train_screw_flow import Model as ScrewModel
from part_embedding.taxpose4art.train_utils import (
    create_flowbot_art_dataset,
    record_actuation_result,
)

"""
Naive FlowBot V2. No MPC.

Trust whatever we have (flow, axis, etc) and do an open loop trajectory.
"""


def rotate_on_the_fly(
    axis: torch.Tensor, origin: torch.Tensor, pts: np.ndarray, angle: float
):
    """
    Take as input tensor axis, origin, and pts

    Perform 3D rotation using the axis and angle.
    """
    r_mat = R.from_rotvec(axis.cpu().numpy() * angle / 180 * np.pi).as_matrix()
    rotated_pts = r_mat @ (
        pts.reshape((-1, 3)).T - origin.cpu().numpy().reshape(3, 1)
    ) + origin.cpu().numpy().reshape((3, 1))
    return rotated_pts.T


class RobotEvaluatorScrew(RobotEvalualtor):
    def __init__(
        self,
        obj_id,
        model,
        screw_model,
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

    def infer_screw_flow(self, action, anchor):
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

            pred_flow_screw = self.screwnet(
                action,
                anchor,
            )
            screw = pred_flow_screw[:500][:, 3:]
            screw = screw / screw.norm(dim=1).reshape(-1, 1)
            art_flow = pred_flow_screw[:500][:, :3]
            art_flow = art_flow / art_flow.norm(dim=1).reshape(-1, 1)

            axis_agg = torch.cross(art_flow, screw).mean(dim=0)
            axis_agg = axis_agg / axis_agg.norm()
            origin = pred_flow_screw[:500][:, 3:] + action.pos
            origin = origin.mean(dim=0)
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
            return pred_flow, origin, projection, pred_pose, pred_flow_screw, axis_agg

    def run_eval(self):

        self.get_joint_metadata()

        if self.sem == "slider":
            return None

        if self.offline:
            self.actuate_obj_in_env()
        else:
            self.randomize_obj_in_env()
        if self.offline:
            action, anchor = self.obs_offline()
        else:
            action, anchor = self.obs()

        if self.flowbot:
            action.loc = torch.Tensor(
                [self.end_ang / np.pi * 180 - self.init_ang]
            ).cuda()

        # Run inference
        (
            pred_flow,
            origin,
            projection,
            pred_pose,
            pred_flow_screw,
            axis_agg,
        ) = self.infer_screw_flow(action, anchor)

        # Add the robot to the env
        self.env.add_gripper_to_env()

        # Teleport the gripper to max change point (MAX "flow")
        # TODO: add grasping heuristic maybe and change to a better policy
        top_k = 10

        max_change_pos = np.linalg.norm(pred_flow, axis=1)
        max_flow_idx = np.argpartition(max_change_pos, -top_k)[-top_k:]
        anchor_pcd = anchor.pos.cpu().numpy()
        anchor_pcd_transform = anchor_pcd + pred_flow
        max_flow_pt = np.mean(anchor_pcd[max_flow_idx], axis=0)

        temp = rotate_on_the_fly(
            axis_agg, origin, action.pos.cpu().numpy(), action.loc.item()
        )

        """
        4 points A, B, C, D
        A is proj point on the axis
        B is max_flow_pt (contact)
        C is goal flow pt
        D is ambiguous rotation point

        Check if (AB x AC) . (AB x AD) > 0 if so, same direction
        """
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
        if np.dot(disp0, disp1) < 0:
            axis_agg = -axis_agg
            temp = rotate_on_the_fly(
                axis_agg, origin, action.pos.cpu().numpy(), action.loc.item()
            )
            print("Disambiguated the direction")
        trajectory = [max_flow_pt.reshape(-1, 3)]
        for interv in range(10):
            trajectory.append(
                rotate_on_the_fly(
                    axis_agg,
                    origin,
                    max_flow_pt.reshape(-1, 3),
                    action.loc.item() / 10 * (interv + 1),
                )
            )
        trajectory = np.array(trajectory).squeeze()

        assert trajectory.shape == (11, 3)

        """
        Debug
        """
        # import trimesh

        # points = axis_agg * torch.linspace(-1, 1, 100).reshape(100, 1).cuda() + origin
        # line = trimesh.points.PointCloud(points.cpu(), colors=(255, 0, 0))
        # act = trimesh.points.PointCloud(anchor.pos.cpu())
        # proj = trimesh.points.PointCloud(
        #     projection(action.pos).cpu().T, colors=(0, 255, 0)
        # )
        # add = trimesh.points.PointCloud(
        #     (pred_flow_screw[:500][:, 3:] + action.pos).cpu(), colors=(0, 0, 255)
        # )
        # rot_pts = trimesh.points.PointCloud(temp, colors=(255, 255, 20))
        # traj = trimesh.points.PointCloud(trajectory, colors=(255, 0, 255))
        # scene = trimesh.Scene([act, line, proj, add, rot_pts, traj])
        # scene.show()
        # breakpoint()
        """"""

        # Move the robot. For now, just move the translation.
        link_ = self.env.link_name_to_index[self.move_joints]
        start_imgs = self.env.begin_suction(max_flow_pt, link_, render=self.render)

        # Destination point is contact point transformed
        dest_pt = np.mean(anchor_pcd_transform[max_flow_idx], axis=0)

        tel_imgs = self.env.move_gripper_traj(trajectory, render=self.render)

        # Get joint angle
        # TODO: RN just use the non-zero joint bc joint-link are different
        target_joint = [
            x for x in self.pm_raw_data.obj.joints if x.child == self.move_joints
        ][0].name

        joint_angles = self.env.get_joint_angles()
        for k in joint_angles:
            if k == target_joint:
                if self.sem == "hinge":
                    res_angle = joint_angles[k] / np.pi * 180
                else:
                    res_angle = joint_angles[k]

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
            target_ja[-1] = res_angle - self.init_ang
        else:
            target_ja[-1] = (res_angle - self.init_ang) / 180 * np.pi
        achieved_tf = compute_new_points(
            action.pos.cpu().numpy(),
            self.env.T_world_base,
            chain,
            current_ja=current_ja,
            target_ja=target_ja,
            return_transform=True,
        )
        achieved_pcd = transform_pcd(action.pos.cpu().numpy(), achieved_tf)

        self.results_act = record_actuation_result(
            self.results_act,
            anchor.obj_id[0],
            self.init_ang,
            res_angle,
            self.end_ang,
            self.sem,
            gt_pcd,
            achieved_pcd,
            pred_pose.cpu().numpy(),
            action.pos.cpu().numpy(),
        )

        return start_imgs + tel_imgs, self.results_act


def log_results(norm_pcd_dist=False):
    def classwise(d):
        classdict = {}
        for obj_id, val in d.items():
            tmp = obj_id.split("_")[0] if "_" in obj_id else obj_id
            cat = get_category(tmp)
            if cat not in classdict:
                classdict[cat] = []
            if isinstance(val, np.float64):
                classdict[cat].append(val)
            else:
                classdict[cat].append(val.cpu().numpy())
        return {cat: np.mean(np.stack(ls)) for cat, ls in classdict.items()}

    # angle error after actuation
    ang_errs = {
        obj_id: np.mean(np.stack([d["ang_err"] for d in dlist]))
        for obj_id, dlist in results_act.items()
    }
    ang_norm_dist = {
        obj_id: np.mean(np.stack([d["ang_norm_dist"] for d in dlist]))
        for obj_id, dlist in results_act.items()
    }
    if norm_pcd_dist:
        pred_norm_errs = {
            obj_id: np.mean(np.stack([d["pred_norm_dist"] for d in dlist]))
            for obj_id, dlist in results_act.items()
        }
        achieved_norm_dist = {
            obj_id: np.mean(np.stack([d["achieved_norm_dist"] for d in dlist]))
            for obj_id, dlist in results_act.items()
        }
    ang_errs_summary = classwise(ang_errs)
    ang_norm_dist_summary = classwise(ang_norm_dist)
    if norm_pcd_dist:
        pred_norm_dist_summary = classwise(pred_norm_errs)
        achieved_norm_dist_summary = classwise(achieved_norm_dist)
    num = 0
    total = 0
    total_norm = 0
    total_pred = 0
    total_achieved = 0
    if mode == "unseen":
        cats = [
            "Bucket",
            "Safe",
            "Phone",
            "KitchenPot",
            "Box",
            "Table",
            "Dishwasher",
            "Oven",
            "WashingMachine",
            "Door",
        ]
    else:
        cats = [
            "Stapler",
            "TrashCan",
            "StorageFurniture",
            "Window",
            "Toilet",
            "Laptop",
            "Kettle",
            "Switch",
            "Refrigerator",
            "FoldingChair",
            "Microwave",
        ]

    if norm_pcd_dist:

        for cat in cats:
            if cat in ang_errs_summary:
                print(
                    f"{cat:<20}\ttheta error: {ang_errs_summary[cat]:.2f} \tnorm theta dist: {ang_norm_dist_summary[cat]:.2f} \tSVD norm theta dist: {pred_norm_dist_summary[cat]:.2f} \tachieved norm theta dist: {achieved_norm_dist_summary[cat]:.2f}"
                )
                num += 1
                total += ang_errs_summary[cat]
                total_norm += ang_norm_dist_summary[cat]
                total_achieved += achieved_norm_dist_summary[cat]
                total_pred += pred_norm_dist_summary[cat]
        print(
            f"\n\nAverage\topen error: {total/num:.2f} \tnorm dist: {total_norm/num:.2f} \tSVD norm theta dist:{total_pred/num:.2f} \tachieved norm theta dist:{total_achieved/num:.2f}"
        )
    else:
        for cat in cats:
            if cat in ang_errs_summary:
                print(
                    f"{cat:<20}\ttheta error: {ang_errs_summary[cat]:.2f} \tnorm theta dist: {ang_norm_dist_summary[cat]:.2f}"
                )
                num += 1
                total += ang_errs_summary[cat]
                total_norm += ang_norm_dist_summary[cat]
        print(
            f"\n\nAverage\topen error: {total/num:.2f} \tnorm dist: {total_norm/num:.2f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        help="checkpoint dir name",
    )
    parser.add_argument("--mode", type=str, help="test/unseen", default="test")
    parser.add_argument("--flowbot", action="store_true")
    args = parser.parse_args()
    mode = args.mode
    flowbot = args.flowbot
    flownet_model = Model()
    flownet_model.load_state_dict(
        torch.load(
            "/home/harry/discriminative_embeddings/part_embedding/taxpose4art/checkpoints/all_100_obj_tf-dark-thunder-28/weights_070.pt"
        )
    )
    screw_model = ScrewModel()
    screw_model.load_state_dict(
        torch.load(
            "/home/harry/discriminative_embeddings/part_embedding/taxpose4art/checkpoints/flow_screw-iconic-flower-4/weights_070.pt"
        )
    )
    flownet_model = flownet_model.cuda()
    screw_model = screw_model.cuda()

    save_dir = f"part_embedding/taxpose4art/rollout_res/screw"
    if not os.path.exists(save_dir):
        print("Creating results directory")
        os.makedirs(save_dir, exist_ok=True)

    results_act = {}

    # Create offline data
    dset_cat = "all"
    dset_num = "100"
    dset_name = f"{dset_cat}_{dset_num}_obj_tf"
    root = "/home/harry/partnet-mobility"
    n_repeat = 1
    n_proc = 50
    batch_size = 1

    train_dset, test_dset, unseen_dset = create_flowbot_art_dataset(
        dset_name, root, True, n_repeat, False, n_proc, True, False
    )  # Third one is process

    if mode == "train":
        train_loader = tgl.DataLoader(
            train_dset, batch_size=batch_size, shuffle=True, num_workers=0
        )

    test_loader = tgl.DataLoader(
        test_dset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    unseen_loader = tgl.DataLoader(
        unseen_dset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    flownet_model.eval()
    if mode == "test":
        split_name = "train_test"
        pbar = tqdm(test_loader)
    elif mode == "train":
        split_name = "train_train"
        pbar = tqdm(train_loader)
    else:
        split_name = "test_test"
        pbar = tqdm(unseen_loader)

    # iterate over offline dataset
    for action, anchor in pbar:
        # if get_category(anchor.obj_id[0].split("_")[0]) != "Oven":
        #     continue
        evaluator = RobotEvaluatorScrew(
            anchor.obj_id[0],
            flownet_model,
            screw_model,
            results_act,
            split_name,
            render=False,
            offline=True,
            action=action,
            anchor=anchor,
            flowbot=flowbot,
        )
        result = evaluator.run_eval()

        if result == None:
            evaluator.close_env()
            continue

        imgs, results_act = result

        evaluator.close_env()
        tmp = (
            anchor.obj_id[0].split("_")[0]
            if "_" in anchor.obj_id[0]
            else anchor.obj_id[0]
        )
        cat = get_category(tmp)
        save_dir1 = f"{save_dir}/{cat}"
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1, exist_ok=True)
        imageio.mimsave(f"{save_dir1}/{anchor.obj_id[0]}.gif", imgs, fps=5)
    log_results(False)
    breakpoint()
