import os
import pickle

import imageio
import numpy as np
import pybullet as p
import torch
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from part_embedding.datasets.calc_art import compute_new_points
from part_embedding.datasets.pm.pm_raw import PMRawData
from part_embedding.goal_inference.dataset_v2 import downsample_pcd_fps
from part_embedding.goal_inference.dset_utils import render_input_articulated
from part_embedding.taxpose4art.generate_art_training_data import (
    _compute_tf_from_flow,
    transform_pcd,
)
from part_embedding.taxpose4art.generate_art_training_data_flowbot import (
    get_category,
    get_sem,
)
from part_embedding.taxpose4art.suction_env import PMSuctionEnv
from part_embedding.taxpose4art.train_goalflow import Model
from part_embedding.taxpose4art.train_taxpose4art import Model as TAXModel
from part_embedding.taxpose4art.train_utils import (
    create_flowbot_art_dataset,
    record_actuation_result,
)


def trained_model(ckpt):

    model = TAXModel()
    model.load_state_dict(torch.load(ckpt))

    return model


class RobotEvalualtor:
    def __init__(
        self,
        obj_id,
        model,
        results_act,
        split_name,
        offline=False,
        action=None,
        anchor=None,
        flowbot=False,
        render=False,
    ):
        self.goalflownet = model
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

    def get_joint_metadata(self):
        tmp = self.obj_id.split("_")[0] if "_" in self.obj_id else self.obj_id
        for ent in self.full_sem_dset[get_category(tmp)]:
            if tmp in ent[0]:
                move_joints = ent[1]
                break
        self.sem = get_sem(ent)
        self.lo, self.hi = self.env.get_specific_joints_range(move_joints)  # in radians
        self.move_joints = move_joints

    def randomize_obj_in_env(self):
        # Randomize starting configuration
        if self.sem == "hinge":
            self.init_ang = (
                np.random.uniform(self.lo, self.hi) / np.pi * 180
            )  # in degrees
        elif self.sem == "slider":
            self.init_ang = np.random.uniform(self.lo, self.hi)
        self.env.set_specific_joints_angle(self.move_joints, self.init_ang, self.sem)

        rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.env.render(True)

        # Post-process and segmentation
        pc_seg_obj = np.ones_like(pc_seg) * -1
        for k, (body, link) in segmap.items():
            if body == self.env.obj_id:
                ixs = pc_seg == k
                pc_seg_obj[ixs] = link

        is_obj = pc_seg_obj != -1
        P_world = P_world[is_obj]

        # Randomize ending configuration
        if self.sem == "hinge":
            end = np.random.uniform(10, 70)  # in degrees
            while abs(end - self.init_ang) < 15:
                print("resampling")
                end = np.random.uniform(10, 70)  # in degrees
            self.end_ang = end
        elif self.sem == "slider":
            end = np.random.uniform(self.lo, self.hi)
            if self.hi > 0.7:
                while abs(end - self.init_ang) < 0.15:
                    print("resampling")
                    end = np.random.uniform(self.lo, self.hi)  # in degrees
            self.end_ang = end
        # Get relative angular difference
        ang_diff = self.end_ang - self.init_ang  # in degrees

        # Get transformation
        self.tf_gt = _compute_tf_from_flow(
            self.env,
            self.pm_raw_data,
            self.move_joints,
            P_world,
            pc_seg_obj,
            ang_diff,
            self.sem,
        )

    def actuate_obj_in_env(self):
        tmp_id = self.obj_id.split("_")[0]
        curr_data_entry = self.generated_metadata[get_category(tmp_id)][tmp_id][
            int(self.obj_id.split("_")[1])
        ]
        self.init_ang = curr_data_entry["start"]
        self.end_ang = curr_data_entry["end"]
        if self.flowbot:
            self.end_ang = self.hi
        self.env.set_specific_joints_angle(self.move_joints, self.init_ang, self.sem)

    def obs(self):
        # This is creating test env on the fly
        P_world, pc_seg, rgb, action_mask = render_input_articulated(
            self.env, self.move_joints
        )

        # Separate out the action and anchor points.
        P_action_world = P_world[action_mask]
        P_anchor_world = P_world[~action_mask]

        P_action_world = torch.from_numpy(P_action_world)
        P_anchor_world = torch.from_numpy(P_anchor_world)

        action_pts_num = 500
        # Now, downsample
        action_ixs = downsample_pcd_fps(P_action_world, n=action_pts_num)
        anchor_ixs = downsample_pcd_fps(P_anchor_world, n=2000 - action_pts_num)

        # Rebuild the world
        P_action_world = P_action_world[action_ixs]
        while len(P_action_world) < action_pts_num:
            P_action_world = torch.cat(
                [
                    P_action_world,
                    P_action_world[: (action_pts_num - len(P_action_world))],
                ]
            )
        P_action_world = P_action_world[:500]
        P_anchor_world = P_anchor_world[anchor_ixs]
        P_world = np.concatenate([P_action_world, P_anchor_world], axis=0)

        # # Compute the transform from action object to goal.
        # t_action_anchor = self.tf_gt[:-1, -1]
        # t_action_anchor = torch.from_numpy(t_action_anchor).float().unsqueeze(0)
        # R_action_anchor = self.tf_gt[:-1, :-1]
        # R_action_anchor = torch.from_numpy(R_action_anchor).float().unsqueeze(0)

        # Assemble the data.
        action_data = tgd.Data(
            pos=P_action_world.float(),
            # t_action_anchor=t_action_anchor.float(),
            # R_action_anchor=R_action_anchor.float(),
            loc=self.end_ang,
        )
        anchor_data = tgd.Data(
            obj_id=self.obj_id,
            pos=torch.from_numpy(P_world).float(),
        )
        device = "cuda:0"

        return (
            Batch.from_data_list([action_data]).to(device),
            Batch.from_data_list([anchor_data]).to(device),
        )

    def obs_offline(self):
        device = "cuda:0"
        return self.action.to(device), self.anchor.to(device)

    def run_eval(self):

        self.get_joint_metadata()
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
        with torch.no_grad():
            if isinstance(self.goalflownet, Model):
                pred_flow = self.goalflownet(
                    action,
                    anchor,
                )
                R_pred, t_pred = self.goalflownet.svd(
                    action.pos, pred_flow, anchor.flow
                )
                pred_pose = (
                    (
                        torch.bmm(
                            action.pos.reshape(-1, 500, 3), R_pred.transpose(-1, -2)
                        )
                        + t_pred
                    )
                    .reshape(-1, 3)
                    .cuda()
                )

                pred_flow = pred_flow.cpu().numpy()
                pred_flow[500:] = 0
            else:
                R_pred, t_pred, aws, _, _, _ = self.goalflownet(
                    action,
                    anchor,
                )
                tf_pred = np.eye(4)
                tf_pred[:3, :3] = R_pred.cpu().numpy()[0]
                tf_pred[:3, -1] = t_pred.cpu().numpy()[0]
                pred_pose = (
                    (
                        torch.bmm(
                            action.pos.reshape(-1, 500, 3), R_pred.transpose(-1, -2)
                        )
                        + t_pred
                    )
                    .reshape(-1, 3)
                    .cuda()
                )
                pred_flow = (pred_pose - action.pos).cpu().numpy()
                pred_flow = np.vstack([pred_flow, np.zeros((1500, 3))])

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

        """
        DEBUG
        """
        # import trimesh

        # pcd1 = trimesh.points.PointCloud(action.pos.cpu().numpy())
        # pcd2 = trimesh.points.PointCloud(action.pos.cpu().numpy() + pred_flow[:500])
        # pcd3 = trimesh.points.PointCloud(anchor.pos.cpu().numpy()[500:])
        # scene = trimesh.Scene([pcd1, pcd2, pcd3])
        # pcd1 = trimesh.points.PointCloud(action.pos.cpu().numpy(), colors=(255, 0, 0))
        # pcd2 = trimesh.points.PointCloud(
        #     action.pos.cpu().numpy() + pred_flow[:500], colors=(0, 255, 0)
        # )
        # scene = trimesh.Scene([pcd1, pcd2, pcd3])
        # scene.show()
        """
        DEBUG
        """

        # Move the robot. For now, just move the translation.
        link_ = self.env.link_name_to_index[self.move_joints]
        start_imgs = self.env.begin_suction(max_flow_pt, link_, render=self.render)

        # Destination point is contact point transformed
        dest_pt = np.mean(anchor_pcd_transform[max_flow_idx], axis=0)

        tel_imgs = self.env.move_gripper_vel_to(dest_pt, render=self.render)

        # Get joint angle
        # TODO: RN just use the non-zero joint bc joint-link are different
        joint_angles = self.env.get_joint_angles()
        for k in joint_angles:
            if joint_angles[k] != 0:
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

        # import trimesh

        # pcd1 = trimesh.points.PointCloud(gt_pcd)
        # pcd2 = trimesh.points.PointCloud(anchor.pos.cpu().numpy(), colors=(255, 0, 0))
        # pcd3 = trimesh.points.PointCloud(achieved_pcd, colors=(0, 255, 0))
        # pcd4 = trimesh.points.PointCloud(pred_pose.cpu().numpy(), colors=(0, 0, 255))
        # scene = trimesh.Scene([pcd1, pcd2, pcd3, pcd4])
        # scene.show()

        return start_imgs + tel_imgs, self.results_act

    def close_env(self):
        p.disconnect()


def log_results():
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

    for cat in cats:
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
    ckpt = args.ckpt
    mode = args.mode
    flowbot = args.flowbot
    ckpt_dir = f"part_embedding/taxpose4art/checkpoints/{ckpt}/weights_030.pt"
    flownet_model = trained_model(ckpt_dir).cuda()

    save_dir = f"part_embedding/taxpose4art/rollout_res/{ckpt}"
    if not os.path.exists(save_dir):
        print("Creating results directory")
        os.makedirs(save_dir, exist_ok=True)

    results_act = {}

    # Create offline data
    dset_cat = ckpt.split("_")[0]
    dset_num = ckpt.split("_")[1]
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
        # if get_category(anchor.obj_id[0].split("_")[0]) != "Kettle":
        #     continue
        evaluator = RobotEvalualtor(
            anchor.obj_id[0],
            flownet_model,
            results_act,
            split_name,
            render=False,
            offline=True,
            action=action,
            anchor=anchor,
            flowbot=flowbot,
        )
        imgs, results_act = evaluator.run_eval()
        evaluator.close_env()
        imageio.mimsave(f"{save_dir}/{anchor.obj_id[0]}.gif", imgs, fps=2)

    log_results()
