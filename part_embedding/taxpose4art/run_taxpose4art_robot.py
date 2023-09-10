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

import part_embedding.goal_inference.create_pm_goal_dataset as pgc
from part_embedding.datasets.pm.pm_raw import PMRawData
from part_embedding.goal_inference.dataset import SEM_CLASS_DSET_PATH
from part_embedding.goal_inference.dataset_v2 import downsample_pcd_fps
from part_embedding.goal_inference.dset_utils import render_input_articulated
from part_embedding.taxpose4art.dataset import CATEGORIES
from part_embedding.taxpose4art.eval_taxpose4art import t_err, theta_err, trained_model
from part_embedding.taxpose4art.generate_art_training_data import (
    _compute_tf_from_flow,
    transform_pcd,
)
from part_embedding.taxpose4art.suction_env import PMSuctionEnv
from part_embedding.taxpose4art.train_utils import create_art_dataset


class RobotEvalualtor:
    def __init__(
        self,
        obj_id,
        model,
        results,
        results_act,
        dset_name,
        offline=False,
        action=None,
        anchor=None,
        render=False,
    ):
        self.taxpose = model
        self.offline = offline
        if self.offline:
            self.anchor = anchor
            self.action = action
        self.obj_id = obj_id
        self.generated_metadata = pickle.load(
            open(f"part_embedding/taxpose4art/training_data/{dset_name}.pkl", "rb")
        )
        tmp = obj_id.split("_")[0] if "_" in obj_id else obj_id
        self.env = PMSuctionEnv(
            tmp,
            os.path.expanduser("~/partnet-mobility/raw"),
            camera_pos=[-2.5, 0, 2.5],
            gui=False,
        )
        self.object_dict = pgc.all_objs[CATEGORIES[tmp].lower()]
        self.partsem = self.object_dict[f"{tmp}_0"]["partsem"]
        self.part_ind = self.object_dict[f"{tmp}_0"]["ind"]
        self.full_sem_dset = pickle.load(open(SEM_CLASS_DSET_PATH, "rb"))
        self.block_dset = pickle.load(
            open(
                os.path.expanduser(
                    "~/discriminative_embeddings/goal_inf_dset/all_block_dset_multi.pkl"
                ),
                "rb",
            )
        )
        self.render = render
        self.results = results
        self.results_act = results_act
        self.pm_raw_data = PMRawData(
            os.path.join(os.path.expanduser("~/partnet-mobility/raw"), tmp)
        )

    def get_joint_metadata(self):
        tmp = self.obj_id.split("_")[0] if "_" in self.obj_id else self.obj_id
        for mode in self.full_sem_dset:
            if self.partsem in self.full_sem_dset[mode]:
                if tmp in self.full_sem_dset[mode][self.partsem]:
                    move_joints = self.full_sem_dset[mode][self.partsem][tmp]

        self.lo, self.hi = self.env.get_specific_joints_range(
            move_joints[self.part_ind]
        )  # in radians
        self.move_joints = move_joints

    def randomize_obj_in_env(self):
        # Randomize starting configuration
        self.init_ang = np.random.uniform(self.lo, self.hi) / np.pi * 180  # in degrees
        self.env.set_specific_joints_angle(
            self.move_joints[self.part_ind], self.init_ang
        )

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
        self.end_ang = np.random.uniform(10, 70)  # in degrees
        # Get relative angular difference
        ang_diff = self.end_ang - self.init_ang  # in degrees

        # Get transformation
        self.tf_gt = _compute_tf_from_flow(
            self.env,
            self.pm_raw_data,
            self.move_joints[self.part_ind],
            P_world,
            pc_seg_obj,
            ang_diff,
        )

    def actuate_obj_in_env(self):
        tmp_id = self.obj_id.split("_")[0]
        curr_data_entry = self.generated_metadata[CATEGORIES[tmp_id]][tmp_id][
            int(self.obj_id.split("_")[1])
        ]
        self.init_ang = curr_data_entry["start"]
        self.end_ang = curr_data_entry["end"]
        self.env.set_specific_joints_angle(
            self.move_joints[self.part_ind], self.init_ang
        )

    def obs(self):
        # This is creating test env on the fly
        P_world, pc_seg, rgb, action_mask = render_input_articulated(
            self.env, self.move_joints[self.part_ind]
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
        if len(P_action_world) < action_pts_num:
            P_action_world = torch.cat(
                [
                    P_action_world,
                    P_action_world[: (action_pts_num - len(P_action_world))],
                ]
            )
        P_anchor_world = P_anchor_world[anchor_ixs]
        P_world = np.concatenate([P_action_world, P_anchor_world], axis=0)

        # Compute the transform from action object to goal.
        t_action_anchor = self.tf_gt[:-1, -1]
        t_action_anchor = torch.from_numpy(t_action_anchor).float().unsqueeze(0)
        R_action_anchor = self.tf_gt[:-1, :-1]
        R_action_anchor = torch.from_numpy(R_action_anchor).float().unsqueeze(0)

        # Assemble the data.
        action_data = tgd.Data(
            pos=P_action_world.float(),
            t_action_anchor=t_action_anchor.float(),
            R_action_anchor=R_action_anchor.float(),
            loc=self.end_ang,
        )
        anchor_data = tgd.Data(
            obj_id=self.obj_id,
            pos=P_anchor_world.float(),
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
        R_gt = action.R_action_anchor.reshape(-1, 3, 3)
        t_gt = action.t_action_anchor

        with torch.no_grad():

            R_pred, t_pred, aws, _, _, _ = self.taxpose(
                action,
                anchor,
            )
            tf_pred = np.eye(4)
            tf_pred[:3, :3] = R_pred.cpu().numpy()[0]
            tf_pred[:3, -1] = t_pred.cpu().numpy()[0]

        self.results = record_pred_result(
            self.results, anchor.obj_id[0], R_pred, R_gt, t_pred, t_gt
        )

        # Add the robot to the env
        self.env.add_gripper_to_env()

        # Teleport the gripper to max change point (MAX "flow")
        # TODO: add grasping heuristic maybe and change to a better policy
        top_k = 1

        action_pcd = action.pos.cpu().numpy()
        action_pcd_transform = transform_pcd(action_pcd, tf_pred)
        max_change_pos = np.linalg.norm(action_pcd_transform - action_pcd, axis=1)
        max_flow_idx = np.argpartition(max_change_pos, -top_k)[-top_k:]
        max_flow_pt = np.mean(action_pcd[max_flow_idx], axis=0)

        # Move the robot. For now, just move the translation.
        link_ = self.env.link_name_to_index[self.move_joints[self.part_ind]]
        start_imgs = self.env.begin_suction(max_flow_pt, link_, render=self.render)

        # Destination point is contact point transformed
        dest_pt = np.mean(action_pcd_transform[max_flow_idx], axis=0)

        tel_imgs = self.env.move_gripper_vel_to(dest_pt, render=self.render)

        # Get joint angle
        # TODO: RN just use the non-zero joint bc joint-link are different
        joint_angles = self.env.get_joint_angles()
        for k in joint_angles:
            if joint_angles[k] != 0:
                res_angle = joint_angles[k] / np.pi * 180

        self.results_act = record_actuation_result(
            self.results_act, anchor.obj_id[0], self.init_ang, res_angle, self.end_ang
        )
        return start_imgs + tel_imgs, self.results, self.results_act

    def close_env(self):
        p.disconnect()


def record_pred_result(results, obj_id, R_pred, R_gt, t_pred, t_gt):
    if obj_id not in results:
        results[obj_id] = []
    results[obj_id].append(
        {
            "theta_err": theta_err(R_pred.cuda(), R_gt.cuda()),
            "t_err": t_err(t_pred.cuda(), t_gt.cuda()),
            "t_norm": torch.min(
                torch.tensor(1).cuda(),
                t_err(t_pred.cuda(), t_gt.cuda())
                / t_gt.cuda().norm(dim=-1).squeeze().item(),
            ),
            "t_mag": t_gt.cuda().norm(dim=-1).squeeze().item(),
        }
    )
    return results


def record_actuation_result(results, obj_id, start_ang, res_ang, end_ang):
    if obj_id not in results:
        results[obj_id] = []
    results[obj_id].append(
        {
            "ang_err": np.abs(np.clip(res_ang, 10, 70) - end_ang),
            "ang_norm_dist": min(
                1, np.abs(res_ang - end_ang) / np.abs(end_ang - start_ang)
            ),
        }
    )
    return results


def log_results():
    # prediction error
    th_errs = {
        obj_id: torch.mean(torch.stack([d["theta_err"] for d in dlist]))
        for obj_id, dlist in results.items()
    }

    t_errs = {
        obj_id: torch.mean(torch.stack([d["t_err"] for d in dlist]))
        for obj_id, dlist in results.items()
    }

    t_norms = {
        obj_id: torch.mean(torch.stack([d["t_norm"] for d in dlist]))
        for obj_id, dlist in results.items()
    }

    def classwise(d):
        classdict = {}
        for obj_id, val in d.items():
            tmp = obj_id.split("_")[0] if "_" in obj_id else obj_id
            cat = CATEGORIES[tmp]
            if cat not in classdict:
                classdict[cat] = []
            if isinstance(val, np.float64):
                classdict[cat].append(val)
            else:
                classdict[cat].append(val.cpu().numpy())
        return {cat: np.mean(np.stack(ls)) for cat, ls in classdict.items()}

    th_errs_summary = classwise(th_errs)
    t_errs_summary = classwise(t_errs)
    t_norms_summary = classwise(t_norms)
    print("\n\n")
    for cat in th_errs_summary.keys():
        print(
            f"{cat:<15}\ttheta: {th_errs_summary[cat]:.2f} \tt_err: {t_errs_summary[cat]:.2f}\tt_normalized: {t_norms_summary[cat]:.2f}"
        )
    # angle error after actuation
    ang_errs = {
        obj_id: np.mean(np.stack([d["ang_err"] for d in dlist]))
        for obj_id, dlist in results_act.items()
    }
    ang_norm_dist = {
        obj_id: np.mean(np.stack([d["ang_norm_dist"] for d in dlist]))
        for obj_id, dlist in results_act.items()
    }
    ang_errs_summary = classwise(ang_errs)
    ang_norm_dist_summary = classwise(ang_norm_dist)
    for cat in th_errs_summary.keys():
        print(
            f"{cat:<15}\tangle error: {ang_errs_summary[cat]:.2f} \tnormalized distance: {ang_norm_dist_summary[cat]:.2f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        help="checkpoint dir name",
    )
    parser.add_argument("--mode", type=str, help="train/test", default="test")
    args = parser.parse_args()
    ckpt = args.ckpt
    mode = args.mode
    ckpt_dir = f"part_embedding/taxpose4art/checkpoints/{ckpt}/weights_050.pt"
    taxpose_model = trained_model(ckpt_dir).cuda()

    save_dir = f"part_embedding/taxpose4art/rollout_res/{ckpt}"
    if not os.path.exists(save_dir):
        print("Creating results directory")
        os.makedirs(save_dir, exist_ok=True)

    results = {}
    results_act = {}

    # Create offline data
    dset_cat = ckpt.split("_")[0]
    dset_num = ckpt.split("_")[1]
    dset_name = f"{dset_cat}_{dset_num}_obj_tf"
    root = "/home/harry/partnet-mobility"
    n_repeat = 1
    n_proc = 50
    batch_size = 1

    train_dset, test_dset = create_art_dataset(
        dset_name, root, True, n_repeat, False, n_proc, True, False
    )  # Third one is process

    train_loader = tgl.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = tgl.DataLoader(
        test_dset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    taxpose_model.eval()
    if mode == "train":
        pbar = tqdm(train_loader)
    else:
        pbar = tqdm(test_loader)

    # iterate over offline dataset
    for action, anchor in pbar:
        evaluator = RobotEvalualtor(
            anchor.obj_id[0],
            taxpose_model,
            results,
            results_act,
            dset_name,
            render=False,
            offline=True,
            action=action,
            anchor=anchor,
        )
        imgs, results, results_act = evaluator.run_eval()
        evaluator.close_env()
        imageio.mimsave(f"{save_dir}/{anchor.obj_id[0]}.gif", imgs, fps=2)

    log_results()
