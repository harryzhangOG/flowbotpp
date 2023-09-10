import csv
import json
import os
import pickle

import numpy as np
import pybullet as p
from tqdm import tqdm

from part_embedding.datasets.calc_art import compute_new_points
from part_embedding.datasets.pm.pm_raw import PMRawData
from part_embedding.envs.render_sim import PMRenderEnv


def get_partnet_ids(category="all"):
    split_file = json.load(
        open(
            os.path.expanduser(
                "~/discriminative_embeddings/goalcond-pm-objs-split.json"
            )
        )
    )
    split_file = split_file["train"]

    if category != "all":
        return {category.capitalize(): split_file[category.capitalize()]}
    else:
        return split_file


def _compute_flow(sim, pm_raw_data, link_name, P_world, pc_seg):

    flow = np.zeros_like(P_world)

    linkname_to_id = {
        p.getBodyInfo(sim.obj_id, physicsClientId=sim.client_id)[0].decode("UTF-8"): -1
    }
    for _id in range(p.getNumJoints(sim.obj_id, physicsClientId=sim.client_id)):
        _name = p.getJointInfo(sim.obj_id, _id, physicsClientId=sim.client_id)[
            12
        ].decode("UTF-8")
        linkname_to_id[_name] = _id

    try:
        link_id = linkname_to_id[link_name]
    except:
        breakpoint()
    link_ixs = pc_seg == link_id
    filtered_pc = P_world[link_ixs]

    chain = pm_raw_data.obj.get_chain(link_name)
    current_ja = np.zeros(len(chain))
    target_ja = np.zeros(len(chain))
    target_ja[-1] = 0.1

    filtered_new_pc = compute_new_points(
        filtered_pc, sim.T_world_base, chain, current_ja=current_ja, target_ja=target_ja
    )

    part_flow = filtered_new_pc - filtered_pc
    flow[link_ixs] = part_flow
    return flow


def _compute_tf_from_flow(sim, pm_raw_data, link_name, P_world, pc_seg, ang_diff, sem):

    flow = np.zeros_like(P_world)

    linkname_to_id = {
        p.getBodyInfo(sim.obj_id, physicsClientId=sim.client_id)[0].decode("UTF-8"): -1
    }
    for _id in range(p.getNumJoints(sim.obj_id, physicsClientId=sim.client_id)):
        _name = p.getJointInfo(sim.obj_id, _id, physicsClientId=sim.client_id)[
            12
        ].decode("UTF-8")
        linkname_to_id[_name] = _id

    try:
        link_id = linkname_to_id[link_name]
    except:
        breakpoint()
    link_ixs = pc_seg == link_id
    filtered_pc = P_world[link_ixs]

    chain = pm_raw_data.obj.get_chain(link_name)
    current_ja = np.zeros(len(chain))
    target_ja = np.zeros(len(chain))
    if sem == "hinge":
        target_ja[-1] = ang_diff / 180 * np.pi
    else:
        target_ja[-1] = ang_diff

    tf = compute_new_points(
        filtered_pc,
        sim.T_world_base,
        chain,
        current_ja=current_ja,
        target_ja=target_ja,
        return_transform=True,
    )

    return tf


def transform_pcd(P_world_pts, T):
    N = len(P_world_pts)
    Ph_world_pts = np.concatenate([P_world_pts, np.ones((N, 1))], axis=1)
    Ph_world_ptsnew = (T @ Ph_world_pts.T).T
    assert Ph_world_ptsnew.shape == (N, 4)
    P_world_ptsnew: np.ndarray = Ph_world_ptsnew[:, :3]
    return P_world_ptsnew


def get_sem(oid, move_joint):
    sem_file = csv.reader(
        open(os.path.expanduser(f"~/partnet-mobility/raw/{oid}/semantics.txt")),
        delimiter=" ",
    )
    for line in sem_file:
        if move_joint in line:
            return line[1]


def generate_transform(cat, obj_id, total_num):
    """
    Generate transform for the given obj_id for total_num times.
    The object part to articulate is determined by link_name
    """
    result = []

    # Create a rendering sim
    sim = PMRenderEnv(
        obj_id,
        os.path.expanduser("~/partnet-mobility/raw"),
        camera_pos=[-2, 0, 2],
        gui=False,
    )
    # Get part semantic label
    part_sem = block_dset[cat][f"{obj_id}_0"]["partsem"]
    # Get link index
    part_ind = block_dset[cat][f"{obj_id}_0"]["ind"]

    for mode in full_sem_dset:
        if part_sem in full_sem_dset[mode]:
            if obj_id in full_sem_dset[mode][part_sem]:
                move_joints = full_sem_dset[mode][part_sem][obj_id]

    lo, hi = sim.get_specific_joints_range(move_joints[part_ind])  # in radians

    pm_raw_data = PMRawData(
        os.path.join(os.path.expanduser("~/partnet-mobility/raw"), obj_id)
    )

    sem = get_sem(obj_id, move_joints[part_ind])

    for _ in tqdm(range(total_num)):

        # Randomize starting configuration
        if sem == "hinge":
            init_ang = np.random.uniform(lo, hi) / np.pi * 180  # in degrees
        else:
            init_ang = np.random.uniform(lo, hi)

        sim.set_specific_joints_angle(move_joints[part_ind], init_ang)

        rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = sim.render(True)

        # Post-process and segmentation
        pc_seg_obj = np.ones_like(pc_seg) * -1
        for k, (body, link) in segmap.items():
            if body == sim.obj_id:
                ixs = pc_seg == k
                pc_seg_obj[ixs] = link

        is_obj = pc_seg_obj != -1
        P_world = P_world[is_obj]

        # Randomize ending configuration
        end_ang = hi if sem != "hinge" else hi / np.pi * 180  # in degrees
        # Get relative angular difference
        ang_diff = end_ang - init_ang  # in degrees

        # Get transformation
        tf = _compute_tf_from_flow(
            sim, pm_raw_data, move_joints[part_ind], P_world, pc_seg_obj, ang_diff, sem
        )
        link_ = sim.link_name_to_index[move_joints[part_ind]]

        flow2tf_res = transform_pcd(P_world[pc_seg_obj == link_], tf)

        # Record meta result
        curr_meta_res = {"transformation": tf, "start": init_ang, "end": end_ang}

        # Append to dataset
        result.append(curr_meta_res)

    p.disconnect()

    return result


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", type=str)
    parser.add_argument("--num", type=str)
    args = parser.parse_args()

    # Which category we want to generate
    category = args.cat
    # How many data points per object id?
    total_num = args.num

    full_sem_dset = pickle.load(
        open(
            os.path.expanduser(
                "~/discriminative_embeddings/goal_inf_dset/sem_class_transfer_dset_more.pkl"
            ),
            "rb",
        )
    )

    block_dset = pickle.load(
        open(
            os.path.expanduser(
                "~/discriminative_embeddings/goal_inf_dset/all_block_dset_multi.pkl"
            ),
            "rb",
        )
    )

    part_ids_dict = get_partnet_ids(category)

    generated_dataset = {}
    for c in part_ids_dict:
        if c not in ["Chair"]:
            cat_dataset = {}
            all_cat_obj_ids = part_ids_dict[c]["train"] + part_ids_dict[c]["test"]
            for obj_id in all_cat_obj_ids:
                curr_datapoint = generate_transform(c.lower(), obj_id, int(total_num))
                cat_dataset[obj_id] = curr_datapoint
            generated_dataset[c] = cat_dataset

    save_dir = os.path.expanduser(
        "~/discriminative_embeddings/part_embedding/flowtron/dataset/training_data"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the generated dataset
    pickle.dump(
        generated_dataset, open(f"{save_dir}/{category}_{total_num}_obj_tf.pkl", "wb")
    )

    print("Data generated")
