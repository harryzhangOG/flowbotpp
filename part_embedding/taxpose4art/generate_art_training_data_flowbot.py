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


def get_category(oid):
    meta_file = json.load(
        open(os.path.expanduser(f"~/partnet-mobility/raw/{oid}/meta.json"))
    )
    return meta_file["model_cat"]


def get_sem(oid):
    sem_file = csv.reader(
        open(
            os.path.expanduser(
                f"~/partnet-mobility/raw/{oid[0].split('_')[0]}/semantics.txt"
            )
        ),
        delimiter=" ",
    )
    for line in sem_file:
        if oid[1] in line:
            return line[1]


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


def generate_transform(oid):
    # Create a rendering sim
    obj_id = oid[0].split("_")[0]
    sim = PMRenderEnv(
        obj_id,
        os.path.expanduser("~/partnet-mobility/raw"),
        camera_pos=[-2, 0, 2],
        gui=False,
    )
    target_joint = oid[1]
    sem = get_sem(oid)

    lo, hi = sim.get_specific_joints_range(target_joint)

    pm_raw_data = PMRawData(
        os.path.join(os.path.expanduser("~/partnet-mobility/raw"), obj_id)
    )

    # Randomize starting configuration
    if sem == "hinge":
        init = np.random.uniform(lo, hi) / np.pi * 180  # in degrees
    elif sem == "slider":
        init = np.random.uniform(lo, hi)
    sim.set_specific_joints_angle(target_joint, init, sem)

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
    if sem == "hinge":
        end = np.random.uniform(10, 70)  # in degrees
        while abs(end - init) < 10 or abs(end - init) > 70:
            print("resampling")
            end = np.random.uniform(lo, hi) / np.pi * 180  # in degrees
    elif sem == "slider":
        end = np.random.uniform(lo, hi)
        if hi > 0.7:
            while abs(end - init) < 0.1 or abs(end - init) > 0.7:
                print("resampling")
                end = np.random.uniform(lo, hi)  # in degrees

    # Get relative angular difference
    ang_diff = end - init  # in degrees

    # Get transformation
    tf = _compute_tf_from_flow(
        sim, pm_raw_data, target_joint, P_world, pc_seg_obj, ang_diff, sem
    )

    flow2tf_res = transform_pcd(P_world[pc_seg_obj == 1], tf)

    # Record meta result
    curr_meta_res = {"transformation": tf, "start": init, "end": end}

    p.disconnect()

    return curr_meta_res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sem",
        type=str,
        help="checkpoint dir name",
    )
    args = parser.parse_args()
    sem_label = args.sem

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

    train_train = pickle.load(
        open(
            os.path.expanduser(
                "~/discriminative_embeddings/part_embedding/taxpose4art/flowbot_split/train_train_aug.pkl"
            ),
            "rb",
        )
    )
    train_test = pickle.load(
        open(
            os.path.expanduser(
                "~/discriminative_embeddings/part_embedding/taxpose4art/flowbot_split/train_test.pkl"
            ),
            "rb",
        )
    )
    test_test = pickle.load(
        open(
            os.path.expanduser(
                "~/discriminative_embeddings/part_embedding/taxpose4art/flowbot_split/test_test.pkl"
            ),
            "rb",
        )
    )
    save_dir = os.path.expanduser(
        "~/discriminative_embeddings/part_embedding/taxpose4art/flowbot_data"
    )
    generated_dataset = {}
    for c in train_train:
        cat_dataset = {}
        all_cat_obj_ids = train_train[c]
        # if c != "StorageFurniture":
        #     continue
        print(c)
        for oid in tqdm(all_cat_obj_ids):
            # if get_sem(oid) != sem_label:
            #     continue
            obj_id = oid[0].split("_")[0]
            if obj_id not in cat_dataset:
                cat_dataset[obj_id] = []
            curr_datapoint = generate_transform(oid)
            cat_dataset[obj_id].append(curr_datapoint)
        generated_dataset[c] = cat_dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pickle.dump(generated_dataset, open(f"{save_dir}/train_train_aug.pkl", "wb"))

    # Save the generated dataset

    # generated_dataset = {}
    # for c in train_test:
    #     # if c not in ["Stapler", "Microwave"]:
    #     #     continue
    #     print(c)
    #     cat_dataset = {}
    #     all_cat_obj_ids = train_test[c]
    #     for oid in tqdm(all_cat_obj_ids):
    #         # if get_sem(oid) != sem_label:
    #         #     continue
    #         obj_id = oid[0].split("_")[0]
    #         if obj_id not in cat_dataset:
    #             cat_dataset[obj_id] = []
    #         curr_datapoint = generate_transform(oid)
    #         cat_dataset[obj_id].append(curr_datapoint)
    #     generated_dataset[c] = cat_dataset

    # save_dir = os.path.expanduser(
    #     "~/discriminative_embeddings/part_embedding/taxpose4art/flowbot_data"
    # )
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # # Save the generated dataset
    # pickle.dump(generated_dataset, open(f"{save_dir}/train_test.pkl", "wb"))

    # generated_dataset = {}
    # # generated_dataset = pickle.load(open(f"{save_dir}/test_test.pkl", "rb"))

    # for c in test_test:
    #     # if c not in ["Door"]:
    #     #     continue
    #     print(c)
    #     cat_dataset = {}
    #     all_cat_obj_ids = test_test[c]
    #     for oid in tqdm(all_cat_obj_ids):
    #         # if get_sem(oid) != sem_label:
    #         #     continue
    #         obj_id = oid[0].split("_")[0]
    #         if obj_id not in cat_dataset:
    #             cat_dataset[obj_id] = []
    #         curr_datapoint = generate_transform(oid)
    #         cat_dataset[obj_id].append(curr_datapoint)
    #     generated_dataset[c] = cat_dataset

    # save_dir = os.path.expanduser(
    #     "~/discriminative_embeddings/part_embedding/taxpose4art/flowbot_data"
    # )
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # # Save the generated dataset
    # pickle.dump(generated_dataset, open(f"{save_dir}/test_test.pkl", "wb"))

    print("Data generated")
