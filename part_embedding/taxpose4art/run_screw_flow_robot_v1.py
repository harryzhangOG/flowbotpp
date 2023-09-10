import os

import imageio
import numpy as np
import torch
import torch_geometric.loader as tgl
from tqdm import tqdm

from part_embedding.flowtron.models.flowbotv2 import ArtClassModel
from part_embedding.taxpose4art.evaluator.screw_flow_robot_v1 import RobotEvaluatorMPC
from part_embedding.taxpose4art.generate_art_training_data_flowbot import get_category
from part_embedding.taxpose4art.train_goalflow import Model
from part_embedding.taxpose4art.train_screw_flow import Model as ScrewModel
from part_embedding.taxpose4art.train_utils import create_flowbot_art_dataset


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
    parser.add_argument("--step", type=int)
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    mode = args.mode
    flowbot = args.flowbot
    step = args.step
    outdir = args.dir
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
    sem_class_model = ArtClassModel()
    sem_class_model.load_state_dict(
        torch.load(
            "/home/harry/discriminative_embeddings/part_embedding/flowtron/checkpoints/all_100_obj_tf-sweet-fire-1/weights_070.pt"
        )
    )
    flownet_model = flownet_model.cuda()
    screw_model = screw_model.cuda()
    sem_class_model = sem_class_model.cuda()

    save_dir = f"part_embedding/taxpose4art/rollout_res/{outdir}"
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
        # if get_category(anchor.obj_id[0].split("_")[0]) != "Kettle":
        #     continue
        evaluator = RobotEvaluatorMPC(
            anchor.obj_id[0],
            flownet_model,
            screw_model,
            sem_class_model,
            results_act,
            split_name,
            render=False,
            offline=True,
            action=action,
            anchor=anchor,
            flowbot=flowbot,
        )
        result = evaluator.run_eval(exec_step=step, mpc=True, gs=True)
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

    """
    CUDA_VISIBLE_DEVICES=1 python -m part_embedding.taxpose4art.run_screw_flow_robot_v1 --mode test --flowbot --dir screw-v2-step1 --step 2
    """
