import json
import math
import pickle

import numpy as np
import torch
from plotly.subplots import make_subplots

import part_embedding.visualization.plots as pvp
from part_embedding.flowtron.dataset.classifier_dataset import ClassifierDataset
from part_embedding.flowtron.dataset.classifier_dataset_v2 import SegmenterDataset
from part_embedding.flowtron.dataset.full_sys_dataset import FullSysDataset
from part_embedding.flowtron.dataset.weighted_dataset import WeightedDataset
from part_embedding.losses.formnet_loss import artflownet_loss
from part_embedding.taxpose4art.dataset import TAXPoseDataset
from part_embedding.taxpose4art.flowbot_dataset import (
    TAXPoseDataset as FBTAXPoseDataset,
)
from part_embedding.taxpose4art.screw_dataset import ScrewDataset


def create_art_dataset(
    dset_name,
    root,
    process,
    n_repeat,
    randomize_camera,
    n_proc,
    even_downsample,
    rotate_anchor,
):
    dset = pickle.load(
        open(f"part_embedding/taxpose4art/training_data/{dset_name}.pkl", "rb")
    )
    obj_ids_train = []
    obj_ids_test = []
    nrep = int(dset_name.split("_")[1])
    split_file = json.load(
        open("/home/harry/discriminative_embeddings/goalcond-pm-objs-split.json")
    )["train"]

    for cat in dset:
        for partnet_id in dset[cat]:
            if partnet_id in split_file[cat]["train"]:
                for i in range(nrep):
                    obj_ids_train.append(f"{partnet_id}_{i}")
            else:
                for i in range(nrep):
                    obj_ids_test.append(f"{partnet_id}_{i}")

    train_dset = TAXPoseDataset(
        root,
        obj_ids_train,
        dset_name,
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    test_dset = TAXPoseDataset(
        root,
        obj_ids_test,
        dset_name,
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    return train_dset, test_dset, None


def create_flowbot_art_dataset(
    dset_name,
    root,
    process,
    n_repeat,
    randomize_camera,
    n_proc,
    even_downsample,
    rotate_anchor,
    sem_label=None,
    fraction=1,
):
    sem_label_appd = f"_{sem_label}" if sem_label else ""
    train_train_dset = pickle.load(
        open(
            f"part_embedding/taxpose4art/flowbot_data/train_train{sem_label_appd}.pkl",
            "rb",
        )
    )
    train_test_dset = pickle.load(
        open(
            f"part_embedding/taxpose4art/flowbot_data/train_test{sem_label_appd}.pkl",
            "rb",
        )
    )
    test_test_dset = pickle.load(
        open(
            f"part_embedding/taxpose4art/flowbot_data/test_test{sem_label_appd}.pkl",
            "rb",
        )
    )
    obj_ids_train = []
    obj_ids_test = []
    obj_ids_unseen = []

    for cat in train_train_dset:
        if cat == "StorageFurniture":
            len_dset = len(train_train_dset[cat])
            for partnet_id in list(train_train_dset[cat].keys())[
                : int(len_dset // fraction)
            ]:
                if partnet_id in ["10239", "102802"]:
                    continue
                nrep = math.ceil(0.82 * len(train_train_dset[cat][partnet_id]))
                for i in range(nrep):
                    obj_ids_train.append(f"{partnet_id}_{i}")
        else:
            # continue
            len_dset = len(train_train_dset[cat])
            for partnet_id in list(train_train_dset[cat].keys())[
                : int(len_dset // fraction)
            ]:
                if partnet_id in [
                    "103012",
                    "102192",
                    "10239",
                    "103070",
                    "103063",
                    "102805",
                    "103058",
                    "102906",
                    "103253",
                    "103684",
                    "102802",
                    "103032",
                    "103042",
                    "10248",
                ]:
                    continue
                nrep = len(train_train_dset[cat][partnet_id])
                for i in range(nrep):
                    obj_ids_train.append(f"{partnet_id}_{i}")

    for cat in train_test_dset:
        # if cat != "StorageFurniture":
        #     continue
        for partnet_id in train_test_dset[cat]:
            if partnet_id == "102177":
                continue
            nrep = len(train_test_dset[cat][partnet_id])
            for i in range(nrep):
                obj_ids_test.append(f"{partnet_id}_{i}")

    for cat in test_test_dset:
        # if cat != "Table":
        #     continue
        for partnet_id in test_test_dset[cat]:
            if partnet_id in [
                "30739",
                "22433",
                "32601",
                "9280",
                "11826",
                "23782",
                "8893",
                "48492",
                "8930",  # door
                "8903",  # door
                "101599",  # safe
                "26899",  # Table
            ]:
                continue
            nrep = len(test_test_dset[cat][partnet_id])
            for i in range(nrep):
                obj_ids_unseen.append(f"{partnet_id}_{i}")

    train_dset = FBTAXPoseDataset(
        root,
        obj_ids_train,
        f"train_train_aug{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    test_dset = FBTAXPoseDataset(
        root,
        obj_ids_test,
        f"train_test{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    unseen_dset = FBTAXPoseDataset(
        root,
        obj_ids_unseen,
        f"test_test{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    return train_dset, test_dset, unseen_dset


def create_screwflow_dataset(
    dset_name,
    root,
    process,
    n_repeat,
    randomize_camera,
    n_proc,
    even_downsample,
    rotate_anchor,
    sem_label=None,
    fraction=1,
):
    sem_label_appd = f"_{sem_label}" if sem_label else ""
    train_train_dset = pickle.load(
        open(
            f"part_embedding/taxpose4art/flowbot_data/train_train_aug{sem_label_appd}.pkl",
            "rb",
        )
    )
    train_test_dset = pickle.load(
        open(
            f"part_embedding/taxpose4art/flowbot_data/train_test{sem_label_appd}.pkl",
            "rb",
        )
    )
    test_test_dset = pickle.load(
        open(
            f"part_embedding/taxpose4art/flowbot_data/test_test{sem_label_appd}.pkl",
            "rb",
        )
    )
    obj_ids_train = []
    obj_ids_test = []
    obj_ids_unseen = []

    for cat in train_train_dset:
        if cat == "StorageFurniture":
            len_dset = len(train_train_dset[cat])
            for partnet_id in list(train_train_dset[cat].keys())[
                : int(len_dset // fraction)
            ]:
                if partnet_id in ["10239", "102802"]:
                    continue
                nrep = math.ceil(0.82 * len(train_train_dset[cat][partnet_id]))
                for i in range(nrep):
                    obj_ids_train.append(f"{partnet_id}_{i}")
        else:
            # continue
            len_dset = len(train_train_dset[cat])
            for partnet_id in list(train_train_dset[cat].keys())[
                : int(len_dset // fraction)
            ]:
                if partnet_id in [
                    "103012",
                    "102192",
                    "10239",
                    "103070",
                    "103063",
                    "102805",
                    "103058",
                    "102906",
                    "103253",
                    "103684",
                    "102802",
                    "103032",
                    "103042",
                    "10248",
                ]:
                    continue
                nrep = len(train_train_dset[cat][partnet_id])
                for i in range(nrep):
                    obj_ids_train.append(f"{partnet_id}_{i}")

    for cat in train_test_dset:
        # if cat != "StorageFurniture":
        #     continue
        for partnet_id in train_test_dset[cat]:
            if partnet_id == "102177":
                continue
            nrep = len(train_test_dset[cat][partnet_id])
            for i in range(nrep):
                obj_ids_test.append(f"{partnet_id}_{i}")

    for cat in test_test_dset:
        # if cat != "Table":
        #     continue
        for partnet_id in test_test_dset[cat]:
            if partnet_id in [
                "30739",
                "22433",
                "32601",
                "9280",
                "11826",
                "23782",
                "8893",
                "48492",
            ]:
                continue
            nrep = len(test_test_dset[cat][partnet_id])
            for i in range(nrep):
                obj_ids_unseen.append(f"{partnet_id}_{i}")

    train_dset = ScrewDataset(
        root,
        obj_ids_train,
        f"train_train_aug{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    test_dset = ScrewDataset(
        root,
        obj_ids_test,
        f"train_test{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    unseen_dset = ScrewDataset(
        root,
        obj_ids_unseen,
        f"test_test{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    return train_dset, test_dset, unseen_dset


def create_weighted_dataset(
    dset_name,
    root,
    process,
    n_repeat,
    randomize_camera,
    n_proc,
    even_downsample,
    rotate_anchor,
    sem_label=None,
    fraction=1,
):
    sem_label_appd = f"_{sem_label}" if sem_label else ""
    dset = pickle.load(
        open(f"part_embedding/flowtron/dataset/training_data/all_100_obj_tf.pkl", "rb")
    )
    obj_ids_train = []
    obj_ids_test = []
    nrep = 100
    split_file = json.load(
        open("/home/harry/discriminative_embeddings/goalcond-pm-objs-split.json")
    )["train"]

    for cat in dset:
        for partnet_id in dset[cat]:
            if partnet_id in split_file[cat]["train"]:
                if partnet_id in [
                    "48169",
                    "27044",
                    "26525",
                    "27044",
                    "24644",
                    "26503",
                    "22367",
                    "20279",
                    "101940",
                    "103369",
                    "12428",
                    "10347",
                ]:
                    continue
                for i in range(nrep):
                    obj_ids_train.append(f"{partnet_id}_{i}")
            else:
                if partnet_id in ["101943", "27189"]:
                    continue
                for i in range(nrep):
                    obj_ids_test.append(f"{partnet_id}_{i}")

    train_dset = WeightedDataset(
        root,
        obj_ids_train,
        f"train_train_aug{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    test_dset = WeightedDataset(
        root,
        obj_ids_test,
        f"train_test{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    return train_dset, test_dset, None


def create_classifier_dataset(
    dset_name,
    root,
    process,
    n_repeat,
    randomize_camera,
    n_proc,
    even_downsample,
    rotate_anchor,
    sem_label=None,
    fraction=1,
):
    sem_label_appd = f"_{sem_label}" if sem_label else ""
    dset = pickle.load(
        open(f"part_embedding/flowtron/dataset/training_data/all_100_obj_tf.pkl", "rb")
    )
    obj_ids_train = []
    obj_ids_test = []
    nrep = 100
    split_file = json.load(
        open("/home/harry/discriminative_embeddings/goalcond-pm-objs-split.json")
    )["train"]

    for cat in dset:
        for partnet_id in dset[cat]:
            if partnet_id in split_file[cat]["train"]:
                if partnet_id in [
                    "48169",
                    "27044",
                    "26525",
                    "27044",
                    "24644",
                    "26503",
                    "22367",
                    "20279",
                    "101940",
                    "103369",
                    "12428",
                    "10347",
                ]:
                    continue
                if cat in ["Table", "Drawer"]:
                    for i in range(1000):
                        obj_ids_train.append(f"{partnet_id}_{i}")
                else:
                    for i in range(nrep):
                        obj_ids_train.append(f"{partnet_id}_{i}")
            else:
                if partnet_id in ["101943", "27189"]:
                    continue
                for i in range(nrep):
                    obj_ids_test.append(f"{partnet_id}_{i}")

    train_dset = ClassifierDataset(
        root,
        obj_ids_train,
        f"train_train_aug{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    test_dset = ClassifierDataset(
        root,
        obj_ids_test,
        f"train_test{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    return train_dset, test_dset, None


def create_segmenter_dataset(
    dset_name,
    root,
    process,
    n_repeat,
    randomize_camera,
    n_proc,
    even_downsample,
    rotate_anchor,
    sem_label=None,
    fraction=1,
):
    sem_label_appd = f"_{sem_label}" if sem_label else ""
    dset = pickle.load(
        open(f"part_embedding/flowtron/dataset/training_data/all_100_obj_tf.pkl", "rb")
    )
    obj_ids_train = []
    obj_ids_test = []
    nrep = 100
    split_file = json.load(
        open("/home/harry/discriminative_embeddings/goalcond-pm-objs-split.json")
    )["train"]

    for cat in dset:
        for partnet_id in dset[cat]:
            if partnet_id in split_file[cat]["train"]:
                if partnet_id in [
                    "48169",
                    "27044",
                    "26525",
                    "27044",
                    "24644",
                    "26503",
                    "22367",
                    "20279",
                    "101940",
                    "103369",
                    "12428",
                    "10347",
                ]:
                    continue
                if cat in ["Table", "Drawer"]:
                    for i in range(600):
                        obj_ids_train.append(f"{partnet_id}_{i}")
                else:
                    for i in range(nrep):
                        obj_ids_train.append(f"{partnet_id}_{i}")
            else:
                if partnet_id in ["101943", "27189"]:
                    continue
                for i in range(nrep):
                    obj_ids_test.append(f"{partnet_id}_{i}")

    train_dset = SegmenterDataset(
        root,
        obj_ids_train,
        f"train_train_aug{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    test_dset = SegmenterDataset(
        root,
        obj_ids_test,
        f"train_test{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    return train_dset, test_dset, None


def create_fullsys_dataset(
    dset_name,
    root,
    process,
    n_repeat,
    randomize_camera,
    n_proc,
    even_downsample,
    rotate_anchor,
    sem_label=None,
    fraction=1,
):
    sem_label_appd = f"_{sem_label}" if sem_label else ""
    dset = pickle.load(
        open(f"part_embedding/flowtron/dataset/training_data/all_100_obj_tf.pkl", "rb")
    )
    obj_ids_train = []
    obj_ids_test = []
    nrep = 100
    split_file = json.load(
        open("/home/harry/discriminative_embeddings/goalcond-pm-objs-split.json")
    )["train"]

    for cat in dset:
        for partnet_id in dset[cat]:
            if partnet_id in split_file[cat]["train"]:
                if partnet_id in [
                    "48169",
                    "27044",
                    "26525",
                    "27044",
                    "24644",
                    "26503",
                    "22367",
                    "20279",
                    "101940",
                    "103369",
                    "12428",
                    "10347",
                    "12592",
                ]:
                    continue
                if cat in ["Table", "Drawer"]:
                    for i in range(600):
                        obj_ids_train.append(f"{partnet_id}_{i}")
                else:
                    for i in range(nrep):
                        obj_ids_train.append(f"{partnet_id}_{i}")
            else:
                if partnet_id in ["101943", "27189"]:
                    continue
                for i in range(nrep):
                    obj_ids_test.append(f"{partnet_id}_{i}")

    train_dset = FullSysDataset(
        root,
        obj_ids_train,
        f"train_train_aug{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    test_dset = FullSysDataset(
        root,
        obj_ids_test,
        f"train_test{sem_label_appd}",
        use_processed=process,
        n_repeat=n_repeat,
        randomize_camera=randomize_camera,
        n_proc=n_proc,
        even_downsample=even_downsample,
        rotate_anchor=rotate_anchor,
    )

    return train_dset, test_dset, None


def goalflow_hybrid_loss(flow_pred, flow_gt, n_nodes):
    nonzero_gt_flowixs = torch.where(flow_gt.norm(dim=1) != 0.0)
    zero_gt_flowixs = torch.where(flow_gt.norm(dim=1) == 0.0)
    flow_pred[zero_gt_flowixs] = 0

    mse_loss = artflownet_loss(flow_pred, flow_gt, None, n_nodes)

    # flow_pred_norm = flow_pred.reshape(-1, 2000, 3)
    # flow_pred_norm = flow_pred_norm / flow_pred_norm.norm(dim=-1).max(dim=1)[0].reshape(
    #     -1, 1, 1
    # )
    # flow_pred_norm = flow_pred_norm.reshape(-1, 3)
    # flow_gt_norm = flow_gt.reshape(-1, 2000, 3)
    # flow_gt_norm = flow_gt_norm / flow_gt_norm.norm(dim=-1).max(dim=1)[0].reshape(
    #     -1, 1, 1
    # )
    # flow_gt_norm = flow_gt_norm.reshape(-1, 3)
    # norm_mse_loss = artflownet_loss(flow_pred_norm, flow_gt_norm, None, n_nodes)

    # nonzero_gt_flowixs = torch.where(flow_gt.norm(dim=1) != 0.0)
    # gt_flow_nz = flow_gt[nonzero_gt_flowixs]
    # pred_flow_nz = flow_pred[nonzero_gt_flowixs]
    # cos_loss = -torch.cosine_similarity(pred_flow_nz, gt_flow_nz, dim=1).mean()

    return mse_loss, nonzero_gt_flowixs


def goalflow_plot(
    action_pos, svd_pose, anchor_pos, flow_gt, flow_pred, task_sp, obj_id
):
    oacp, oanp = action_pos.detach().cpu(), anchor_pos.detach().cpu()
    flowg = flow_gt.detach().cpu().squeeze()
    flowp = flow_pred.detach().cpu().squeeze()
    svd_pose = svd_pose.detach().cpu()

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "Alignment",
            "Action weight",
        ),
    )

    # Segmentation.
    oanp_goal_gt = (oanp + flowg)[:500]
    oanp_goal_pred = (oanp + flowp)[:500]
    pos = torch.cat([oanp, oanp_goal_gt, oanp_goal_pred, svd_pose], dim=0)
    labels = torch.zeros(len(pos)).int()
    labels[: len(oanp)] = 0
    labels[len(oanp) : len(oanp) + len(oanp_goal_gt)] = 1
    labels[
        len(oanp)
        + len(oanp_goal_gt) : len(oanp)
        + len(oanp_goal_gt)
        + len(oanp_goal_pred)
    ] = 2
    labels[len(oanp) + len(oanp_goal_gt) + len(oanp_goal_pred) :] = 3
    labelmap = {
        0: f"Obj {obj_id} init {task_sp}",
        1: "GT",
        2: "Pred",
        3: "SVD",
    }

    fig.add_traces(pvp._segmentation_traces(pos, labels, labelmap, "scene1"))
    fig.update_layout(
        scene1=pvp._3d_scene(pos),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(xanchor="left", x=0, yanchor="top", y=0.75),
        title=f"Object {obj_id} Task Specification {task_sp}",
    )
    return fig


def flowscrew_plot(
    action_pos, anchor_pos, flow_gt, flow_pred, disp_gt, disp_pred, obj_id
):
    oacp, oanp = action_pos.detach().cpu(), anchor_pos.detach().cpu()
    flowg = flow_gt.detach().cpu().squeeze()
    flowp = flow_pred.detach().cpu().squeeze()
    dispg = disp_gt.detach().cpu().squeeze()
    dispp = disp_pred.detach().cpu().squeeze()

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "Axis",
            "GT Flow",
            "Pred Flow",
        ),
    )

    # Segmentation, axis
    oanp_goal_gt = (oanp + dispg)[:500]
    oanp_goal_pred = (oanp + dispp)[:500]
    pos = torch.cat([oanp_goal_gt, oanp_goal_pred, oanp], dim=0)
    labels = torch.zeros(len(pos)).int()
    labels[:500] = 0
    labels[500:1000] = 1
    labels[1000:] = 2
    labelmap = {0: "GT axis", 1: "Pred axis", 2: "Object"}
    fig.add_traces(pvp._segmentation_traces(pos, labels, labelmap, "scene1"))
    fig.update_layout(
        scene1=pvp._3d_scene(pos),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(xanchor="left", x=0, yanchor="top", y=0.75),
    )

    n_f_target = flowg / flowg.norm(dim=1).max()
    n_f_pred = flowp / flowp.norm(dim=1).max()

    # GT flow
    fig.add_trace(pvp.pointcloud(oanp.T, downsample=1, scene="scene2"), row=1, col=2)
    ts = pvp._flow_traces_v2(oanp, n_f_target, scene="scene2")
    for t in ts:
        fig.add_trace(t, row=1, col=2)
    fig.update_layout(scene2=pvp._3d_scene(oanp))

    # Pred flow
    fig.add_trace(pvp.pointcloud(oanp.T, downsample=1, scene="scene3"), row=1, col=3)
    ts = pvp._flow_traces_v2(oanp, n_f_pred, scene="scene3")
    for t in ts:
        fig.add_trace(t, row=1, col=3)
    fig.update_layout(scene3=pvp._3d_scene(oanp))
    fig.update_layout(title=f"Object {obj_id}")

    return fig


def record_actuation_result(
    results,
    obj_id,
    start_ang,
    res_ang,
    end_ang,
    sem,
    gt_pcd,
    achieved_pcd,
    pred_pcd,
    start_pcd,
):

    if obj_id not in results:
        results[obj_id] = []
    if sem != "slider":
        end_ang = end_ang / np.pi * 180
        if np.abs(res_ang - start_ang) < 10:
            ang_norm_dist = np.abs(res_ang - end_ang) / np.abs(end_ang - start_ang)
        if np.abs(res_ang - end_ang) < 10:
            ang_norm_dist = 0
        else:
            ang_norm_dist = min(
                1, np.abs(res_ang - end_ang) / np.abs(end_ang - start_ang)
            )
    else:
        if np.abs(res_ang - end_ang) < 0.05:
            ang_norm_dist = 0
        else:
            ang_norm_dist = min(
                1, np.abs(res_ang - end_ang) / np.abs(end_ang - start_ang)
            )
    if gt_pcd is not None:
        results[obj_id].append(
            {
                "ang_err": np.abs(res_ang - end_ang),
                "ang_norm_dist": ang_norm_dist,
                "pred_norm_dist": min(
                    1,
                    np.linalg.norm(gt_pcd - pred_pcd)
                    / np.linalg.norm(gt_pcd - start_pcd),
                ),
                "achieved_norm_dist": min(
                    1,
                    np.linalg.norm(gt_pcd - achieved_pcd)
                    / np.linalg.norm(gt_pcd - start_pcd),
                ),
            }
        )
    else:
        results[obj_id].append(
            {
                "ang_err": np.abs(res_ang - end_ang),
                "ang_norm_dist": ang_norm_dist,
            }
        )

    return results
