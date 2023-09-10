import os

import numpy as np
import torch.nn as nn
import torch.optim
import torch_geometric.loader as tgl
import wandb
from tqdm import tqdm

from part_embedding.flow_prediction.artflownet import ArtFlowNetParams
from part_embedding.losses.formnet_loss import artflownet_loss
from part_embedding.taxpose4art.gc_goalflow_net import GCGoalFlowNet
from part_embedding.taxpose4art.train_utils import (
    create_flowbot_art_dataset,
    goalflow_hybrid_loss,
    goalflow_plot,
)


class SVD(nn.Module):
    def __init__(self):
        super(SVD, self).__init__()
        self.emb_dims = 16

    def forward(self, src, pred_flow, gt_flow):
        nonzero_gt_flowixs = torch.where(gt_flow.norm(dim=1) != 0.0)
        pred_flow_nz = pred_flow[nonzero_gt_flowixs].reshape(-1, 500, 3)
        src = (src).reshape(-1, 500, 3)
        # R, t = flow2pose(src, pred_flow_nz)

        X = src
        Y = src + pred_flow_nz
        X_bar = X.mean(axis=1).reshape(-1, 1, 3)
        Y_bar = Y.mean(axis=1).reshape(-1, 1, 3)
        H = torch.bmm((X - X_bar).transpose(1, 2), (Y - Y_bar))
        U, _, Vh = torch.linalg.svd(H)
        R = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))
        t = -torch.bmm(X_bar, R.transpose(1, 2)) + Y_bar

        return R, t


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        p = ArtFlowNetParams()
        # self.net = create_flownet(in_channels=1, out_channels=3, p=p.net)
        self.net = GCGoalFlowNet()
        self.svd = SVD()

    def forward(self, action, anchor):
        if isinstance(self.net, GCGoalFlowNet):
            # anchor.task_sp = action.loc.float().reshape(-1, 1)
            task_sps = []
            for sp in action.loc.float().reshape(-1, 1):
                if abs(sp.item()) < 1:
                    task_sps.append([0, sp.item()])
                else:
                    task_sps.append([1, sp.item()])
            task_sps = np.array(task_sps)
            anchor.task_sp = torch.from_numpy(task_sps).cuda().float().reshape(-1, 2)
            anchor.x = None
        flow = self.net(anchor)
        return flow


def main(
    dset_name: str,
    root="/home/harry/partnet-mobility/raw",
    batch_size: int = 16,
    use_bc_loss: bool = True,
    n_epochs: int = 50,
    lr: float = 0.001,
    n_repeat: int = 50,
    embedding_dim: int = 512,
    n_proc: int = 60,
    sem_label=None,
    wandb_log=False,
    frac=1,
):
    torch.autograd.set_detect_anomaly(True)
    n_print = 1

    device = "cuda:0"

    # Set up wandb config
    dict_config = {
        "learning_rate": lr,
        "n_repeat": n_repeat,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "dataset": dset_name,
    }

    train_dset, test_dset, unseen_dset = create_flowbot_art_dataset(
        dset_name,
        root,
        True,
        n_repeat,
        False,
        n_proc,
        True,
        False,
        sem_label=sem_label,
        fraction=frac,
    )  # Third one is process

    train_loader = tgl.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = tgl.DataLoader(
        test_dset, batch_size=min(len(test_dset), 4), shuffle=True, num_workers=0
    )
    if unseen_dset:
        unseen_loader = tgl.DataLoader(
            unseen_dset,
            batch_size=min(len(unseen_dset), 4),
            shuffle=True,
            num_workers=0,
        )
    if wandb_log:
        wandb.init(
            project="goalflow-hybrid-loss", entity="harryzhangog", config=dict_config
        )
        run_name = wandb.run.name
        run_name_log = f"{dset_name}-{run_name}"
        wandb.run.name = run_name_log
    else:
        run_name_log = "debug"

    model = Model().to(device)

    # opt = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    d = f"part_embedding/taxpose4art/checkpoints/{run_name_log}"
    os.makedirs(d, exist_ok=True)

    # Train
    train_step = 0
    val_step = 0
    unseen_step = 0
    for i in range(1, n_epochs + 1):
        pbar = tqdm(train_loader)
        pbar_val = tqdm(test_loader)
        if unseen_dset:
            pbar_unseen = tqdm(unseen_loader)
        for action, anchor in pbar:
            train_step += 1
            action = action.to(device)
            anchor = anchor.to(device)

            opt.zero_grad()

            flow_gt = anchor.flow
            flow_pred = model(action, anchor)
            n_nodes = torch.as_tensor([d.num_nodes for d in anchor.to_data_list()]).to(
                device
            )

            # Hybrid loss training
            hybrid_loss, idx = goalflow_hybrid_loss(flow_pred, flow_gt, n_nodes)
            R_pred, t_pred = model.svd(action.pos, flow_pred, flow_gt)
            pred_pose = (
                (
                    torch.bmm(action.pos.reshape(-1, 500, 3), R_pred.transpose(-1, -2))
                    + t_pred
                )
                .reshape(-1, 3)
                .to(device)
            )

            n_nodes_action = torch.as_tensor(
                [d.num_nodes for d in action.to_data_list()]
            ).to(device)
            loss = (
                hybrid_loss
                + artflownet_loss(
                    pred_pose, action.pos + flow_gt[idx], None, n_nodes_action
                )
                + artflownet_loss(
                    pred_pose, action.pos + flow_pred[idx], None, n_nodes_action
                )
            )
            # loss = artflownet_loss(flow_pred, flow_gt, None, n_nodes)

            if wandb_log:
                wandb.log({"train_loss": loss, "train-x-axis": train_step})

            loss.backward()
            opt.step()

            if i % n_print == 0:
                desc = (
                    f"Epoch {i:03d}:  Step {train_step}  Train Loss:{loss.item():.3f}"
                )
                pbar.set_description(desc)

        if i % 10 == 0:
            torch.save(model.state_dict(), os.path.join(d, f"weights_{i:03d}.pt"))

        if wandb_log:
            flow_gt = flow_gt.reshape(-1, 2000, 3)
            flow_pred = flow_pred.reshape(-1, 2000, 3)
            flow_pred[0][500:] = 0

            task_sp = action.loc[0].item()
            obj_id = anchor.obj_id[0]
            wandb.log(
                {
                    "train_rand_plot": goalflow_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        pred_pose.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((-1, 2000, 3))[0],
                        flow_gt[0],
                        flow_pred[0],
                        task_sp,
                        obj_id,
                    )
                }
            )

        # Validation
        for action, anchor in pbar_val:
            val_step += 1
            action = action.to(device)
            anchor = anchor.to(device)

            flow_gt = anchor.flow

            with torch.no_grad():
                flow_pred = model(action, anchor)
                n_nodes = torch.as_tensor(
                    [d.num_nodes for d in anchor.to_data_list()]
                ).to(device)

                # Hybrid loss training
                hybrid_loss, idx = goalflow_hybrid_loss(flow_pred, flow_gt, n_nodes)
                R_pred, t_pred = model.svd(action.pos, flow_pred, flow_gt)
                pred_pose = (
                    (
                        torch.bmm(
                            action.pos.reshape(-1, 500, 3), R_pred.transpose(-1, -2)
                        )
                        + t_pred
                    )
                    .reshape(-1, 3)
                    .to(device)
                )

                n_nodes_action = torch.as_tensor(
                    [d.num_nodes for d in action.to_data_list()]
                ).to(device)
                loss = (
                    hybrid_loss
                    + artflownet_loss(
                        pred_pose, action.pos + flow_gt[idx], None, n_nodes_action
                    )
                    + artflownet_loss(
                        pred_pose, action.pos + flow_pred[idx], None, n_nodes_action
                    )
                )
                # loss = artflownet_loss(flow_pred, flow_gt, None, n_nodes)

                if wandb_log:
                    wandb.log({"val_loss": loss, "val-x-axis": val_step})

                if i % n_print == 0:
                    desc = f"Epoch {i:03d}: Step {val_step}  Val Loss:{loss.item():.3f}"
                    pbar_val.set_description(desc)
        if wandb_log:
            flow_gt = flow_gt.reshape(-1, 2000, 3)
            flow_pred = flow_pred.reshape(-1, 2000, 3)
            flow_pred[0][500:] = 0

            task_sp = action.loc[0].item()

            obj_id = anchor.obj_id[0]
            wandb.log(
                {
                    "val_rand_plot": goalflow_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        pred_pose.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((-1, 2000, 3))[0],
                        flow_gt[0],
                        flow_pred[0],
                        task_sp,
                        obj_id,
                    )
                }
            )

        # Unseen
        if unseen_dset:
            for action, anchor in pbar_unseen:
                unseen_step += 1
                action = action.to(device)
                anchor = anchor.to(device)

                flow_gt = anchor.flow

                with torch.no_grad():
                    flow_pred = model(action, anchor)
                    n_nodes = torch.as_tensor(
                        [d.num_nodes for d in anchor.to_data_list()]
                    ).to(device)

                    # Hybrid loss training
                    hybrid_loss, idx = goalflow_hybrid_loss(flow_pred, flow_gt, n_nodes)
                    R_pred, t_pred = model.svd(action.pos, flow_pred, flow_gt)
                    pred_pose = (
                        (
                            torch.bmm(
                                action.pos.reshape(-1, 500, 3), R_pred.transpose(-1, -2)
                            )
                            + t_pred
                        )
                        .reshape(-1, 3)
                        .to(device)
                    )

                    n_nodes_action = torch.as_tensor(
                        [d.num_nodes for d in action.to_data_list()]
                    ).to(device)
                    loss = (
                        hybrid_loss
                        + artflownet_loss(
                            pred_pose, action.pos + flow_gt[idx], None, n_nodes_action
                        )
                        + artflownet_loss(
                            pred_pose, action.pos + flow_pred[idx], None, n_nodes_action
                        )
                    )
                    # loss = artflownet_loss(flow_pred, flow_gt, None, n_nodes)

                    if wandb_log:
                        wandb.log({"unseen_loss": loss, "unseen-x-axis": unseen_step})

                    if i % n_print == 0:
                        desc = f"Epoch {i:03d}: Step {val_step}  Unseen Loss:{loss.item():.3f}"
                        pbar_val.set_description(desc)
            if wandb_log:
                flow_gt = flow_gt.reshape(-1, 2000, 3)
                flow_pred = flow_pred.reshape(-1, 2000, 3)
                flow_pred[0][500:] = 0

                task_sp = action.loc[0].item()

                obj_id = anchor.obj_id[0]
                wandb.log(
                    {
                        "unseen_rand_plot": goalflow_plot(
                            action.pos.reshape((-1, 500, 3))[0],
                            pred_pose.reshape((-1, 500, 3))[0],
                            anchor.pos.reshape((-1, 2000, 3))[0],
                            flow_gt[0],
                            flow_pred[0],
                            task_sp,
                            obj_id,
                        )
                    }
                )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sem",
        type=str,
        default=None,
        help="Sem Label hinge/slider/None (both).",
    )
    parser.add_argument(
        "--cat",
        type=str,
        default="fridge",
        help="Generated dataset category name to pass in.",
    )
    parser.add_argument(
        "--num",
        type=str,
        default="100",
        help="Generated dataset nrepeat to pass in.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="if we want to wandb",
    )
    parser.add_argument(
        "--frac",
        type=str,
        default="1",
        help="Generated dataset category name to pass in.",
    )

    args = parser.parse_args()
    dset_cat = args.cat
    dset_num = args.num
    sem_label = args.sem
    wandb_log = args.wandb
    frac = args.frac

    main(
        dset_name=f"{dset_cat}_{dset_num}_obj_tf",
        root=os.path.expanduser("~/partnet-mobility"),
        batch_size=16,
        use_bc_loss=True,
        n_epochs=75,
        lr=0.0003,
        n_repeat=1,
        embedding_dim=512,
        n_proc=60,
        sem_label=sem_label,
        wandb_log=wandb_log,
        frac=float(frac),
    )
