import os

import numpy as np
import pytorch3d
import pytorch3d.transforms
import torch.nn as nn
import torch.optim
import torch_geometric.loader as tgl
import wandb
from tqdm import tqdm

from part_embedding.goal_inference.model_sg import SVDHead, dcp_sg_plot
from part_embedding.goal_inference_brian.brian_chuer_model import BrianChuerAdapter
from part_embedding.goal_inference_brian.loss import BrianChuerLoss
from part_embedding.taxpose4art.train_utils import create_flowbot_art_dataset


class Model(nn.Module):
    def __init__(self, attn=True, embedding_dim=512):
        super().__init__()
        self.net = BrianChuerAdapter(emb_dims=embedding_dim, gc=True)

        self.attn = attn

        self.head = SVDHead()

    @staticmethod
    def norm_scale(pos):
        mean = pos.mean(dim=1).unsqueeze(1)
        pos = pos - mean
        scale = pos.abs().max(dim=2)[0].max(dim=1)[0]
        pos = pos / (scale.view(-1, 1, 1) + 1e-8)
        return pos

    # def forward(self, action, anchor):
    #     X, Y = action.pos, anchor.pos
    #     Xs = X.view(action.num_graphs, -1, 3)
    #     Ys = Y.view(anchor.num_graphs, -1, 3)
    #     R_pred, t_pred, pred_T_action, Fx, Fy = self.net(
    #         Xs, Ys, action.loc.unsqueeze(-1)
    #     )
    #     return R_pred, t_pred, None, pred_T_action, Fx, Fy
    def forward(self, action, anchor):
        X, Y = action.pos, anchor.pos
        task_sps = []
        for sp in action.loc.float().reshape(-1, 1):
            if abs(sp.item()) < 1:
                task_sps.append([0, sp.item()])
            else:
                task_sps.append([1, sp.item()])
        task_sps = np.array(task_sps)
        task_sps = torch.from_numpy(task_sps).cuda().float().reshape(-1, 2)
        Xs = X.view(action.num_graphs, -1, 3)
        Ys = Y.view(anchor.num_graphs, -1, 3)
        R_pred, t_pred, pred_T_action, Fx, Fy = self.net(Xs, Ys, task_sps)
        return R_pred, t_pred, None, pred_T_action, Fx, Fy


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
            project="scene-taxpose-art", entity="harryzhangog", config=dict_config
        )
        run_name = wandb.run.name
        run_name_log = f"{dset_name}-{run_name}"
        wandb.run.name = run_name_log
    else:
        run_name_log = "debug"

    model = Model(attn=True, embedding_dim=embedding_dim).to(device)

    # opt = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    crit = BrianChuerLoss()

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

            R_gt = action.R_action_anchor.reshape(-1, 3, 3)
            t_gt = action.t_action_anchor
            mat = torch.zeros(action.num_graphs, 4, 4).to(device)
            mat[:, :3, :3] = R_gt
            mat[:, :3, 3] = t_gt
            mat[:, 3, 3] = 1

            R_pred, t_pred, _, pred_T_action, Fx, Fy = model(action, anchor)

            gt_T_action = pytorch3d.transforms.Transform3d(
                device=device, matrix=mat.transpose(-1, -2)
            )

            loss = crit(
                action.pos.reshape(action.num_graphs, -1, 3),
                pred_T_action,
                gt_T_action,
                Fx,
            )
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
            wandb.log(
                {
                    "train_rand_plot": dcp_sg_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((-1, 2000, 3))[0],
                        t_gt[0],
                        t_pred[0],
                        R_gt[0],
                        R_pred[0],
                        None,
                    )
                }
            )

        # Validation
        for action, anchor in pbar_val:
            val_step += 1
            action = action.to(device)
            anchor = anchor.to(device)

            R_gt = action.R_action_anchor.reshape(-1, 3, 3)
            t_gt = action.t_action_anchor
            mat = torch.zeros(action.num_graphs, 4, 4).to(device)
            mat[:, :3, :3] = R_gt
            mat[:, :3, 3] = t_gt
            mat[:, 3, 3] = 1

            with torch.no_grad():
                R_pred, t_pred, _, pred_T_action, Fx, Fy = model(action, anchor)

                gt_T_action = pytorch3d.transforms.Transform3d(
                    device=device, matrix=mat.transpose(-1, -2)
                )

                loss = crit(
                    action.pos.reshape(action.num_graphs, -1, 3),
                    pred_T_action,
                    gt_T_action,
                    Fx,
                )
                if wandb_log:
                    wandb.log({"val_loss": loss, "val-x-axis": val_step})

                if i % n_print == 0:
                    desc = f"Epoch {i:03d}: Step {val_step}  Val Loss:{loss.item():.3f}"
                    pbar_val.set_description(desc)
        if wandb_log:
            wandb.log(
                {
                    "val_rand_plot": dcp_sg_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((-1, 2000, 3))[0],
                        t_gt[0],
                        t_pred[0],
                        R_gt[0],
                        R_pred[0],
                        None,
                    )
                }
            )

        # Unseen
        if unseen_dset:
            for action, anchor in pbar_unseen:
                unseen_step += 1
                action = action.to(device)
                anchor = anchor.to(device)

                R_gt = action.R_action_anchor.reshape(-1, 3, 3)
                t_gt = action.t_action_anchor
                mat = torch.zeros(action.num_graphs, 4, 4).to(device)
                mat[:, :3, :3] = R_gt
                mat[:, :3, 3] = t_gt
                mat[:, 3, 3] = 1

                with torch.no_grad():
                    R_pred, t_pred, _, pred_T_action, Fx, Fy = model(action, anchor)

                    gt_T_action = pytorch3d.transforms.Transform3d(
                        device=device, matrix=mat.transpose(-1, -2)
                    )

                    loss = crit(
                        action.pos.reshape(action.num_graphs, -1, 3),
                        pred_T_action,
                        gt_T_action,
                        Fx,
                    )
                    wandb.log({"unseen_loss": loss, "unseen-x-axis": unseen_step})

                    if i % n_print == 0:
                        desc = f"Epoch {i:03d}: Step {val_step}  Unseen Loss:{loss.item():.3f}"
                        pbar_val.set_description(desc)
            if wandb_log:
                wandb.log(
                    {
                        "unseen_rand_plot": dcp_sg_plot(
                            action.pos.reshape((-1, 500, 3))[0],
                            anchor.pos.reshape((-1, 2000, 3))[0],
                            t_gt[0],
                            t_pred[0],
                            R_gt[0],
                            R_pred[0],
                            None,
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
        help="Sem Label hinge/slider/both.",
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

    args = parser.parse_args()
    dset_cat = args.cat
    dset_num = args.num
    sem_label = args.sem
    wandb_log = args.wandb

    main(
        dset_name=f"{dset_cat}_{dset_num}_obj_tf",
        root=os.path.expanduser("~/partnet-mobility"),
        batch_size=16,
        use_bc_loss=True,
        n_epochs=50,
        lr=0.001,
        n_repeat=1,
        embedding_dim=512,
        n_proc=60,
        sem_label=sem_label,
        wandb_log=wandb_log,
    )
