import os

import torch.nn as nn
import torch.optim
import torch_geometric.loader as tgl
import wandb
from tqdm import tqdm

from part_embedding.flow_prediction.artflownet import create_flownet
from part_embedding.losses.formnet_loss import artflownet_loss
from part_embedding.taxpose4art.train_utils import (
    create_screwflow_dataset,
    flowscrew_plot,
    goalflow_hybrid_loss,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = create_flownet(0, 6)

    def forward(self, action, anchor):
        flowscrew = self.net(anchor)
        return flowscrew


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

    train_dset, test_dset, unseen_dset = create_screwflow_dataset(
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
        wandb.init(project="flow+screw", entity="harryzhangog", config=dict_config)
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

            flow_disp_gt = anchor.flow_disp
            flow_disp_pred = model(action, anchor)
            n_nodes = torch.as_tensor([d.num_nodes for d in anchor.to_data_list()]).to(
                device
            )

            # Hybrid loss training
            hybrid_loss, idx = goalflow_hybrid_loss(
                flow_disp_pred, flow_disp_gt, n_nodes
            )

            n_nodes_action = torch.as_tensor(
                [d.num_nodes for d in action.to_data_list()]
            ).to(device)

            # Supervise the projected axis
            pred_line = action.pos + flow_disp_pred[idx][:, 3:]
            gt_line = action.pos + flow_disp_gt[idx][:, 3:]
            loss = hybrid_loss + artflownet_loss(
                pred_line, gt_line, None, n_nodes_action
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
            flow_gt = flow_disp_gt[:, :3]
            disp_gt = flow_disp_gt[:, 3:]
            flow_gt = flow_gt.reshape(-1, 2000, 3)
            disp_gt = disp_gt.reshape(-1, 2000, 3)
            flow_pred = flow_disp_pred[:, :3]
            disp_pred = flow_disp_pred[:, 3:]
            disp_pred = disp_pred.reshape(-1, 2000, 3)
            flow_pred = flow_pred.reshape(-1, 2000, 3)
            flow_pred[0][500:] = 0
            disp_pred[0][500:] = 0

            obj_id = anchor.obj_id[0]
            wandb.log(
                {
                    "train_rand_plot": flowscrew_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((-1, 2000, 3))[0],
                        flow_gt[0],
                        flow_pred[0],
                        disp_gt[0],
                        disp_pred[0],
                        obj_id,
                    )
                }
            )

        # Validation
        for action, anchor in pbar_val:
            val_step += 1
            action = action.to(device)
            anchor = anchor.to(device)

            flow_disp_gt = anchor.flow_disp

            with torch.no_grad():
                flow_disp_pred = model(action, anchor)
                n_nodes = torch.as_tensor(
                    [d.num_nodes for d in anchor.to_data_list()]
                ).to(device)

                # Hybrid loss training
                hybrid_loss, idx = goalflow_hybrid_loss(
                    flow_disp_pred, flow_disp_gt, n_nodes
                )

                n_nodes_action = torch.as_tensor(
                    [d.num_nodes for d in action.to_data_list()]
                ).to(device)

                # Supervise the projected axis
                pred_line = action.pos + flow_disp_pred[idx][:, 3:]
                gt_line = action.pos + flow_disp_gt[idx][:, 3:]
                loss = hybrid_loss + artflownet_loss(
                    pred_line, gt_line, None, n_nodes_action
                )

                if wandb_log:
                    wandb.log({"val_loss": loss, "val-x-axis": val_step})

                if i % n_print == 0:
                    desc = f"Epoch {i:03d}: Step {val_step}  Val Loss:{loss.item():.3f}"
                    pbar_val.set_description(desc)
        if wandb_log:
            flow_gt = flow_disp_gt[:, :3]
            disp_gt = flow_disp_gt[:, 3:]
            flow_gt = flow_gt.reshape(-1, 2000, 3)
            disp_gt = disp_gt.reshape(-1, 2000, 3)
            flow_pred = flow_disp_pred[:, :3]
            disp_pred = flow_disp_pred[:, 3:]
            disp_pred = disp_pred.reshape(-1, 2000, 3)
            flow_pred = flow_pred.reshape(-1, 2000, 3)
            flow_pred[0][500:] = 0
            disp_pred[0][500:] = 0

            obj_id = anchor.obj_id[0]
            wandb.log(
                {
                    "val_rand_plot": flowscrew_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((-1, 2000, 3))[0],
                        flow_gt[0],
                        flow_pred[0],
                        disp_gt[0],
                        disp_pred[0],
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

                flow_disp_gt = anchor.flow_disp

                with torch.no_grad():
                    flow_disp_pred = model(action, anchor)
                    n_nodes = torch.as_tensor(
                        [d.num_nodes for d in anchor.to_data_list()]
                    ).to(device)

                    # Hybrid loss training
                    hybrid_loss, idx = goalflow_hybrid_loss(
                        flow_disp_pred, flow_disp_gt, n_nodes
                    )

                    n_nodes_action = torch.as_tensor(
                        [d.num_nodes for d in action.to_data_list()]
                    ).to(device)

                    # Supervise the projected axis
                    pred_line = action.pos + flow_disp_pred[idx][:, 3:]
                    gt_line = action.pos + flow_disp_gt[idx][:, 3:]
                    loss = hybrid_loss + artflownet_loss(
                        pred_line, gt_line, None, n_nodes_action
                    )

                    if wandb_log:
                        wandb.log({"unseen_loss": loss, "unseen-x-axis": unseen_step})

                    if i % n_print == 0:
                        desc = f"Epoch {i:03d}: Step {val_step}  Unseen Loss:{loss.item():.3f}"
                        pbar_val.set_description(desc)
            if wandb_log:
                flow_gt = flow_disp_gt[:, :3]
                disp_gt = flow_disp_gt[:, 3:]
                flow_gt = flow_gt.reshape(-1, 2000, 3)
                disp_gt = disp_gt.reshape(-1, 2000, 3)
                flow_pred = flow_disp_pred[:, :3]
                disp_pred = flow_disp_pred[:, 3:]
                disp_pred = disp_pred.reshape(-1, 2000, 3)
                flow_pred = flow_pred.reshape(-1, 2000, 3)
                flow_pred[0][500:] = 0
                disp_pred[0][500:] = 0

                obj_id = anchor.obj_id[0]
                wandb.log(
                    {
                        "unseen_rand_plot": flowscrew_plot(
                            action.pos.reshape((-1, 500, 3))[0],
                            anchor.pos.reshape((-1, 2000, 3))[0],
                            flow_gt[0],
                            flow_pred[0],
                            disp_gt[0],
                            disp_pred[0],
                            obj_id,
                        )
                    }
                )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

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
    sem_label = None
    wandb_log = args.wandb
    frac = args.frac

    main(
        dset_name=f"flow_screw",
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
