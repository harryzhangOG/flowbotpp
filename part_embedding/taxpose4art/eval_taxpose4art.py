import torch
import torch_geometric.loader as tgl
from tqdm import tqdm

from part_embedding.goal_inference.dataset_v2 import CATEGORIES
from part_embedding.taxpose4art.train_taxpose4art import Model
from part_embedding.taxpose4art.train_utils import create_art_dataset


def trained_model(ckpt):

    model = Model()
    model.load_state_dict(torch.load(ckpt))

    return model


def n_rot(r):
    return r / r.norm(dim=0).unsqueeze(0)


def theta_err(R_pred, R_gt):
    assert len(R_pred) == 1
    dR = n_rot(R_pred[0].squeeze()).T @ n_rot(R_gt.squeeze())
    dR = n_rot(dR)
    th = torch.arccos((torch.trace(dR) - 1) / 2.0)

    if torch.isnan(th):
        breakpoint()

    return torch.rad2deg(th)


def t_err(t_pred, t_gt):
    assert len(t_pred) == 1
    return torch.norm(t_pred.squeeze() - t_gt.squeeze())


def main(ckpt):

    device = "cuda:0"
    ckpt_dir = f"part_embedding/taxpose4art/checkpoints/{ckpt}/weights_050.pt"
    model = trained_model(ckpt_dir)
    model = model.to(device)

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

    model.eval()

    results = {}
    results_val = {}

    with torch.no_grad():
        pbar = tqdm(train_loader)
        pbar_val = tqdm(test_loader)

        for action, anchor in pbar:
            action = action.to(device)
            anchor = anchor.to(device)
            if anchor.obj_id[0] not in results:
                results[anchor.obj_id[0]] = []

            action = action.to(device)
            anchor = anchor.to(device)

            R_gt = action.R_action_anchor.reshape(-1, 3, 3)
            t_gt = action.t_action_anchor

            R_pred, t_pred, aws, _, _, _ = model(action, anchor)

            results[anchor.obj_id[0]].append(
                {
                    "theta_err": theta_err(R_pred, R_gt),
                    "t_err": t_err(t_pred, t_gt),
                    "t_norm": torch.min(
                        torch.tensor(1).cuda(),
                        t_err(t_pred, t_gt) / t_gt.norm(dim=-1).squeeze().item(),
                    ),
                    "t_mag": t_gt.norm(dim=-1).squeeze().item(),
                }
            )
        for action, anchor in pbar_val:
            action = action.to(device)
            anchor = anchor.to(device)
            if anchor.obj_id[0] not in results_val:
                results_val[anchor.obj_id[0]] = []

            R_gt = action.R_action_anchor.reshape(-1, 3, 3)
            t_gt = action.t_action_anchor

            R_pred, t_pred, aws, _, _, _ = model(action, anchor)

            results_val[anchor.obj_id[0]].append(
                {
                    "theta_err": theta_err(R_pred, R_gt),
                    "t_err": t_err(t_pred, t_gt),
                    "t_norm": torch.min(
                        torch.tensor(1).cuda(),
                        t_err(t_pred, t_gt) / t_gt.norm(dim=-1).squeeze().item(),
                    ),
                    "t_mag": t_gt.norm(dim=-1).squeeze().item(),
                }
            )

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

    th_errs_val = {
        obj_id: torch.mean(torch.stack([d["theta_err"] for d in dlist]))
        for obj_id, dlist in results_val.items()
    }

    t_errs_val = {
        obj_id: torch.mean(torch.stack([d["t_err"] for d in dlist]))
        for obj_id, dlist in results_val.items()
    }

    t_norms_val = {
        obj_id: torch.mean(torch.stack([d["t_norm"] for d in dlist]))
        for obj_id, dlist in results_val.items()
    }

    def classwise(d):
        classdict = {}
        for obj_id, val in d.items():
            cat = CATEGORIES[obj_id.split("_")[0]]
            if cat not in classdict:
                classdict[cat] = []
            classdict[cat].append(val)

        return {cat: torch.mean(torch.stack(ls)) for cat, ls in classdict.items()}

    th_errs_summary = classwise(th_errs)
    t_errs_summary = classwise(t_errs)
    t_norms_summary = classwise(t_norms)
    th_errs_summary_val = classwise(th_errs_val)
    t_errs_summary_val = classwise(t_errs_val)
    t_norms_summary_val = classwise(t_norms_val)
    print("\n\n")
    for cat in th_errs_summary.keys():
        print(
            f"Train/Valid  {cat:<15}\ttheta: {th_errs_summary[cat]:.2f} | {th_errs_summary_val[cat]:.2f}\tt_err: {t_errs_summary[cat]:.2f} | {t_errs_summary_val[cat]:.2f}\tt_normalized: {t_norms_summary[cat]:.2f} | {t_norms_summary_val[cat]:.2f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        help="checkpoint dir name",
    )
    args = parser.parse_args()
    ckpt = args.ckpt

    main(ckpt)
