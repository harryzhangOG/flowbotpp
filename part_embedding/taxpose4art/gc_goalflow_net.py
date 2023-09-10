import torch
import torch.nn as nn
import torch.nn.functional as F

import part_embedding.nets.pointnet2 as pnp
from part_embedding.nets.pointnet2 import PN2DenseParams


class GCGoalFlowNet(nn.Module):
    def __init__(self, p: PN2DenseParams = PN2DenseParams()):
        super().__init__()
        p.in_dim = 0
        p.final_outdim = 3

        # The Set Aggregation modules.
        self.sa1 = pnp.SAModule(3 + p.in_dim, p.sa1_outdim, p=p.sa1)
        self.sa2 = pnp.SAModule(3 + p.sa1_outdim, p.sa2_outdim, p=p.sa1)
        self.sa3 = pnp.GlobalSAModule(3 + p.sa2_outdim, p.gsa_outdim, p=p.gsa)

        # The Feature Propagation modules.
        self.fp3 = pnp.FPModule(p.gsa_outdim + p.sa2_outdim, p.sa2_outdim, p.fp3)
        self.fp2 = pnp.FPModule(p.sa2_outdim + p.sa1_outdim, p.sa1_outdim, p.fp2)
        self.fp1 = pnp.FPModule(p.sa1_outdim + p.in_dim, p.fp1_outdim, p.fp1)

        # Final linear layers at the output.
        self.lin1 = torch.nn.Linear(p.fp1_outdim, p.lin1_dim)
        self.lin2 = torch.nn.Linear(p.lin1_dim, p.lin2_dim)
        self.lin3 = torch.nn.Linear(p.lin2_dim, p.final_outdim)  # Flow output.
        self.proj1 = torch.nn.Linear(p.sa2_outdim // 2, p.sa2_outdim)
        self.proj2 = torch.nn.Linear(p.sa1_outdim, p.sa1_outdim)
        self.task_mlp = nn.Sequential(nn.Linear(2, 128), nn.ReLU())

    def forward(self, data):
        sa0_out = (data.x, data.pos.float(), data.batch)

        task_sp = self.task_mlp(data.task_sp)

        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        x3, pos3, batch3 = self.sa3(*sa2_out)

        # CLIPort network uses tiling then hadamard product at each subsequent layer.
        x3 = torch.mul(x3, task_sp.tile(1, 8))
        sa3_out = x3, pos3, batch3

        fp3_x, fp3_pos, fp3_batch = self.fp3(*sa3_out, *sa2_out)
        temp = torch.cat(
            [
                task_sp[i].tile((fp3_batch == i).sum(), 2)
                for i in range(task_sp.shape[0])
            ]
        )
        fp3_x = torch.mul(fp3_x, temp)
        fp3_out = fp3_x, fp3_pos, fp3_batch
        fp2_x, fp2_pos, fp2_batch = self.fp2(*fp3_out, *sa1_out)
        temp = torch.cat(
            [
                task_sp[i].tile((fp2_batch == i).sum(), 1)
                for i in range(task_sp.shape[0])
            ]
        )
        fp2_x = torch.mul(fp2_x, temp)
        fp2_out = fp2_x, fp2_pos, fp2_batch
        x, _, _ = self.fp1(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x
