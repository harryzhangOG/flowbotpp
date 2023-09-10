import os
import pickle
from typing import Dict, List, Optional, Protocol

import numpy as np
import pybullet as pbullet
import torch
import torch.utils.data as td
import torch_geometric.data as tgd

from part_embedding.datasets.calc_art import compute_new_points, fk
from part_embedding.datasets.pm.pm_raw import PMRawData
from part_embedding.datasets.pm.utils import SingleObjDataset, parallel_sample
from part_embedding.envs.render_sim import PMRenderEnv
from part_embedding.goal_inference.dataset_v2 import downsample_pcd_fps
from part_embedding.goal_inference.dset_utils import render_input_articulated
from part_embedding.taxpose4art.generate_art_training_data_flowbot import (
    get_category,
    get_sem,
)


def compute_flow_and_localscrew(sim, pm_raw_data, link_name, P_world, mask):

    """
    Return:
     - Point-wise flow (nx3)
     - Point-wise displacement to articulation axis (nx3)
     - Articulation axis direction
     - Articulation axis origin
    """

    flow = np.zeros_like(P_world)

    linkname_to_id = {
        pbullet.getBodyInfo(sim.obj_id, physicsClientId=sim.client_id)[0].decode(
            "UTF-8"
        ): -1
    }
    for _id in range(pbullet.getNumJoints(sim.obj_id, physicsClientId=sim.client_id)):
        _name = pbullet.getJointInfo(sim.obj_id, _id, physicsClientId=sim.client_id)[
            12
        ].decode("UTF-8")
        linkname_to_id[_name] = _id

    link_ixs = mask == 1
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

    target_joint = [x for x in pm_raw_data.obj.joints if x.child == link_name][0]
    joint = pm_raw_data.obj.get_joint(target_joint.name)
    angles = sim.get_joint_angles()
    jas = [angles[jt.name] for jt in chain[:-1]]
    T_base_link = fk(chain[:-1], jas)
    T_world_base = np.copy(sim.T_world_base)
    T_world_link = T_world_base @ T_base_link

    # Take the origin and axis in the link frame.
    o_link, _ = joint.origin
    a_link = joint.axis
    p = np.array([o_link[0], o_link[1], o_link[2], 1.0]).reshape((4, 1))
    v = np.array([a_link[0], a_link[1], a_link[2]]).reshape((3, 1))
    origin_base = (T_world_link @ p)[:3, 0]
    axis_base = (T_world_link[:3, :3] @ v)[:3, 0]
    origin = origin_base.reshape(3, 1)
    axis = axis_base.reshape(3, 1)
    projection = (
        lambda x: (axis @ axis.T) @ x.T / np.dot(axis_base, axis_base)
        + (np.eye(3) - (axis @ axis.T) / np.dot(axis_base, axis_base)) @ origin
    )
    proj_action = np.copy(P_world)
    proj_action[link_ixs] = projection(P_world[link_ixs]).T
    dist_to_axis = proj_action - P_world

    return flow, dist_to_axis, origin, axis


class ArtData(Protocol):

    # Action Info
    action_pos: torch.FloatTensor
    t_action_anchor: Optional[torch.FloatTensor]
    R_action_anchor: Optional[torch.FloatTensor]
    flow: Optional[torch.FloatTensor]

    # Anchor Info
    obj_id: str
    anchor_pos: torch.FloatTensor

    # Task specification
    loc: Optional[float]


class ScrewDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        obj_ids: List[str],
        dset_name: str = None,
        use_processed: bool = True,
        n_repeat: int = 50,
        randomize_camera: bool = False,
        n_proc: int = 60,
        even_downsample: bool = False,
        rotate_anchor: bool = False,
    ):

        self.obj_ids = obj_ids
        self.dset_name = dset_name
        self.generated_metadata = pickle.load(
            open(f"part_embedding/taxpose4art/flowbot_data/{dset_name}.pkl", "rb")
        )

        # Cache the environments. Only cache at the object level though.
        self.envs: Dict[str, PMRenderEnv] = {}
        if "hinge" in dset_name or "slider" in dset_name:
            if "hinge" in dset_name:
                split_dset_name = dset_name[:-6]
            else:
                split_dset_name = dset_name[:-7]
        else:
            split_dset_name = dset_name
        self.full_sem_dset = pickle.load(
            open(
                f"part_embedding/taxpose4art/flowbot_split/{split_dset_name}.pkl", "rb"
            )
        )

        self.use_processed = use_processed
        self.n_repeat = n_repeat
        self.randomize_camera = randomize_camera
        self.n_proc = n_proc
        self.even_downsample = even_downsample
        self.rotate_anchor = rotate_anchor

        super().__init__(root)
        if self.use_processed:
            # Map from scene_id to dataset. Very hacky way of getting scene id.
            self.inmem_map: Dict[str, td.Dataset[ArtData]] = {
                data_path: SingleObjDataset(data_path)
                for data_path in self.processed_paths
            }
            self.inmem: td.ConcatDataset = td.ConcatDataset(
                list(self.inmem_map.values())
            )

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{key}_{self.n_repeat}.pt" for key in self.obj_ids]

    @property
    def processed_dir(self) -> str:
        chunk = ""
        if self.randomize_camera:
            chunk += "_random"
        if self.even_downsample:
            chunk += "_even"
        return os.path.join(self.root, f"flow_screw_{self.dset_name}" + chunk)

    def process(self):
        if not self.use_processed:
            return

        else:
            # Run parallel sampling!
            get_data_args = [(f.split("_")[0], f.split("_")[1]) for f in self.obj_ids]
            parallel_sample(
                dset_cls=ScrewDataset,
                dset_args=(
                    self.root,
                    self.obj_ids,
                    self.dset_name,
                    False,
                    self.n_repeat,
                    self.randomize_camera,
                    self.n_proc,
                    self.even_downsample,
                    self.rotate_anchor,
                ),
                # Needs to be a tuple of arguments, so we can expand it when calling get_args.
                get_data_args=get_data_args,
                n_repeat=self.n_repeat,
                n_proc=self.n_proc,
            )

    def len(self) -> int:
        return len(self.obj_ids) * self.n_repeat

    def get(self, idx: int) -> ArtData:
        if self.use_processed:
            try:
                data = self.inmem[idx]  # type: ignore
            except:
                breakpoint()
        else:
            idx = idx // self.n_repeat
            obj_id = self.obj_ids[idx]
            data = self.get_data(obj_id)

        return data  # type: ignore

    def get_data(
        self,
        obj_id: str,
    ) -> ArtData:
        """Get a single observation sample.

        Args:
            obj_id: The anchor object ID from Partnet-Mobility.
        Returns:
            ObsActionData and AnchorData, both in the world frame, with a relative transform.
        """

        tmp_id = obj_id.split("_")[0]

        # First, create an environment which will generate our source observations.
        env = PMRenderEnv(tmp_id, self.raw_dir, camera_pos=[-2.5, 0, 2.5], gui=False)

        # Obtain entries from meta data
        curr_data_entry = self.generated_metadata[get_category(tmp_id)][tmp_id][
            int(obj_id.split("_")[1])
        ]
        start_ang = curr_data_entry["start"]

        # Next, check to see if the object needs to be opened in any way.
        for ent in self.full_sem_dset[get_category(tmp_id)]:
            if tmp_id in ent[0]:
                links_tomove = ent[1]
                break
        env.set_specific_joints_angle(links_tomove, start_ang, sem=get_sem(ent))
        pm_raw_data = PMRawData(os.path.join(self.root, "raw", tmp_id))

        # Render the scene.
        if self.randomize_camera:
            env.randomize_camera()

        # Render the scene.
        P_world, pc_seg, rgb, action_mask = render_input_articulated(env, links_tomove)

        # Separate out the action and anchor points.
        P_action_world = P_world[action_mask]
        P_anchor_world = P_world[~action_mask]

        P_action_world = torch.from_numpy(P_action_world)
        P_anchor_world = torch.from_numpy(P_anchor_world)

        action_pts_num = 500
        # Now, downsample
        if self.even_downsample:
            action_ixs = downsample_pcd_fps(P_action_world, n=action_pts_num)
            anchor_ixs = downsample_pcd_fps(P_anchor_world, n=2000 - action_pts_num)
        else:
            action_ixs = torch.randperm(len(P_action_world))[:action_pts_num]
            anchor_ixs = torch.randperm(len(P_anchor_world))[: (2000 - action_pts_num)]

        # Rebuild the world
        P_action_world = P_action_world[action_ixs]
        while len(P_action_world) < action_pts_num:
            temp = np.random.choice(np.arange(len(P_action_world)))
            P_action_world = torch.cat(
                [
                    P_action_world,
                    P_action_world[temp : temp + 1],
                ]
            )

        if len(P_action_world) != 500:
            print(len(P_action_world))

        P_anchor_world = P_anchor_world[anchor_ixs]
        while len(P_anchor_world) < 2000 - action_pts_num:
            temp = np.random.choice(np.arange(len(P_anchor_world)))
            P_anchor_world = torch.cat(
                [
                    P_anchor_world,
                    P_anchor_world[temp : temp + 1],
                ]
            )
        P_world = np.concatenate([P_action_world, P_anchor_world], axis=0)

        # Regenerate a mask.
        mask_act = torch.ones(len(P_action_world)).int()
        mask_anc = torch.zeros(len(P_anchor_world)).int()
        mask = torch.cat([mask_act, mask_anc])

        # Get GT flow, local art param displacement, axis direction, and origin
        flow, disp, axis_dir, axis_orig = compute_flow_and_localscrew(
            env, pm_raw_data, links_tomove, P_world, mask
        )

        # Assemble the data.
        action_data = tgd.Data(
            pos=P_action_world.float(),
        )

        flow_disp = np.hstack([flow, disp])

        assert flow_disp.shape[1] == 6

        anchor_data = tgd.Data(
            obj_id=obj_id,
            pos=torch.from_numpy(P_world).float(),
            flow_disp=torch.from_numpy(flow_disp).float(),
            axis_dir=torch.from_numpy(axis_dir).float().reshape(-1, 3, 1),
            axis_orig=torch.from_numpy(axis_orig).float().reshape(-1, 3, 1),
            rgb=torch.from_numpy(rgb)
            .float()
            .reshape(-1, rgb.shape[0], rgb.shape[1], 3),
        )
        pbullet.disconnect()

        return action_data, anchor_data  # type: ignore
