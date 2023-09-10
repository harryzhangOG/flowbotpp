import os
import pickle
from typing import Dict, List, Optional, Protocol

import numpy as np
import torch
import torch.utils.data as td
import torch_geometric.data as tgd

import part_embedding.goal_inference.create_pm_goal_dataset as pgc
from part_embedding.datasets.pm.utils import SingleObjDataset, parallel_sample
from part_embedding.envs.render_sim import PMRenderEnv
from part_embedding.goal_inference.dataset import SEM_CLASS_DSET_PATH
from part_embedding.goal_inference.dataset_v2 import downsample_pcd_fps
from part_embedding.goal_inference.dset_utils import render_input_articulated
from part_embedding.taxpose4art.generate_art_training_data import transform_pcd


def __id_to_cat():
    id_to_cat = {}
    for cat, cat_dict in pgc.split_data["train"].items():
        for _, obj_ids in cat_dict.items():
            for obj_id in obj_ids:
                id_to_cat[obj_id] = cat
    return id_to_cat


def find_link_index_to_open_art(full_sem_dset, partsem, obj_id, object_dict):
    move_joints = None
    _id = obj_id.split("_")[0]
    for mode in full_sem_dset:
        if partsem in full_sem_dset[mode]:
            if _id in full_sem_dset[mode][partsem]:
                move_joints = full_sem_dset[mode][partsem][_id]

    assert move_joints is not None
    link_id = object_dict[f"{_id}_0"]["ind"]
    links_tomove = move_joints[link_id]

    return links_tomove


CATEGORIES = __id_to_cat()


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


class TAXPoseDataset(tgd.Dataset):
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
            open(f"part_embedding/taxpose4art/training_data/{dset_name}.pkl", "rb")
        )

        # Cache the environments. Only cache at the object level though.
        self.envs: Dict[str, PMRenderEnv] = {}

        # IDK what this really is....
        self.full_sem_dset = pickle.load(open(SEM_CLASS_DSET_PATH, "rb"))

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
        return os.path.join(self.root, f"taxpose_art_{self.dset_name}" + chunk)

    def process(self):
        if not self.use_processed:
            return

        else:
            # Run parallel sampling!
            get_data_args = [(f.split("_")[0], f.split("_")[1]) for f in self.obj_ids]
            parallel_sample(
                dset_cls=TAXPoseDataset,
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
        if obj_id not in self.envs:
            env = PMRenderEnv(tmp_id, self.raw_dir, camera_pos=[-2, 0, 2], gui=False)
            self.envs[obj_id] = env
        env = self.envs[obj_id]
        object_dict = pgc.all_objs[CATEGORIES[tmp_id].lower()]

        # Obtain entries from meta data
        curr_data_entry = self.generated_metadata[CATEGORIES[tmp_id]][tmp_id][
            int(obj_id.split("_")[1])
        ]
        start_ang = curr_data_entry["start"]
        end_ang = curr_data_entry["end"]
        transform = curr_data_entry["transformation"]

        # Next, check to see if the object needs to be opened in any way.
        partsem = object_dict[f"{tmp_id}_0"]["partsem"]
        if partsem != "none":
            links_tomove = find_link_index_to_open_art(
                self.full_sem_dset, partsem, obj_id, object_dict
            )
            env.set_specific_joints_angle(links_tomove, start_ang)

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
        if len(P_action_world) < action_pts_num:
            P_action_world = torch.cat(
                [
                    P_action_world,
                    P_action_world[: (action_pts_num - len(P_action_world))],
                ]
            )
        P_anchor_world = P_anchor_world[anchor_ixs]
        P_world = np.concatenate([P_action_world, P_anchor_world], axis=0)

        # Regenerate a mask.
        mask_act = torch.ones(len(P_action_world)).int()
        mask_anc = torch.zeros(len(P_anchor_world)).int()
        mask = torch.cat([mask_act, mask_anc])

        # Compute the transform from action object to goal.
        t_action_anchor = transform[:-1, -1]
        t_action_anchor = torch.from_numpy(t_action_anchor).float().unsqueeze(0)
        R_action_anchor = transform[:-1, :-1]
        R_action_anchor = torch.from_numpy(R_action_anchor).float().unsqueeze(0)

        # Compute the ground-truth flow.
        flow = np.zeros_like(P_world)
        flow2tf_res = transform_pcd(P_world[mask == 1], transform)
        flow[mask == 1] = flow2tf_res

        flow = torch.from_numpy(flow[mask == 1]).float()
        if len(flow) != len(P_action_world):
            breakpoint()

        # Assemble the data.
        action_data = tgd.Data(
            pos=P_action_world.float(),
            t_action_anchor=t_action_anchor.float(),
            R_action_anchor=R_action_anchor.float(),
            flow=flow.float(),
            loc=end_ang,
        )
        anchor_data = tgd.Data(
            obj_id=obj_id,
            pos=P_anchor_world.float(),
        )

        return action_data, anchor_data  # type: ignore
