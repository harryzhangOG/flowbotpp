import numpy as np
import pybullet as p

from part_embedding.envs.floating_vacuum_gripper import FloatingSuctionGripper
from part_embedding.envs.render_sim import PMRenderEnv


class PMSuctionEnv(PMRenderEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_gripper_to_env(self):
        self.gripper = FloatingSuctionGripper(self.obj_id, self.client_id)
        self.gripper.set_pose(
            [-1, 0.6, 0.8], p.getQuaternionFromEuler([0, np.pi / 2, 0])
        )

    def remove_gripper_from_env(self):
        p.removeBody(self.gripper.body_id)
        p.removeBody(self.gripper.base_id)

    def teleport_gripper_to(self, pos, render=False):
        imgs = []
        rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.render(True)
        imgs.append(rgb)
        self.gripper.set_pose(pos)
        for i in range(1000):
            p.stepSimulation(self.client_id)
            if render:
                (
                    rgb,
                    depth,
                    seg,
                    P_cam,
                    P_world,
                    P_rgb,
                    pc_seg,
                    segmap,
                ) = self.render(True)
                imgs.append(rgb)
        rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.render(True)
        imgs.append(rgb)
        return imgs

    def move_gripper_vel_to(self, dest_pt, render=False):
        imgs = []
        rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.render(True)
        pos, ori = p.getBasePositionAndOrientation(
            self.gripper.body_id, physicsClientId=self.client_id
        )
        vel = np.array(dest_pt) - np.array(pos)
        imgs.append(rgb)
        for i in range(1, 100):
            self.gripper.set_pose(np.array(pos) + 0.01 * i * vel)
            p.stepSimulation(self.client_id)
        rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.render(True)
        imgs.append(rgb)

        return imgs

    def move_gripper_traj(self, traj, render=False):
        imgs = self.move_gripper_vel_to(traj[0], render=False)

        pos, ori = p.getBasePositionAndOrientation(
            self.gripper.body_id, physicsClientId=self.client_id
        )
        for i in range(1, len(traj)):
            vel = traj[i] - traj[i - 1]
            rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.render(True)
            imgs.append(rgb)
            for j in range(1, 10):
                self.gripper.set_pose(traj[i - 1] + 0.1 * j * vel)
                p.stepSimulation(self.client_id)

        return imgs

    def begin_suction(self, contact_point, link, render=False):
        imgs = []
        rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.render(True)
        imgs.append(rgb)
        self.move_gripper_vel_to(contact_point)
        pos, ori = p.getBasePositionAndOrientation(
            self.gripper.body_id, physicsClientId=self.client_id
        )
        dist = np.linalg.norm(np.array(pos) - np.array(contact_point))
        self.gripper.activate_general(0.1, link)
        rgb, depth, seg, P_cam, P_world, P_rgb, pc_seg, segmap = self.render(True)
        imgs.append(rgb)
        return imgs
