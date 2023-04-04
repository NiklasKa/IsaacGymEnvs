import os

import numpy as np
import torch
from isaacgym import gymtorch, gymapi
from typing import Tuple, List, Dict

from .base.vec_task import VecTask
from ..utils.torch_jit_utils import *


class PointMass(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 2

        self.targets = self.cfg["env"].get("targets", {})
        self.targets = list(self.targets.values())

        self.action_cost = self.cfg["env"].get("actionCost", 0.0) if self.cfg["env"].get("actionCost", 0.0) else 0.0

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_actuators, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_actuators, 2)[..., 1]

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "mjcf/point_mass.xml"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        pointmass_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_actuators = self.gym.get_asset_actuator_count(pointmass_asset)
        self.num_dof = self.gym.get_asset_dof_count(pointmass_asset)

        # get motor efforts and control range
        actuator_props = self.gym.get_asset_actuator_properties(pointmass_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self._motor_effort = to_torch(motor_efforts, device=self.device)

        ctrl_min = [prop.lower_control_limit for prop in actuator_props]
        self._ctrl_min = to_torch(ctrl_min, device=self.device)

        ctrl_max = [prop.upper_control_limit for prop in actuator_props]
        self._ctrl_max = to_torch(ctrl_max, device=self.device)

        # get tendon to dof mapping
        self._tendon_to_joint = torch.zeros((self.num_actuators, self.num_dof), device=self.device)

        for i in range(self.gym.get_asset_tendon_count(pointmass_asset)):
            joint_coefficients = self.gym.get_asset_fixed_tendon_joint_coefficients(pointmass_asset, i)
            for j, coef in enumerate(joint_coefficients):
                joint_name = self.gym.get_asset_fixed_tendon_joint_name(pointmass_asset, i, j)
                joint_index = self.gym.find_asset_joint_index(pointmass_asset, joint_name)

                self._tendon_to_joint[i, joint_index] = coef

        # asset is rotated z-up by default, no additional rotations needed
        pose = gymapi.Transform()
        pose.p.z = 0.0
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.pointmass_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            pointmass_handle = self.gym.create_actor(env_ptr, pointmass_asset, pose, "pointmass", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, pointmass_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT
            self.gym.set_actor_dof_properties(env_ptr, pointmass_handle, dof_props)

            self.envs.append(env_ptr)
            self.pointmass_handles.append(pointmass_handle)

    def compute_reward(self):
        x_pos = self.obs_buf[:, 0]
        y_pos = self.obs_buf[:, 2]

        self.rew_buf[:], self.reset_buf[:] = compute_pointmass_reward(x_pos, y_pos, self.actions, self.reset_buf,
                                                                      self.progress_buf, self.targets, self.action_cost, self.max_episode_length)

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.02 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)  # +/-0.01
        velocities = 0.02 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5) # +/-0.01

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions: torch.Tensor):
        # clip actions, apply motor effort and map to tendons
        clamped_actions = torch.clamp(actions.clone().to(self.device), min=self._ctrl_min, max=self._ctrl_max)
        self.actions = (clamped_actions * self._motor_effort) @ self._tendon_to_joint

        # map tendon to joint
        force_tensor = gymtorch.unwrap_tensor(self.actions)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()

        self.compute_reward()


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_pointmass_reward(x_pos, y_pos, actions, reset_buf, progress_buf, targets, action_cost_factor, max_episode_length):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, float]], float, float) -> Tuple[torch.Tensor, torch.Tensor]

    # compute dist to all targets if specified
    target_reward = torch.zeros_like(reset_buf).float()
    for target in targets:
        target_size = target["size"]
        lower_bound = 0.0
        upper_bound = target_size
        reward_margin = 0.1

        dist = torch.sqrt((x_pos - target["xpos"])**2 + (y_pos - target["ypos"])**2)
        in_bounds = torch.logical_and(lower_bound <= dist, dist <= upper_bound)

        # compute sigmoid reward outside bounds
        # d = torch.where(dist < lower_bound, lower_bound - dist, dist - upper_bound) / target_size
        # scale = torch.sqrt(-2. * torch.log(reward_margin))
        # sigmoid_d =  torch.exp(-0.5 * (d * scale) ** 2)

        # value = torch.where(in_bounds, 1.0, sigmoid_d)
        value = torch.where(in_bounds, 1.0, 0.0)
        target_reward += target["reward"] * value

    # compute action cost
    action_cost = action_cost_factor * torch.sum(actions**2, dim=1) / actions.shape[1]
    reward = target_reward - action_cost

    # adjust reward for reset agents
    reset = torch.where(progress_buf >= (max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return reward, reset

