import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask


def print_asset_info(gym, asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" %
          (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))


def print_actor_info(gym, env, actor_handle):

    name = gym.get_actor_name(env, actor_handle)

    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    body_dict = gym.get_actor_rigid_body_dict(env, actor_handle)

    joint_names = gym.get_actor_joint_names(env, actor_handle)
    joint_dict = gym.get_actor_joint_dict(env, actor_handle)

    dof_names = gym.get_actor_dof_names(env, actor_handle)
    dof_dict = gym.get_actor_dof_dict(env, actor_handle)
    dof_props = gym.get_actor_dof_properties(env, actor_handle)

    print()
    print("===== Actor: %s =======================================" % name)

    print("\nBodies")
    print(body_names)
    print(body_dict)

    print("\nJoints")
    print(joint_names)
    print(joint_dict)

    print("\nDegrees Of Freedom (DOFs)")
    print(dof_names)
    print(dof_dict)
    print()

    for i, name in enumerate(dof_names):
        print(f"\nDOF {name} properties")
        for k in ["hasLimits", "lower", "upper", "driveMode", "stiffness", "damping", "velocity", "effort", "friction",
                  "armature"]:
            print(f"\t{k}: {dof_props[k][i]}")

    # Get body state information
    body_states = gym.get_actor_rigid_body_states(
        env, actor_handle, gymapi.STATE_ALL)

    # Print some state slices
    print("\nPoses from Body State:")
    print(body_states['pose'])          # print just the poses

    print("\nVelocities from Body State:")
    print(body_states['vel'])          # print just the velocities
    print()

    # iterate through bodies and print name and position
    body_positions = body_states['pose']['p']
    for i in range(len(body_names)):
        print("Body '%s' has position" % body_names[i], body_positions[i])

    print("\nDOF states:")

    # get DOF states
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    # print some state slices
    # Print all states for each degree of freedom
    print(dof_states)
    print()

    # iterate through DOFs and print name and position
    dof_positions = dof_states['pos']
    for i in range(len(dof_names)):
        print("DOF '%s' has position" % dof_names[i], dof_positions[i])

def print_actuators(gym, env, actor_handle):
    print("\n===== Actuators =======================================")

    # get all actuators
    n_actuators = gym.get_actor_actuator_count(env, actor_handle)
    props = gym.get_actor_actuator_properties(env, actor_handle)

    for i in range(n_actuators):
        name = gym.get_actor_actuator_name(env, actor_handle, i)
        prop = props[i]

        print(f"\nActor {name} properties")
        print(f"\tcontrol_limited {prop.control_limited}")
        print(f"\tlower_control_limit {prop.lower_control_limit}")
        print(f"\tupper_control_limit {prop.upper_control_limit}")
        print(f"\tforce_limited {prop.force_limited}")
        print(f"\tlower_force_limit {prop.lower_force_limit}")
        print(f"\tupper_force_limit {prop.upper_force_limit}")
        print(f"\tmotor_effort {prop.motor_effort}")
        print(f"\tkp {prop.kp}")
        print(f"\tkv {prop.kv}")



class PointMass(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

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
        self.num_dof = self.gym.get_asset_dof_count(pointmass_asset)

        # for debugging
        print_asset_info(self.gym, pointmass_asset, "Pointmass")

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
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, pointmass_handle, dof_props)

            # for debugging
            print_actor_info(self.gym, env_ptr, pointmass_handle)
            print_actuators(self.gym, env_ptr, pointmass_handle)

            self.envs.append(env_ptr)
            self.pointmass_handles.append(pointmass_handle)

    def compute_reward(self):
        x_pos = self.obs_buf[:, 0]
        x_vel = self.obs_buf[:, 1]
        y_pos = self.obs_buf[:, 2]
        y_vel = self.obs_buf[:, 3]

        self.rew_buf[:], self.reset_buf[:] = compute_pointmass_reward(x_pos, x_vel, y_pos, y_vel, self.reset_buf,
                                                                      self.progress_buf, self.max_episode_length)

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
        # positions = 0.1 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)  # +/-0.05
        positions = 0. * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)  # +/-0.05
        # velocities = 0.1 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5) # +/-0.05
        velocities = 0. * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5) # +/-0.05

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions: torch.Tensor):
        actions_tensor = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        # @TODO get motor limits and clip actions
        # @TODO multiply actions by gear
        # @TODO get tendon factors and use coefficients to compute effort of each joint

        actions_tensor[::self.num_dof] = actions.to(self.device) * 0.1
        forces = gymtorch.unwrap_tensor(actions_tensor)
        print("\n\n###### PRE PHYSICS STEP")
        print(f"Forces {actions}")
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        print("\n\n###### POST PHYSICS STEP")

        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()

        for i in range(self.num_dof):
            print(f"DOF {i}")
            print(f"\tpos: {self.dof_pos[0, i].squeeze().cpu()}")
            print(f"\tvel: {self.dof_vel[0, i].squeeze().cpu()}")

        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_pointmass_reward(x_pos, x_vel, y_pos, y_vel, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # @TODO use correct reward function
    reward = 1.0 - x_pos * x_pos - y_pos * y_pos

    # adjust reward for reset agents
    reset = progress_buf >= max_episode_length - 1

    return reward, reset
