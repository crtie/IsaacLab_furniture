$conda activate env_isaaclab


$ python scripts/environments/zero_agent.py --task Isaac-Franka-Lego1-Direct-v0 --num_envs 1 --disable_fabric

$ ./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_test.usd \
  --make-instanceable \
  --collision-approximation sdf \
  --mass 1.0


#### How to convert mesh to usd and import into isaac lab
1. convert to usd
$ ./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_sdf_ar.usd \
  --make-instanceable \
  --collision-approximation sdf \
  --mass 1.0

2. open an empty stage, import the usd, select the **root prim** , add-physics-articulation root; and select the mesh, change sdf parameter (increase sdf resolution)
3. export the prim
4. import like this
  '''
    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.25, 0.80), rot=(0.707, 0.707, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
  '''
  !NOTE: articulation_props is necessary



$ python scripts/environments/teleoperation/teleop_se3_agent_custom.py --task Isaac-Franka-Lego1-Direct-v0 --num_envs 1 --teleop_device keyboard --sensitivity 10


$ python scripts/reinforcement_learning/rl_games/train.py --task Isaac-Factory-NutThread-Direct-v0 --num_envs 128 --headless

$ python scripts/reinforcement_learning/rl_games/play.py --task Isaac-Factory-NutThread-Direct-v0 --checkpoint logs/rl_games/Factory/test/nn/Factory.pth


  ./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/zhujinxuan/IsaacLab_furniture/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/mesh/chair_all.urdf \
  /home/zhujinxuan/IsaacLab_furniture/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/mesh/chair_all.usd \
  --collision-approximation sdf


./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/zhujinxuan/IsaacLab_furniture/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/lego/hand.obj \
  /home/zhujinxuan/IsaacLab_furniture/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/lego/hand.usd \
  --collision-approximation sdf \
  --mass 0.01


!!! when importing fixed asset (represented as articulation), turn the 'articulation_props' in convert_mesh.py to None, and manually add artuculation_root in GUI
!!! when importing held asset (RigidObject), turn on the 'articulation_props' in convert_mesh.py