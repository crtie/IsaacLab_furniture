$conda activate env_isaaclab


$ python scripts/environments/zero_agent.py --task Isaac-Franka-Chair-Direct-v0 --num_envs 1 --disable_fabric

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



./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/rod.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/rod.usd \
  --collision-approximation sdf \
  --mass 0.1


$ python scripts/environments/teleoperation/teleop_se3_agent_custom.py --task Isaac-Franka-Chair-Direct-v0 --num_envs 1 --teleop_device keyboard --sensitivity 10

$ python scripts/environments/teleoperation/teleop_se3_agent_custom.py --task Isaac-Factory-NutThread-Direct-v0 --num_envs 1 --teleop_device keyboard --sensitivity 10

$ python scripts/reinforcement_learning/rl_games/train.py --task Isaac-Factory-NutThread-Direct-v0 --num_envs 128 --headless

$ python scripts/reinforcement_learning/rl_games/play.py --task Isaac-Factory-NutThread-Direct-v0 --checkpoint logs/rl_games/Factory/test/nn/Factory.pth

TODO:
1. integrate the created joints into original articulation class

./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/screw_ree.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/screw_ree3.usd \
  --collision-approximation sdf \
  --mass 0.01

./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_re.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_re.usd \
  --collision-approximation sdf \
  --mass 1.0


./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/rod.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/rod6.usd \
  --make-instanceable \
  --collision-approximation sdf \
  --mass 0.01


./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/backrest.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/backrest3.usd \
  --collision-approximation sdf \
  --mass 0.01


./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_back.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_back.usd \
  --collision-approximation sdf \
  --mass 0.01


./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_back.urdf \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_back.usd \
  --collision-approximation sdf


./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/rod.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/rod.usd \
  --collision-approximation sdf \
  --mass 0.01

./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_back_rod.urdf \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_back_rod.usd \
  --collision-approximation sdf

./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/crtie/crtie/Manual2Skill2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/mesh/frame_back_rod.urdf \
  /home/crtie/crtie/Manual2Skill2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/mesh/frame_back_rod.usd \
  --collision-approximation sdf

./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/crtie/crtie/Manual2Skill2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/mesh/frame_back.urdf \
  /home/crtie/crtie/Manual2Skill2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/mesh/frame_back.usd \
  --collision-approximation sdf

./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/crtie/crtie/Manual2Skill2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/mesh/frame_back_rod_rod.urdf \
  /home/crtie/crtie/Manual2Skill2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/mesh/frame_back_rod_rod.usd \
  --collision-approximation sdf

./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_mirror.obj \
  /home/crtie/crtie/Manual2Skill2/chair_real/frame_mirror.usd \
  --collision-approximation sdf \
  --mass 0.01

