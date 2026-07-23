"""IsaacLab builders for the declarative SharpaWave asset contract.

The dependency-light morphology contract lives in :mod:`sharpawave`.  This
module is the only place that imports IsaacLab classes, which keeps schema
tests runnable without Kit and prevents unrelated actuator settings from leaking
into the SharpaWave configuration.
"""

from __future__ import annotations

from typing import Any

from .sharpawave import (
    SHARPAWAVE_VARIANT_FLOATING,
    SHARPAWAVE_VARIANT_PEG_FIXEDROT,
    SharpaWaveVariantConfig,
    get_sharpawave_variant,
)


def build_sharpawave_articulation_cfg(
    variant: str = SHARPAWAVE_VARIANT_FLOATING,
    *,
    prim_path: str = "/World/envs/env_.*/Robot",
    activate_contact_sensors: bool = True,
    translation_stiffness: float = 3000.0,
    translation_damping: float = 100.0,
    rotation_stiffness: float = 20.0,
    rotation_damping: float = 0.5,
    finger_stiffness: float = 20.0,
    finger_damping: float = 0.5,
) -> Any:
    """Build an ``ArticulationCfg`` without reading the vendor absolute paths.

    Defaults reproduce the supplied URDF-conversion ``config.yaml`` position
    drives: xyz stiffness/damping 3000/100 and every other ``right_.*`` joint
    20/0.5. They are asset-derived simulation parameters, not hardware or grasp
    calibration, and remain explicit/overrideable in run manifests.
    """

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    config = get_sharpawave_variant(variant)
    base_names = tuple(config.base_joint_names)
    translation_names = tuple(name for name in base_names if name in {"right_x_joint", "right_y_joint", "right_z_joint"})
    rotation_names = tuple(name for name in base_names if name not in translation_names)
    finger_names = tuple(config.finger_joint_names)
    base_effort = {spec.name: float(spec.effort_limit) for spec in config.joint_specs if spec.name in base_names}
    base_velocity = {spec.name: float(spec.velocity_limit) for spec in config.joint_specs if spec.name in base_names}
    finger_effort = {spec.name: float(spec.effort_limit) for spec in config.joint_specs if spec.name in finger_names}
    finger_velocity = {spec.name: float(spec.velocity_limit) for spec in config.joint_specs if spec.name in finger_names}

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(config.usd_path),
            activate_contact_sensors=bool(activate_contact_sensors),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=2.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=10,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "translation": ImplicitActuatorCfg(
                joint_names_expr=list(translation_names),
                effort_limit_sim={name: base_effort[name] for name in translation_names},
                velocity_limit_sim={name: base_velocity[name] for name in translation_names},
                stiffness=float(translation_stiffness),
                damping=float(translation_damping),
                friction=0.0,
                armature=0.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=list(finger_names),
                effort_limit_sim=finger_effort,
                velocity_limit_sim=finger_velocity,
                stiffness=float(finger_stiffness),
                damping=float(finger_damping),
                friction=0.0,
                armature=0.0,
            ),
        } | ({
            "rotation": ImplicitActuatorCfg(
                joint_names_expr=list(rotation_names),
                effort_limit_sim={name: base_effort[name] for name in rotation_names},
                velocity_limit_sim={name: base_velocity[name] for name in rotation_names},
                stiffness=float(rotation_stiffness),
                damping=float(rotation_damping),
                friction=0.0,
                armature=0.0,
            )
        } if rotation_names else {}),
        soft_joint_pos_limit_factor=1.0,
    ).replace(prim_path=prim_path)


def build_sharpawave_contact_sensor_cfgs(
    variant: str = SHARPAWAVE_VARIANT_FLOATING,
    *,
    prim_prefix: str = "/World/envs/env_.*/Robot",
) -> tuple[Any, ...]:
    """Return contact sensor configs for elastomer bodies.

    The preconverted USD layers do not guarantee ``PhysxContactReportAPI``;
    callers must run a canary and treat unavailable sensors as a capability
    error rather than interpreting missing data as no contact.
    """

    from isaaclab.sensors import ContactSensorCfg

    config = get_sharpawave_variant(variant)
    return tuple(
        ContactSensorCfg(
            prim_path=f"{prim_prefix}/{item.contact_body}",
            update_period=0.0,
            history_length=3,
            track_air_time=True,
            force_threshold=0.05,
        )
        for item in config.fingertip_specs
    )


__all__ = [
    "build_sharpawave_articulation_cfg",
    "build_sharpawave_contact_sensor_cfgs",
]
