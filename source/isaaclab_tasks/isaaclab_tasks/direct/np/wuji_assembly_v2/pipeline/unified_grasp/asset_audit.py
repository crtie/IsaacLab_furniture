"""Asset and physics audit for v80 grasp objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .physics_profile import get_physics_profile, write_physics_profile_config
from .v80_reports import V80_PARTS, write_csv, write_json


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
NP_ROOT = Path(__file__).resolve().parents[3]
CHAIR_ASSET_DIR = NP_ROOT / "asset" / "chair"

PART_ASSETS: dict[str, dict[str, Any]] = {
    "Plug2": {
        "family": "Plug",
        "usd": CHAIR_ASSET_DIR / "plug.usd",
        "visual": CHAIR_ASSET_DIR / "plug.obj",
        "mass": 0.001,
        "static_friction": 0.75,
        "dynamic_friction": 0.75,
        "analytic_geometry": "small_cylinder_plug",
    },
    "Screw1": {
        "family": "Screw",
        "usd": CHAIR_ASSET_DIR / "screw_ree3.usd",
        "visual": CHAIR_ASSET_DIR / "screw_ree3.usd",
        "mass": 0.001,
        "static_friction": 0.75,
        "dynamic_friction": 0.75,
        "analytic_geometry": "small_cylinder_screw",
    },
    "Backrest": {
        "family": "Backrest",
        "usd": CHAIR_ASSET_DIR / "backrest3.usd",
        "visual": CHAIR_ASSET_DIR / "backrest.obj",
        "mass": 0.1,
        "static_friction": 0.75,
        "dynamic_friction": 0.75,
        "analytic_geometry": "broad_plate_edges",
    },
    "Rod": {
        "family": "Rod",
        "usd": CHAIR_ASSET_DIR / "rod.usd",
        "visual": CHAIR_ASSET_DIR / "rod.obj",
        "mass": 0.05,
        "static_friction": 0.75,
        "dynamic_friction": 0.75,
        "analytic_geometry": "thin_rod_axis",
    },
    "Frame": {
        "family": "Frame",
        "usd": CHAIR_ASSET_DIR / "frame_mirror.usd",
        "visual": CHAIR_ASSET_DIR / "frame_mirror.obj",
        "mass": 0.01,
        "static_friction": 0.75,
        "dynamic_friction": 0.75,
        "analytic_geometry": "frame_bars",
    },
}


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def audit_assets(
    run_dir: str | Path,
    *,
    parts: list[str] | None = None,
    physics_profile: str = "canonical",
) -> dict[str, Any]:
    profile = get_physics_profile(physics_profile)
    rows: list[dict[str, Any]] = []
    for part in parts or V80_PARTS:
        asset = PART_ASSETS.get(part, {})
        usd = Path(asset.get("usd", ""))
        visual = Path(asset.get("visual", ""))
        mass = float(asset.get("mass", 0.0)) * profile.mass_scale
        static_friction = float(asset.get("static_friction", 0.0)) * profile.static_friction_scale
        dynamic_friction = float(asset.get("dynamic_friction", 0.0)) * profile.dynamic_friction_scale
        rows.append(
            {
                "part_name": part,
                "asset_family": asset.get("family", part),
                "physics_profile": profile.name,
                "mass": mass,
                "inertia": "",
                "center_of_mass": "",
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
                "restitution": profile.restitution,
                "linear_damping": profile.linear_damping,
                "angular_damping": profile.angular_damping,
                "contact_offset": profile.contact_offset_m,
                "rest_offset": profile.rest_offset_m,
                "collision_mesh_available": usd.exists(),
                "collision_mesh_path": str(usd),
                "collision_mesh_size_bytes": _file_size(usd),
                "visual_mesh_path": str(visual),
                "visual_mesh_available": visual.exists(),
                "visual_mesh_size_bytes": _file_size(visual),
                "collision_vs_visual_sanity": "ok" if usd.exists() and visual.exists() else "missing_asset_file",
                "table_friction": 0.75 * profile.static_friction_scale,
                "solver_position_iterations": profile.solver_position_iterations,
                "solver_velocity_iterations": profile.solver_velocity_iterations,
                "analytic_geometry": asset.get("analytic_geometry", ""),
                "asset_audit_ok": bool(usd.exists()),
            }
        )
    run_path = Path(run_dir)
    csv_path = write_csv(run_path / "asset_physics_audit.csv", rows)
    json_path = write_json(run_path / "asset_physics_audit.json", rows)
    profile_path = write_physics_profile_config(run_path)
    return {
        "rows": rows,
        "asset_physics_audit_csv": str(csv_path),
        "asset_physics_audit_json": str(json_path),
        "physics_profile_config_json": str(profile_path),
    }
