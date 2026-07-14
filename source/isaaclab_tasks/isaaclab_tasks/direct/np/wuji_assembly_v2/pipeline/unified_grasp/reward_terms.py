"""Reward terms for v80 unified grasp RL."""

from __future__ import annotations

from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def compute_reward_terms(
    contact_state: dict[str, Any],
    *,
    part_name: str,
    close_achieved: bool = False,
    support_gate_ok: bool = False,
    hold_success: bool = False,
    lift_success: bool = False,
    action_norm: float = 0.0,
    sticky_before_support: bool = False,
    object_write_attempted: bool = False,
    proxy_action_attempted: bool = False,
) -> dict[str, float]:
    force_count = _float(contact_state.get("effective_contact_count_force"))
    distance_count = _float(contact_state.get("effective_contact_count_distance"))
    motion = _float(contact_state.get("object_motion_before_contact_m"))
    penetration = _float(contact_state.get("penetration_depth_m"))
    force_peak = _float(contact_state.get("force_contact_peak_n"))
    contact_streak = _float(contact_state.get("force_contact_streak_steps"))
    object_displacement = _float(contact_state.get("object_displacement_m"))
    table_collision = _bool(contact_state.get("table_collision"))
    contact_sensor_available = _bool(contact_state.get("contact_sensor_available"))
    small_object = part_name in {"Plug2", "Screw1"}

    positive = 0.0
    positive += 1.5 * min(force_count, 4.0)
    positive += 2.0 if close_achieved else 0.0
    positive += 8.0 if support_gate_ok else 0.0
    positive += 3.0 if hold_success else 0.0
    positive += 5.0 if lift_success else 0.0
    if small_object and force_count >= 2:
        positive += 2.0
    if not small_object and force_count >= 2:
        positive += 1.5
    positive += 0.08 * min(contact_streak, 24.0)

    negative = 0.0
    negative += 12.0 * max(0.0, motion)
    negative += 2.5 if force_count <= 0 and close_achieved else 0.0
    negative += 3.0 if small_object and force_count == 1 else 0.0
    negative += 1500.0 * max(0.0, penetration)
    negative += 6.0 if table_collision else 0.0
    negative += 2.0 if distance_count > 0 and force_count <= 0 else 0.0
    negative += 0.15 * max(0.0, action_norm)
    negative += 2.0 * max(0.0, force_peak - 150.0) / 150.0
    negative += 8.0 * max(0.0, object_displacement - 0.03)
    negative += 20.0 if sticky_before_support else 0.0
    negative += 20.0 if object_write_attempted else 0.0
    negative += 20.0 if proxy_action_attempted else 0.0
    negative += 4.0 if not contact_sensor_available else 0.0

    return {
        "reward_total": positive - negative,
        "reward_positive": positive,
        "reward_negative": negative,
        "reward_force_contact": 1.5 * min(force_count, 4.0),
        "reward_contact_duration": 0.08 * min(contact_streak, 24.0),
        "reward_support": 8.0 if support_gate_ok else 0.0,
        "penalty_distance_only_contact": 2.0 if distance_count > 0 and force_count <= 0 else 0.0,
        "penalty_object_displacement": 8.0 * max(0.0, object_displacement - 0.03),
        "penalty_forbidden_action": 20.0 if (sticky_before_support or object_write_attempted or proxy_action_attempted) else 0.0,
    }
