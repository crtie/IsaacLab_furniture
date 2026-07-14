"""Deterministic v82 policy-to-Isaac action mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - runtime-specific
    import torch
except Exception:  # pragma: no cover
    torch = None


POLICY_ACTION_DIM = 16
DEFAULT_ISAAC_ACTION_DIM = 26
WRIST_COLS = list(range(0, 6))
CLOSE_POLICY_COLS = list(range(6, 16))
CLOSE_ISAAC_COLS = list(range(16, 26))


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _env_action_width(env: Any, default: int = DEFAULT_ISAAC_ACTION_DIM) -> int:
    action_space = getattr(env, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if shape:
        return int(shape[-1])
    base = _base_env(env)
    return int(getattr(base, "num_actions", default) or default)


def _close_sign_values(env: Any, *, device: Any = None) -> tuple[Any, str]:
    base = _base_env(env)
    sign = getattr(base, "dex_hand_close_action_sign", None)
    if sign is None:
        values = [1.0] * len(CLOSE_ISAAC_COLS)
        if torch is not None:
            values = torch.ones((len(CLOSE_ISAAC_COLS),), dtype=torch.float32, device=device)
        return values, "default_positive"
    try:
        sliced = sign[10:20]
        if torch is not None and hasattr(sliced, "to"):
            return sliced.to(device=device, dtype=torch.float32), "dex_hand_close_action_sign[10:20]"
        if torch is not None:
            return torch.tensor(list(sliced), dtype=torch.float32, device=device), "dex_hand_close_action_sign[10:20]"
        return [float(item) for item in list(sliced)], "dex_hand_close_action_sign[10:20]"
    except Exception:
        values = [1.0] * len(CLOSE_ISAAC_COLS)
        if torch is not None:
            values = torch.ones((len(CLOSE_ISAAC_COLS),), dtype=torch.float32, device=device)
        return values, "default_positive_after_sign_read_failure"


@dataclass
class UnifiedActionMapper:
    """Map v80/v81 16D residual policy actions to the 26D Isaac action contract."""

    policy_action_dim: int = POLICY_ACTION_DIM

    def map_batch(
        self,
        env: Any,
        policy_actions: Any,
        *,
        device: Any = None,
        env_indices: list[int] | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        isaac_action_dim = _env_action_width(env)
        if torch is None:
            policy_rows = self._rows_as_lists(policy_actions)
            mapped_rows, audits = self._map_list_rows(env, policy_rows, isaac_action_dim, env_indices=env_indices)
            return mapped_rows, audits
        if hasattr(policy_actions, "detach"):
            policy_tensor = policy_actions.to(device=device, dtype=torch.float32)
        else:
            policy_tensor = torch.tensor(policy_actions, dtype=torch.float32, device=device)
        if policy_tensor.ndim == 1:
            policy_tensor = policy_tensor.unsqueeze(0)
        if policy_tensor.shape[-1] < self.policy_action_dim:
            pad = torch.zeros(
                (policy_tensor.shape[0], self.policy_action_dim - policy_tensor.shape[-1]),
                dtype=torch.float32,
                device=policy_tensor.device,
            )
            policy_tensor = torch.cat([policy_tensor, pad], dim=-1)
        policy_tensor = torch.clamp(policy_tensor[:, : self.policy_action_dim], -1.0, 1.0)
        mapped = torch.zeros((policy_tensor.shape[0], isaac_action_dim), dtype=torch.float32, device=policy_tensor.device)
        wrist_width = min(6, isaac_action_dim)
        mapped[:, :wrist_width] = policy_tensor[:, :wrist_width]
        sign, sign_source = _close_sign_values(env, device=policy_tensor.device)
        close_cols = [col for col in CLOSE_ISAAC_COLS if col < isaac_action_dim]
        if close_cols:
            close_values = policy_tensor[:, CLOSE_POLICY_COLS[: len(close_cols)]]
            close_sign = sign[: len(close_cols)] if hasattr(sign, "__getitem__") else sign
            mapped[:, close_cols] = close_values * close_sign.reshape(1, -1)
        audits = self._audit_rows(
            policy_tensor=policy_tensor,
            mapped=mapped,
            isaac_action_dim=isaac_action_dim,
            close_cols=close_cols,
            sign_source=sign_source,
            env_indices=env_indices,
        )
        return mapped, audits

    def _rows_as_lists(self, actions: Any) -> list[list[float]]:
        if not actions:
            return [[0.0] * self.policy_action_dim]
        if isinstance(actions[0], (int, float)):
            actions = [actions]
        rows = []
        for row in actions:
            raw = [max(-1.0, min(1.0, float(value))) for value in list(row)[: self.policy_action_dim]]
            raw.extend([0.0] * (self.policy_action_dim - len(raw)))
            rows.append(raw)
        return rows

    def _map_list_rows(
        self,
        env: Any,
        policy_rows: list[list[float]],
        isaac_action_dim: int,
        *,
        env_indices: list[int] | None,
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:
        sign, sign_source = _close_sign_values(env)
        sign_values = [float(item) for item in list(sign)]
        close_cols = [col for col in CLOSE_ISAAC_COLS if col < isaac_action_dim]
        mapped_rows = []
        for row in policy_rows:
            mapped = [0.0] * isaac_action_dim
            for src, dst in enumerate(WRIST_COLS[: min(6, isaac_action_dim)]):
                mapped[dst] = row[src]
            for offset, col in enumerate(close_cols):
                mapped[col] = row[CLOSE_POLICY_COLS[offset]] * sign_values[offset]
            mapped_rows.append(mapped)
        audits = []
        for row_index, mapped in enumerate(mapped_rows):
            commanded = max([0.0, *[abs(mapped[col]) for col in close_cols]]) > 1.0e-6
            audits.append(
                {
                    "env_index": env_indices[row_index] if env_indices and row_index < len(env_indices) else row_index,
                    "policy_action_dim": self.policy_action_dim,
                    "isaac_action_dim": isaac_action_dim,
                    "mapped_close_cols": ",".join(str(col) for col in close_cols),
                    "close_sign_source": sign_source,
                    "metric_close_dof_commanded": bool(commanded),
                    "max_abs_close_command": max([0.0, *[abs(mapped[col]) for col in close_cols]]),
                }
            )
        return mapped_rows, audits

    def _audit_rows(
        self,
        *,
        policy_tensor: Any,
        mapped: Any,
        isaac_action_dim: int,
        close_cols: list[int],
        sign_source: str,
        env_indices: list[int] | None,
    ) -> list[dict[str, Any]]:
        audits = []
        for row_index in range(int(mapped.shape[0])):
            max_close = 0.0
            if close_cols:
                max_close = float(torch.max(torch.abs(mapped[row_index, close_cols])).detach().cpu().item())
            audits.append(
                {
                    "env_index": env_indices[row_index] if env_indices and row_index < len(env_indices) else row_index,
                    "policy_action_dim": int(policy_tensor.shape[-1]),
                    "isaac_action_dim": int(isaac_action_dim),
                    "mapped_close_cols": ",".join(str(col) for col in close_cols),
                    "close_sign_source": sign_source,
                    "metric_close_dof_commanded": bool(max_close > 1.0e-6),
                    "max_abs_close_command": max_close,
                }
            )
        return audits
