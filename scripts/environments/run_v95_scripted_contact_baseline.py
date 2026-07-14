"""Run a deterministic Screw1 contact baseline without entering the v95 gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
for rel in ("source/isaaclab", "source/isaaclab_tasks", "source/isaaclab_assets", "source/isaaclab_rl"):
    path = str(REPO_ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)
PACKAGE_ROOT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/wuji_assembly_v2"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _inotify_watch_consumers_top(limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proc in Path("/proc").glob("[0-9]*"):
        pid = proc.name
        count = 0
        try:
            for fdinfo in (proc / "fdinfo").glob("*"):
                try:
                    with fdinfo.open("r", encoding="utf-8", errors="ignore") as stream:
                        count += sum(1 for line in stream if line.startswith("inotify"))
                except Exception:
                    continue
        except Exception:
            continue
        if count <= 0:
            continue
        try:
            comm = (proc / "comm").read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            comm = ""
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        except Exception:
            cmdline = ""
        rows.append({"watch_count": count, "pid": int(pid), "comm": comm, "cmdline": cmdline[:240]})
    rows.sort(key=lambda row: int(row.get("watch_count", 0)), reverse=True)
    return rows[: int(limit)]


def _inotify_watch_usage() -> dict[str, Any]:
    rows = _inotify_watch_consumers_top(limit=1000000)
    max_watches = _read_int_file("/proc/sys/fs/inotify/max_user_watches")
    used = sum(int(row.get("watch_count", 0) or 0) for row in rows)
    free_est = max(0, max_watches - used) if max_watches > 0 else 0
    return {
        "inotify_max_user_watches": max_watches,
        "inotify_watch_count_total": used,
        "inotify_watch_count_top_sum": sum(int(row.get("watch_count", 0) or 0) for row in rows[:10]),
        "inotify_watch_free_estimate": free_est,
        "inotify_watch_consumers_top": rows[:10],
    }


def _read_int_file(path: str, default: int = 0) -> int:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except Exception:
        return int(default)


def _run_text_cmd(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:
        return f"{type(exc).__name__}:{exc}"


def _run_text_cmd_result(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "cmd": cmd,
            "returncode": int(proc.returncode),
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": f"{type(exc).__name__}:{exc}"}


def _inotify_recovery_target(max_watches: int, used: int) -> int:
    target = max(1_048_576, int(used) + 200_000)
    step = 262_144
    rounded = ((target + step - 1) // step) * step
    return max(int(max_watches) * 2 if int(max_watches) >= 1_048_576 else 1_048_576, rounded)


def _du(path: str) -> str:
    return _run_text_cmd(["du", "-sh", path])


def _latest_kit_errno28() -> dict[str, Any]:
    log_root = Path.home() / "miniconda3/envs/isaac/lib/python3.10/site-packages/omni/logs/Kit/Isaac-Sim/4.5"
    try:
        logs = sorted(log_root.glob("kit_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    except Exception:
        logs = []
    if not logs:
        return {"latest_kit_log_path": "", "errno28_seen": False, "errno28_count": 0, "errno28_first_lines": []}
    path = logs[0]
    lines: list[str] = []
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                if "errno=28" in line or "No space left on device" in line:
                    count += 1
                    if len(lines) < 10:
                        lines.append(line.strip())
    except Exception as exc:
        return {
            "latest_kit_log_path": str(path),
            "errno28_seen": False,
            "errno28_count": 0,
            "errno28_first_lines": [f"{type(exc).__name__}:{exc}"],
        }
    return {
        "latest_kit_log_path": str(path),
        "errno28_seen": bool(count > 0),
        "errno28_count": int(count),
        "errno28_first_lines": lines,
    }


def _early_health_snapshot() -> dict[str, Any]:
    kit = _latest_kit_errno28()
    inotify_usage = _inotify_watch_usage()
    try:
        usage = shutil.disk_usage(REPO_ROOT)
        disk_usage: dict[str, Any] = {"total": usage.total, "used": usage.used, "free": usage.free}
    except Exception as exc:
        disk_usage = {"error": f"{type(exc).__name__}:{exc}"}
    tmp_probe = _temp_write_probe(REPO_ROOT / "debug_runs")
    return {
        "df_h": _run_text_cmd(["df", "-h"]),
        "du_debug_runs": _du(str(REPO_ROOT / "debug_runs")),
        "du_ov_cache": _du(os.path.expanduser("~/.cache/ov")),
        "du_tmp": _du("/tmp"),
        "shutil_disk_usage_repo": disk_usage,
        "temp_write_probe": tmp_probe,
        **inotify_usage,
        "inotify_max_user_instances": Path("/proc/sys/fs/inotify/max_user_instances").read_text(encoding="utf-8").strip()
        if Path("/proc/sys/fs/inotify/max_user_instances").is_file()
        else "",
        "inotify_max_queued_events": Path("/proc/sys/fs/inotify/max_queued_events").read_text(encoding="utf-8").strip()
        if Path("/proc/sys/fs/inotify/max_queued_events").is_file()
        else "",
        "historical_latest_kit_errno28_seen": bool(kit.get("errno28_seen")),
        **kit,
    }


def _temp_write_probe(directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".v2_temp_write_probe"
    try:
        path.write_text("ok\n", encoding="utf-8")
        path.unlink(missing_ok=True)
        return {"ok": True, "path": str(path), "error": ""}
    except Exception as exc:
        return {"ok": False, "path": str(path), "error": f"{type(exc).__name__}:{exc}"}


def _early_v2_health_blocker(args: argparse.Namespace) -> dict[str, Any]:
    health_snapshot = _early_health_snapshot()
    max_watches = int(health_snapshot.get("inotify_max_user_watches", 0) or 0)
    used = int(health_snapshot.get("inotify_watch_count_total", 0) or 0)
    free_est = int(health_snapshot.get("inotify_watch_free_estimate", 0) or 0)
    consumers = list(health_snapshot.get("inotify_watch_consumers_top", []) or [])
    disk_free = int(dict(health_snapshot.get("shutil_disk_usage_repo", {}) or {}).get("free", 0) or 0)
    temp_ok = bool(dict(health_snapshot.get("temp_write_probe", {}) or {}).get("ok", False))
    inotify_blocked = bool(max_watches > 0 and (free_est < 50000 or used >= int(0.90 * max_watches)))
    disk_blocked = bool(disk_free < 2_000_000_000 or not temp_ok)
    sysctl_result: dict[str, Any] = {}
    health_after = health_snapshot
    host_action_required = False
    recovery_target = _inotify_recovery_target(max_watches, used)
    if inotify_blocked:
        sysctl_result = _run_text_cmd_result(
            ["sudo", "-n", "sysctl", "-w", f"fs.inotify.max_user_watches={recovery_target}"]
        )
        health_after = _early_health_snapshot()
        max_watches = int(health_after.get("inotify_max_user_watches", 0) or 0)
        used = int(health_after.get("inotify_watch_count_total", 0) or 0)
        free_est = int(health_after.get("inotify_watch_free_estimate", 0) or 0)
        consumers = list(health_after.get("inotify_watch_consumers_top", []) or [])
        inotify_blocked = bool(max_watches > 0 and (free_est < 50000 or used >= int(0.90 * max_watches)))
        host_action_required = bool(inotify_blocked and int(sysctl_result.get("returncode", -1)) != 0)
        recovery_target = _inotify_recovery_target(max_watches, used)
    should_block = bool(inotify_blocked or disk_blocked)
    blocker = ""
    if inotify_blocked:
        blocker = "current_inotify_watch_capacity_exhausted"
    elif disk_blocked:
        blocker = "current_filesystem_write_or_free_space_blocked"
    result_class = "HEALTH_OK"
    if host_action_required:
        result_class = "HOST_ACTION_REQUIRED"
    elif should_block:
        result_class = "HOST_RESOURCE_PREFLIGHT_BLOCKED"
    return {
        "baseline": "screw1_grasp_baseline_v2",
        "target_part": str(getattr(args, "part", "Screw1")),
        "v2_phase": str(getattr(args, "v2_phase", "full")),
        "target_env_index": -1,
        "training_locked": True,
        "object_ready": False,
        "grasp_solved": False,
        "success_claimed": False,
        "sticky_used": False,
        "proxy_success_used": False,
        "forced_pose_success_used": False,
        "distance_only_success_used": False,
        "unfiltered_only_success_used": False,
        "root_pose_writes_after_reset_used": False,
        "close_executed": False,
        "lift_executed": False,
        "health_executed": True,
        "health_pre_app_launcher": True,
        "health_before": health_snapshot,
        "health_after": health_after,
        "health_result_class": result_class,
        "health_current_resource_blocked": bool(should_block),
        "health_historical_kit_errno28_blocks_launch": False,
        "host_health_sysctl_attempted": bool(sysctl_result),
        "host_health_sysctl_target_max_user_watches": recovery_target,
        "host_health_sysctl_result": sysctl_result,
        "errno28_seen_in_local_logs": False,
        "historical_latest_kit_errno28_seen": bool(health_snapshot.get("historical_latest_kit_errno28_seen")),
        "inotify_max_user_watches": max_watches,
        "inotify_watch_count_total": used,
        "inotify_watch_count_top_sum": int(health_after.get("inotify_watch_count_top_sum", used) or used),
        "inotify_watch_free_estimate": free_est,
        "host_resource_inotify_blocked": inotify_blocked,
        "host_resource_disk_blocked": disk_blocked,
        "host_resource_temp_write_ok": temp_ok,
        "inotify_watch_consumers_top": consumers,
        "result_class": result_class,
        "final_allowed_result_class": True,
        "blocker": blocker,
        "errno28_prevented_by_preflight": bool(should_block),
        "host_resource_suggested_action": (
            f"sudo sysctl -w fs.inotify.max_user_watches={recovery_target}"
            if inotify_blocked
            else ""
        ),
    }


def _early_v2_gpu_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Capture current GPU/Isaac process state before launching Kit."""

    nvidia = _run_text_cmd_result(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    compute = _run_text_cmd_result(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    proc = _run_text_cmd_result(["bash", "-lc", "ps -eo pid,ppid,cmd | grep -E \"isaac|kit|python\" | grep -v grep"])
    gpu_rows: list[dict[str, Any]] = []
    min_free_mib: int | None = None
    for line in str(nvidia.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            free_mib = int(float(parts[4]))
            min_free_mib = free_mib if min_free_mib is None else min(min_free_mib, free_mib)
            gpu_rows.append(
                {
                    "gpu_index": int(float(parts[0])),
                    "gpu_name": parts[1],
                    "memory_total_mib": int(float(parts[2])),
                    "memory_used_mib": int(float(parts[3])),
                    "memory_free_mib": free_mib,
                }
            )
        except Exception:
            gpu_rows.append({"raw": line})

    compute_rows: list[dict[str, Any]] = []
    for line in str(compute.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(float(parts[0]))
            used = int(float(parts[2]))
        except Exception:
            pid = -1
            used = 0
        compute_rows.append(
            {
                "pid": pid,
                "process_name": parts[1],
                "used_memory_mib": used,
                "project_owned": False,
            }
        )

    process_rows = []
    for line in str(proc.get("stdout", "")).splitlines():
        cols = line.strip().split(None, 2)
        if len(cols) < 3:
            continue
        process_rows.append(
            {
                "pid": int(cols[0]) if cols[0].isdigit() else cols[0],
                "ppid": int(cols[1]) if cols[1].isdigit() else cols[1],
                "cmd": cols[2],
                "project_owned": str(REPO_ROOT) in cols[2],
            }
        )
    process_by_pid = {int(row["pid"]): str(row["cmd"]) for row in process_rows if isinstance(row.get("pid"), int)}
    for row in compute_rows:
        pid = int(row.get("pid", -1) or -1)
        row["cmd"] = process_by_pid.get(pid, "")
        row["project_owned"] = str(REPO_ROOT) in str(row.get("cmd", ""))

    required_free_mib = int(os.environ.get("WUJI_SCREW1_V2_MIN_GPU_FREE_MIB", "12000"))
    blocked = bool(min_free_mib is not None and min_free_mib < required_free_mib)
    return {
        "gpu_preflight_executed": True,
        "gpu_preflight_result_class": "HOST_GPU_BUSY" if blocked else "HEALTH_OK",
        "gpu_preflight_blocked": blocked,
        "gpu_preflight_required_free_mib": required_free_mib,
        "gpu_preflight_min_free_mib": min_free_mib if min_free_mib is not None else "",
        "gpu_preflight_gpus": gpu_rows,
        "gpu_preflight_compute_processes": compute_rows,
        "gpu_preflight_process_scan": process_rows,
        "gpu_preflight_nvidia_smi": nvidia,
        "gpu_preflight_compute_query": compute,
        "gpu_preflight_process_query": proc,
    }


def _early_guarded_clean_v2_output(output_dir: Path) -> dict[str, Any]:
    resolved = (REPO_ROOT / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    debug_root = (REPO_ROOT / "debug_runs").resolve()
    canonical = (REPO_ROOT / "debug_runs/full_oracle_sticky_assembly_pipeline").resolve()
    try:
        allowed = resolved.is_relative_to(debug_root) and resolved != debug_root and resolved != canonical
    except AttributeError:
        allowed = str(resolved).startswith(str(debug_root) + "/") and resolved != debug_root and resolved != canonical
    info = {
        "clean_v2_output_requested": True,
        "clean_v2_output_path": str(resolved),
        "clean_v2_output_allowed": bool(allowed),
        "clean_v2_output_deleted": False,
        "clean_v2_output_error": "",
    }
    if not allowed:
        info["clean_v2_output_error"] = "refusing_to_delete_non_debug_runs_or_canonical_path"
        return info
    try:
        shutil.rmtree(resolved, ignore_errors=False)
        info["clean_v2_output_deleted"] = True
    except FileNotFoundError:
        info["clean_v2_output_deleted"] = False
    except Exception as exc:
        info["clean_v2_output_error"] = f"{type(exc).__name__}:{exc}"
    return info


def _write_early_v2_blocker(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("episode_summary.csv", "active_step_trace.csv", "hand_action_authority.csv"):
        (output_dir / name).write_text("", encoding="utf-8")
    (output_dir / "v2_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_cli_option(args: list[str], name: str, value: str) -> list[str]:
    out: list[str] = []
    skip = False
    found = False
    for index, item in enumerate(args):
        if skip:
            skip = False
            continue
        if item == name:
            out.extend([name, value])
            found = True
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                skip = True
            continue
        if item.startswith(f"{name}="):
            out.append(f"{name}={value}")
            found = True
            continue
        out.append(item)
    if not found:
        out.extend([name, value])
    return out


def _remove_cli_flag(args: list[str], name: str) -> list[str]:
    out: list[str] = []
    skip = False
    for index, item in enumerate(args):
        if skip:
            skip = False
            continue
        if item == name:
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                skip = True
            continue
        if item.startswith(f"{name}="):
            continue
        out.append(item)
    return out


def _run_v2_collision_offset_ab(args: argparse.Namespace) -> dict[str, Any]:
    """Run isolated no-hand children for collision-offset OFF/ON and pick a spawn-time mode."""

    output_dir = Path(str(getattr(args, "output_dir", "debug_runs/screw1_grasp_baseline_v2")))
    parent_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir
    parent_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for mode, env_value in (("off", "0"), ("on", "1")):
        child_out = parent_dir.with_name(f"{parent_dir.name}_ab_{mode}")
        child_args = list(sys.argv[1:])
        child_args = _replace_cli_option(child_args, "--v2_phase", "no_hand")
        child_args = _replace_cli_option(child_args, "--v2_collision_offset_fix", mode)
        child_args = _replace_cli_option(child_args, "--output_dir", str(child_out))
        child_args = _remove_cli_flag(child_args, "--record_video")
        child_args = _remove_cli_flag(child_args, "--enable_cameras")
        child_args = _remove_cli_flag(child_args, "--alignment_debug")
        if "--clean_v2_output" not in child_args:
            child_args.append("--clean_v2_output")
        env = dict(os.environ)
        env["WUJI_SCREW1_V2_AB_CHILD"] = "1"
        env["WUJI_SCREW1_V2_COLLISION_OFFSET_FIX"] = env_value
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *child_args],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        summary_path = child_out / "v2_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        summary.update(
            {
                "ab_child_mode": mode,
                "ab_child_returncode": int(proc.returncode),
                "ab_child_stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
                "ab_child_stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
                "ab_child_output_dir": str(child_out),
            }
        )
        results[mode] = summary
    def child_complete(summary: dict[str, Any]) -> bool:
        child_returncode = summary.get("ab_child_returncode", -1)
        try:
            child_returncode_int = int(child_returncode)
        except Exception:
            child_returncode_int = -1
        return bool(
            child_returncode_int == 0
            and str(summary.get("result_class") or "") != "V2_IMPLEMENTATION_ERROR"
            and bool(summary.get("health_executed"))
            and bool(summary.get("support_calibration_executed"))
            and bool(summary.get("canonical_support_pose_valid"))
            and bool(summary.get("no_hand_stability_executed"))
            and int(summary.get("no_hand_trials", 0) or 0) >= 5
        )

    off_complete = child_complete(results.get("off", {}))
    on_complete = child_complete(results.get("on", {}))
    off_pass = bool(off_complete and results.get("off", {}).get("no_hand_stability_passed"))
    on_pass = bool(on_complete and results.get("on", {}).get("no_hand_stability_passed"))
    selected = "0"
    reason = "off_selected_default_simpler_config"
    valid = bool(off_complete and on_complete and (off_pass or on_pass))
    if on_pass and not off_pass:
        selected = "1"
        reason = "on_selected_no_hand_passed_off_failed"
    elif on_pass and off_pass:
        selected = "0"
        reason = "both_passed_off_selected_simpler_legal_config"
    elif not off_complete or not on_complete:
        selected = ""
        reason = "ab_incomplete_no_collision_offset_selected"
    elif not on_pass and not off_pass:
        selected = ""
        reason = "both_ab_modes_failed_no_collision_offset_selected"
    ab_summary = {
        "collision_offset_ab_executed": True,
        "collision_offset_ab_valid": valid,
        "collision_offset_ab_off_complete": off_complete,
        "collision_offset_ab_on_complete": on_complete,
        "collision_offset_ab_off_passed": off_pass,
        "collision_offset_ab_on_passed": on_pass,
        "collision_offset_ab_selected_env_value": selected,
        "collision_offset_ab_selection_reason": reason,
        "collision_offset_ab_results": results,
    }
    (parent_dir / "v2_collision_offset_ab_summary.json").write_text(
        json.dumps(ab_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sibling_summary = parent_dir.with_name(f"{parent_dir.name}_ab_selection.json")
    sibling_summary.write_text(json.dumps(ab_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ab_summary["collision_offset_ab_summary_path"] = str(sibling_summary)
    return ab_summary


def _run_v2_wrist_pd_ab(args: argparse.Namespace) -> dict[str, Any]:
    """Run wrist-translation isolation with wrist PD fix OFF/ON before Isaac assets import."""

    output_dir = Path(str(getattr(args, "output_dir", "debug_runs/screw1_grasp_baseline_v2_wrist_translation_isolation")))
    parent_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir
    parent_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for mode, env_value in (("off", "0"), ("on", "1")):
        child_out = parent_dir.with_name(f"{parent_dir.name}_pd_{mode}")
        child_args = list(sys.argv[1:])
        child_args = _replace_cli_option(child_args, "--v2_phase", "wrist_translation_isolation")
        child_args = _replace_cli_option(child_args, "--v2_wrist_pd_fix", mode)
        child_args = _replace_cli_option(child_args, "--output_dir", str(child_out))
        child_args = _remove_cli_flag(child_args, "--record_video")
        child_args = _remove_cli_flag(child_args, "--enable_cameras")
        child_args = _remove_cli_flag(child_args, "--alignment_debug")
        if "--clean_v2_output" not in child_args:
            child_args.append("--clean_v2_output")
        env = dict(os.environ)
        env["WUJI_SCREW1_V2_WRIST_PD_AB_CHILD"] = "1"
        env["WUJI_SCREW1_V2_WRIST_PD_FIX"] = env_value
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *child_args],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        summary_path = child_out / "v2_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        summary.update(
            {
                "wrist_pd_child_mode": mode,
                "wrist_pd_child_returncode": int(proc.returncode),
                "wrist_pd_child_stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
                "wrist_pd_child_stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
                "wrist_pd_child_output_dir": str(child_out),
            }
        )
        results[mode] = summary

    def child_complete(summary: dict[str, Any]) -> bool:
        try:
            return bool(
                int(summary.get("wrist_pd_child_returncode", -1)) == 0
                and str(summary.get("result_class") or "") != "V2_IMPLEMENTATION_ERROR"
                and bool(summary.get("wrist_translation_isolation_executed"))
            )
        except Exception:
            return False

    off_complete = child_complete(results.get("off", {}))
    on_complete = child_complete(results.get("on", {}))
    pass_classes = {"WRIST_TRANSLATION_VALIDATED", "WRIST_TRANSLATION_ISOLATION_PASS"}

    def wrist_corridor_passed(summary: dict[str, Any]) -> bool:
        return bool(
            str(summary.get("result_class") or "") in pass_classes
            or (
                str(summary.get("result_class") or "") == "SEED_PATH_GEOMETRY_ISSUE"
                and bool(summary.get("direct_wrist_x_joint_target_passed"))
                and bool(summary.get("task_space_wrist_x_passed"))
            )
        )

    off_pass = bool(off_complete and wrist_corridor_passed(results.get("off", {})))
    on_pass = bool(on_complete and wrist_corridor_passed(results.get("on", {})))
    selected = ""
    reason = "both_pd_modes_failed_or_incomplete"
    result_class = str(results.get("off", {}).get("result_class") or results.get("on", {}).get("result_class") or "WRIST_CONTROL_BLOCKER_PROVEN")
    if off_pass:
        selected = "0"
        reason = "off_selected_simpler_pd_mode_passed"
        result_class = str(results.get("off", {}).get("result_class") or "WRIST_TRANSLATION_VALIDATED")
    elif on_pass:
        selected = "1"
        reason = "on_selected_only_pd_mode_passed"
        result_class = str(results.get("on", {}).get("result_class") or "WRIST_TRANSLATION_VALIDATED")
    elif off_complete and on_complete:
        reason = "both_pd_modes_completed_without_isolation_pass"
        result_class = str(results.get("off", {}).get("result_class") or results.get("on", {}).get("result_class") or result_class)
    ab_summary = {
        "wrist_pd_ab_executed": True,
        "wrist_pd_ab_valid": bool(off_complete and on_complete),
        "wrist_pd_ab_off_complete": off_complete,
        "wrist_pd_ab_on_complete": on_complete,
        "wrist_pd_ab_off_passed": off_pass,
        "wrist_pd_ab_on_passed": on_pass,
        "wrist_pd_ab_selected_env_value": selected,
        "wrist_pd_ab_selection_reason": reason,
        "wrist_pd_ab_results": results,
        "result_class": result_class,
        "blocker": reason,
        "final_allowed_result_class": True,
    }
    (parent_dir / "v2_wrist_pd_ab_summary.json").write_text(
        json.dumps(ab_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_early_v2_blocker(parent_dir, ab_summary)
    return ab_summary


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--part", default="Screw1")
_pre_parser.add_argument("--output_dir", default="debug_runs/scripted_baseline_screw1")
_pre_parser.add_argument("--baseline_v2", action="store_true")
_pre_parser.add_argument("--contact_adaptive_closure", action="store_true")
_pre_parser.add_argument("--v2_phase", default="full")
_pre_parser.add_argument("--clean_v2_output", action="store_true")
_pre_parser.add_argument("--v2_collision_offset_fix", default="auto", choices=("auto", "off", "on"))
_pre_parser.add_argument("--v2_wrist_pd_fix", default="auto", choices=("auto", "off", "on"))
_pre_args, _ = _pre_parser.parse_known_args()
if bool(_pre_args.contact_adaptive_closure):
    _early = _early_v2_health_blocker(_pre_args)
    _gpu_preflight = _early_v2_gpu_preflight(_pre_args)
    _early.update(_gpu_preflight)
    if bool(_gpu_preflight.get("gpu_preflight_blocked")) or _early.get("blocker"):
        _out = Path(_pre_args.output_dir)
        if bool(getattr(_pre_args, "clean_v2_output", False)):
            _early.update(_early_guarded_clean_v2_output(_out))
        _early["result_class"] = (
            "HOST_GPU_BUSY" if bool(_gpu_preflight.get("gpu_preflight_blocked")) else "HOST_RESOURCE_PREFLIGHT_BLOCKED"
        )
        _early["final_allowed_result_class"] = False
        _write_early_v2_blocker(_out, _early)
        (_out / "contact_adaptive_summary.json").write_text(
            json.dumps(_early, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(_early, indent=2, sort_keys=True), flush=True)
        sys.exit(0)
if bool(_pre_args.baseline_v2):
    if str(_pre_args.v2_wrist_pd_fix) == "on":
        os.environ["WUJI_SCREW1_V2_WRIST_PD_FIX"] = "1"
    elif str(_pre_args.v2_wrist_pd_fix) == "off":
        os.environ["WUJI_SCREW1_V2_WRIST_PD_FIX"] = "0"
    elif str(_pre_args.v2_phase) != "wrist_translation_isolation":
        os.environ.setdefault("WUJI_SCREW1_V2_WRIST_PD_FIX", "1")
    _early = _early_v2_health_blocker(_pre_args)
    _gpu_preflight = _early_v2_gpu_preflight(_pre_args)
    _early.update(_gpu_preflight)
    if bool(_gpu_preflight.get("gpu_preflight_blocked")):
        _out = Path(_pre_args.output_dir)
        if bool(getattr(_pre_args, "clean_v2_output", False)):
            _early.update(_early_guarded_clean_v2_output(_out))
        _early.update(
            {
                "result_class": "HOST_GPU_BUSY",
                "health_result_class": str(_early.get("health_result_class") or "HEALTH_OK"),
                "blocker": "current_gpu_memory_or_compute_process_pressure",
                "final_allowed_result_class": True,
            }
        )
        _write_early_v2_blocker(_out, _early)
        print(json.dumps(_early, indent=2, sort_keys=True), flush=True)
        sys.exit(0)
    if _early.get("blocker") or str(_pre_args.v2_phase) == "health":
        _out = Path(_pre_args.output_dir)
        if bool(getattr(_pre_args, "clean_v2_output", False)):
            _early.update(_early_guarded_clean_v2_output(_out))
            _early["health_after"] = _early_health_snapshot()
        _write_early_v2_blocker(_out, _early)
        print(json.dumps(_early, indent=2, sort_keys=True), flush=True)
        sys.exit(0)
    if (
        str(_pre_args.v2_phase) == "wrist_translation_isolation"
        and str(_pre_args.v2_wrist_pd_fix) == "auto"
        and os.environ.get("WUJI_SCREW1_V2_WRIST_PD_AB_CHILD", "0") != "1"
    ):
        _pd_ab = _run_v2_wrist_pd_ab(_pre_args)
        print(json.dumps(_pd_ab, indent=2, sort_keys=True), flush=True)
        sys.exit(0)
    if str(_pre_args.v2_collision_offset_fix) == "on":
        os.environ["WUJI_SCREW1_V2_COLLISION_OFFSET_FIX"] = "1"
    elif str(_pre_args.v2_collision_offset_fix) == "off":
        os.environ["WUJI_SCREW1_V2_COLLISION_OFFSET_FIX"] = "0"
    elif str(_pre_args.v2_phase) == "full" and os.environ.get("WUJI_SCREW1_V2_AB_CHILD", "0") != "1":
        _ab = _run_v2_collision_offset_ab(_pre_args)
        if not bool(_ab.get("collision_offset_ab_valid")):
            _out = Path(_pre_args.output_dir)
            _summary = {
                **_early,
                **_ab,
                "result_class": "INITIAL_PHYSICS_BLOCKER_PROVEN",
                "blocker": str(_ab.get("collision_offset_ab_selection_reason") or "collision_offset_ab_invalid"),
                "final_allowed_result_class": True,
            }
            _write_early_v2_blocker(_out, _summary)
            print(json.dumps(_summary, indent=2, sort_keys=True), flush=True)
            sys.exit(0)
        os.environ["WUJI_SCREW1_V2_COLLISION_OFFSET_FIX"] = str(_ab.get("collision_offset_ab_selected_env_value") or "0")
        os.environ["WUJI_SCREW1_V2_AB_SUMMARY_PATH"] = str(_ab.get("collision_offset_ab_summary_path") or "")

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--part", default="Screw1", choices=("Screw1", "Plug2"))
parser.add_argument("--num_envs", type=int, default=5)
parser.add_argument("--output_dir", default="debug_runs/scripted_baseline_screw1")
parser.add_argument("--max_approach_steps", type=int, default=150)
parser.add_argument("--max_close_steps", type=int, default=50)
parser.add_argument("--max_lift_steps", type=int, default=60)
parser.add_argument("--record_video", nargs="?", const=True, default=False, type=_parse_bool)
parser.add_argument("--alignment_debug", action="store_true")
parser.add_argument("--runtime_collision_debug", action="store_true")
parser.add_argument("--force_close_without_contact_for_debug", action="store_true")
parser.add_argument("--baseline_v2", action="store_true")
parser.add_argument("--contact_adaptive_closure", action="store_true")
parser.add_argument(
    "--closure_phase",
    default="full",
    choices=(
        "prior_analysis",
        "morphology_calibration",
        "screw1_ab",
        "grasp",
        "repeated_trials",
        "transfer",
        "full",
    ),
)
parser.add_argument("--closure_prior_recording", default="/mnt/data/recording_1.pkl")
parser.add_argument("--closure_development_trials", type=int, default=3)
parser.add_argument(
    "--morphology_calibration_path",
    default="debug_runs/closure_prior/wuji_hand_morphology_calibration.json",
)
parser.add_argument(
    "--v2_phase",
    default="full",
    choices=(
        "full",
        "health",
        "no_hand",
        "wrist_audit",
        "wrist_translation_isolation",
        "action_audit",
        "seed_replay",
        "grasp",
        "repeated_trials",
    ),
)
parser.add_argument("--clean_v2_output", action="store_true")
parser.add_argument("--v2_collision_offset_fix", default="auto", choices=("auto", "off", "on"))
parser.add_argument("--v2_wrist_pd_fix", default="auto", choices=("auto", "off", "on"))
parser.add_argument("--v2_seed_replay_repeats", type=int, default=None)
parser.add_argument("--v2_trajectory_bank_candidate_limit", type=int, default=None)
parser.add_argument("--v2_seed_servo_max_steps", type=int, default=None)
parser.add_argument("--v2_skip_legacy_seed_replay", action="store_true")
parser.add_argument("--v2_resume_seed_replay_progress", action="store_true")
parser.add_argument("--v2_revalidate_wrist_action_gates", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if bool(args_cli.baseline_v2) and str(args_cli.part) != "Screw1":
    parser.error("--baseline_v2 remains Screw1-only; use --contact_adaptive_closure for Plug2")
if bool(args_cli.baseline_v2) and bool(args_cli.contact_adaptive_closure):
    parser.error("choose exactly one of --baseline_v2 or --contact_adaptive_closure")
if bool(args_cli.record_video):
    args_cli.enable_cameras = True
if bool(args_cli.baseline_v2):
    if str(args_cli.v2_wrist_pd_fix) == "on":
        os.environ["WUJI_SCREW1_V2_WRIST_PD_FIX"] = "1"
    elif str(args_cli.v2_wrist_pd_fix) == "off":
        os.environ["WUJI_SCREW1_V2_WRIST_PD_FIX"] = "0"
    else:
        os.environ.setdefault("WUJI_SCREW1_V2_WRIST_PD_FIX", "1")
    if str(args_cli.v2_collision_offset_fix) == "on":
        os.environ["WUJI_SCREW1_V2_COLLISION_OFFSET_FIX"] = "1"
    elif str(args_cli.v2_collision_offset_fix) == "off":
        os.environ["WUJI_SCREW1_V2_COLLISION_OFFSET_FIX"] = "0"
    else:
        os.environ.setdefault("WUJI_SCREW1_V2_COLLISION_OFFSET_FIX", "0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks.direct.np  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from pipeline.video import StreamingVideoRecorder, configure_pipeline_context  # noqa: E402
from pipeline.unified_grasp.scripted_contact_baseline import (  # noqa: E402
    ScriptedBaselineConfig,
    run_scripted_contact_baseline,
)
from pipeline.unified_grasp.screw1_grasp_baseline_v2 import (  # noqa: E402
    Screw1GraspBaselineV2Config,
    run_screw1_grasp_baseline_v2,
)
from pipeline.unified_grasp.contact_adaptive_grasp_baseline import (  # noqa: E402
    ContactAdaptiveGraspConfig,
    run_contact_adaptive_grasp_baseline,
)
from pipeline.unified_grasp.video_log_alignment import make_run_id, write_run_manifest  # noqa: E402


TASK_NAME = "Isaac-Wuji-UnifiedPhysicalGrasp-v83-Direct-v0"
VIDEO_CAMERA_EYE = (0.62, -0.72, 1.28)
VIDEO_CAMERA_LOOKAT = (-0.20, 0.00, 0.80)
VIDEO_RESOLUTION = (1280, 720)

configure_pipeline_context({"REPO_ROOT": REPO_ROOT})


def _make_env() -> Any:
    kwargs: dict[str, Any] = {
        "device": getattr(args_cli, "device", None) or "cuda:0",
        "num_envs": int(args_cli.num_envs),
    }
    if hasattr(args_cli, "disable_fabric"):
        kwargs["use_fabric"] = not bool(getattr(args_cli, "disable_fabric"))
    env_cfg = parse_env_cfg(TASK_NAME, **kwargs)
    if bool(args_cli.record_video) and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = VIDEO_CAMERA_EYE
        env_cfg.viewer.lookat = VIDEO_CAMERA_LOOKAT
        env_cfg.viewer.resolution = VIDEO_RESOLUTION
    render_mode = "rgb_array" if bool(args_cli.record_video) else None
    gym_env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=render_mode)
    if bool(args_cli.record_video):
        try:
            gym_env.unwrapped.sim.set_camera_view(eye=VIDEO_CAMERA_EYE, target=VIDEO_CAMERA_LOOKAT)
        except Exception:
            pass
    return gym_env


def _update_summary(output_dir: Path, updates: dict[str, Any]) -> None:
    path = output_dir / "rollout_summary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _update_json(path: Path, updates: dict[str, Any]) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _guarded_clean_v2_output(output_dir: Path) -> dict[str, Any]:
    resolved = (REPO_ROOT / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    debug_root = (REPO_ROOT / "debug_runs").resolve()
    canonical = (REPO_ROOT / "debug_runs/full_oracle_sticky_assembly_pipeline").resolve()
    try:
        allowed = resolved.is_relative_to(debug_root) and resolved != debug_root and resolved != canonical
    except AttributeError:
        allowed = str(resolved).startswith(str(debug_root) + "/") and resolved != debug_root and resolved != canonical
    info = {
        "clean_v2_output_requested": True,
        "clean_v2_output_path": str(resolved),
        "clean_v2_output_allowed": bool(allowed),
        "clean_v2_output_deleted": False,
        "clean_v2_output_error": "",
    }
    if not allowed:
        info["clean_v2_output_error"] = "refusing_to_delete_non_debug_runs_or_canonical_path"
        return info
    try:
        shutil.rmtree(resolved, ignore_errors=False)
        info["clean_v2_output_deleted"] = True
    except FileNotFoundError:
        info["clean_v2_output_deleted"] = False
    except Exception as exc:
        info["clean_v2_output_error"] = f"{type(exc).__name__}:{exc}"
    return info


def _finalize_video(output_dir: Path, recorder: StreamingVideoRecorder | None) -> dict[str, Any]:
    if recorder is None:
        return {
            "video_available": False,
            "video_path": "",
            "video_unavailable_reason": "record_video_false",
        }
    diagnostics = recorder.diagnostics()
    video_dir = output_dir / "videos"
    desired_path = video_dir / "scripted_screw1_baseline.mp4"
    source_value = diagnostics.get("raw_video_path") or diagnostics.get("streaming_raw_video_path") or ""
    source_path = REPO_ROOT / source_value if source_value and not Path(source_value).is_absolute() else Path(source_value)
    copied = False
    failure = str(diagnostics.get("video_write_failure_reason") or "")
    transcode: dict[str, Any] = {}
    if source_path.is_file():
        try:
            desired_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != desired_path.resolve():
                shutil.copy2(source_path, desired_path)
            copied = desired_path.is_file()
            transcode = _transcode_video_to_h264(desired_path) if copied else {}
        except Exception as exc:
            failure = f"video_copy_failed:{type(exc).__name__}:{exc}"
            transcode = {}
    elif not failure:
        failure = "streaming_video_file_missing"
    return {
        "video_available": bool(copied),
        "video_path": str(desired_path) if copied else "",
        "video_unavailable_reason": "" if copied else failure,
        "video_diagnostics": diagnostics,
        **transcode,
    }


def _transcode_video_to_h264(path: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            return {"video_h264_transcoded": False, "video_h264_error": f"ffmpeg_unavailable:{type(exc).__name__}:{exc}"}
    tmp = path.with_name(f"{path.stem}_h264_tmp{path.suffix}")
    cmd = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["unknown_error"]
        return {"video_h264_transcoded": False, "video_h264_error": tail[0]}
    tmp.replace(path)
    return {"video_h264_transcoded": True, "video_h264_error": ""}


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    clean_info: dict[str, Any] = {}
    if (bool(args_cli.baseline_v2) or bool(args_cli.contact_adaptive_closure)) and bool(args_cli.clean_v2_output):
        clean_info = _guarded_clean_v2_output(output_dir)
        if clean_info.get("clean_v2_output_error"):
            print(json.dumps(clean_info, indent=2, sort_keys=True), flush=True)
            simulation_app.close()
            return
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    start_time = datetime.now(timezone.utc).isoformat()
    if bool(args_cli.alignment_debug):
        write_run_manifest(
            output_dir,
            run_id=run_id,
            start_time=start_time,
            end_time=None,
            command_line=sys.argv,
            repo_root=REPO_ROOT,
            target_part=str(args_cli.part),
            target_env_index=None,
        )
    legacy_raw = output_dir / "videos" / "wuji_v2_full_oracle_sticky_assembly_pipeline_streaming_raw.mp4"
    try:
        legacy_raw.unlink(missing_ok=True)
    except Exception:
        pass
    gym_env = None
    recorder = None
    run_env = None
    try:
        gym_env = _make_env()
        run_env = gym_env
        if bool(args_cli.record_video):
            fps = 30.0
            try:
                fps = float(gym_env.metadata.get("render_fps", fps))
            except Exception:
                pass
            recorder = StreamingVideoRecorder(
                gym_env,
                output_dir / "videos",
                target_fps=fps,
                requested_mode="streaming",
                effective_mode="streaming",
            )
            recorder.raw_video_path = output_dir / "videos" / (
                "contact_adaptive_closure_raw.mp4"
                if bool(args_cli.contact_adaptive_closure)
                else "screw1_grasp_baseline_v2_raw.mp4"
                if bool(args_cli.baseline_v2)
                else "scripted_screw1_baseline_raw.mp4"
            )
            run_env = recorder
        if bool(args_cli.contact_adaptive_closure):
            cfg_closure = ContactAdaptiveGraspConfig(
                part=str(args_cli.part),
                phase=str(args_cli.closure_phase),
                output_dir=str(output_dir),
                run_id=run_id,
                alignment_debug=bool(args_cli.alignment_debug),
                closure_prior_recording=str(args_cli.closure_prior_recording),
                morphology_calibration_path=str(args_cli.morphology_calibration_path),
                development_trials_per_variant=max(1, min(3, int(args_cli.closure_development_trials))),
            )
            summary = run_contact_adaptive_grasp_baseline(run_env, cfg_closure, video_recorder=recorder)
            video_info = _finalize_video(output_dir, recorder)
            _update_json(output_dir / "contact_adaptive_summary.json", video_info)
            if bool(args_cli.alignment_debug):
                write_run_manifest(
                    output_dir,
                    run_id=run_id,
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc).isoformat(),
                    command_line=sys.argv,
                    repo_root=REPO_ROOT,
                    target_part=str(args_cli.part),
                    target_env_index=int(summary.get("target_env_index", -1)),
                    summary_path=output_dir / "contact_adaptive_summary.json",
                    log_path=output_dir / "contact_adaptive_trace.csv",
                    video_path=Path(video_info.get("video_path") or ""),
                    extra={
                        "contact_adaptive_closure": True,
                        "closure_phase": str(args_cli.closure_phase),
                        "morphology_calibration_path": str(args_cli.morphology_calibration_path),
                    },
                )
            print(json.dumps({**summary, **video_info}, indent=2, sort_keys=True), flush=True)
            return
        if bool(args_cli.baseline_v2):
            cfg_v2 = Screw1GraspBaselineV2Config(
                part=str(args_cli.part),
                output_dir=str(output_dir),
                phase=str(args_cli.v2_phase),
                v2_collision_offset_fix=str(args_cli.v2_collision_offset_fix),
                alignment_debug=bool(args_cli.alignment_debug),
                run_id=run_id,
                clean_output_info=clean_info,
            )
            cfg_v2.gpu_preflight_info = dict(globals().get("_gpu_preflight", {}) or {})
            if args_cli.v2_seed_replay_repeats is not None:
                cfg_v2.seed_replay_repeats = max(1, int(args_cli.v2_seed_replay_repeats))
            if args_cli.v2_trajectory_bank_candidate_limit is not None:
                cfg_v2.trajectory_bank_candidate_limit = max(0, int(args_cli.v2_trajectory_bank_candidate_limit))
            if args_cli.v2_seed_servo_max_steps is not None:
                cfg_v2.seed_servo_max_steps = max(1, int(args_cli.v2_seed_servo_max_steps))
            cfg_v2.seed_replay_skip_legacy_seeds = bool(args_cli.v2_skip_legacy_seed_replay)
            cfg_v2.seed_replay_resume_progress = bool(args_cli.v2_resume_seed_replay_progress)
            cfg_v2.accept_prior_wrist_action_gates = not bool(args_cli.v2_revalidate_wrist_action_gates)
            summary = run_screw1_grasp_baseline_v2(run_env, cfg_v2, video_recorder=recorder)
            video_info = _finalize_video(output_dir, recorder)
            _update_json(output_dir / "v2_summary.json", video_info)
            if bool(args_cli.alignment_debug):
                write_run_manifest(
                    output_dir,
                    run_id=run_id,
                    start_time=start_time,
                    end_time=datetime.now(timezone.utc).isoformat(),
                    command_line=sys.argv,
                    repo_root=REPO_ROOT,
                    target_part=str(args_cli.part),
                    target_env_index=int(summary.get("target_env_index", -1)),
                    summary_path=output_dir / "v2_summary.json",
                    log_path=output_dir / "active_step_trace.csv",
                    video_path=Path(video_info.get("video_path") or summary.get("aligned_video_path") or ""),
                    extra={
                        "baseline_v2": True,
                        "v2_phase": str(args_cli.v2_phase),
                        "aligned_video_path": summary.get("aligned_video_path", ""),
                        "contact_alignment_summary_path": summary.get("contact_alignment_summary_json", ""),
                        "body_mapping_path": summary.get("body_mapping_json", ""),
                    },
                )
            print(json.dumps({**summary, **video_info}, indent=2, sort_keys=True), flush=True)
            return
        cfg = ScriptedBaselineConfig(
            part=str(args_cli.part),
            output_dir=str(output_dir),
            max_approach_steps=int(args_cli.max_approach_steps),
            max_close_steps=int(args_cli.max_close_steps),
            max_lift_steps=int(args_cli.max_lift_steps),
            force_close_without_contact_for_debug=bool(args_cli.force_close_without_contact_for_debug),
            alignment_debug=bool(args_cli.alignment_debug),
            runtime_collision_debug=bool(args_cli.runtime_collision_debug),
            run_id=run_id,
        )
        summary = run_scripted_contact_baseline(run_env, cfg, video_recorder=recorder)
        video_info = _finalize_video(output_dir, recorder)
        _update_summary(output_dir, video_info)
        if bool(args_cli.alignment_debug):
            write_run_manifest(
                output_dir,
                run_id=run_id,
                start_time=start_time,
                end_time=datetime.now(timezone.utc).isoformat(),
                command_line=sys.argv,
                repo_root=REPO_ROOT,
                target_part=str(args_cli.part),
                target_env_index=int(summary.get("target_env_index", -1)),
                summary_path=output_dir / "rollout_summary.json",
                log_path=output_dir / "rollout_log.csv",
                video_path=Path(video_info.get("video_path") or summary.get("aligned_video_path") or ""),
                extra={
                    "aligned_video_path": summary.get("aligned_video_path", ""),
                    "contact_alignment_summary_path": summary.get("contact_alignment_summary_json", ""),
                    "body_mapping_path": summary.get("body_mapping_json", ""),
                },
            )
        print(json.dumps({**summary, **video_info}, indent=2, sort_keys=True), flush=True)
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                pass
        elif gym_env is not None:
            try:
                gym_env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
