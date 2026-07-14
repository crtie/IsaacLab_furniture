"""Run bounded external-model smokes without importing third-party packages into project source."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = Path("/tmp/multi_object_privileged_grasp_v3_external")
OUTPUT = REPO_ROOT / "debug_runs/multi_object_privileged_grasp_v3/external_priors"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [_gendex_smoke(), _static_smoke("DRO-Grasp"), _static_smoke("DexGraspNet")]
    payload = {
        "schema_version": 3,
        "models": rows,
        "third_party_code_copied_into_project": False,
        "third_party_checkpoint_packaged": False,
        "physical_success_evidence": False,
    }
    (OUTPUT / "external_model_runtime_smokes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _gendex_smoke() -> dict[str, object]:
    repo = EXTERNAL / "GenDexGrasp"
    module_path = repo / "utils_model/PointNetCVAE.py"
    spec = importlib.util.spec_from_file_location("external_gendex_pointnet_cvae", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load GenDexGrasp PointNetCVAE")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PointNetCVAE().eval()
    checkpoint = repo / "ckpts/SqrtFullRobots/weights/pointnet_cvae_model.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    rod = trimesh.load_mesh(
        REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/np/asset/chair/rod.obj",
        process=False,
    )
    points, face_indices = trimesh.sample.sample_surface(rod, 2048, seed=20260713)
    center = np.mean(points, axis=0)
    scale = max(float(np.max(np.linalg.norm(points - center, axis=1))), 1.0e-9)
    normalized = (points - center) / scale
    with torch.no_grad():
        contact_map = model.inference(
            torch.as_tensor(normalized, dtype=torch.float32).reshape(1, 2048, 3),
            torch.zeros((1, 128), dtype=torch.float32),
        )[0].numpy()
    top = np.argsort(contact_map)[-64:][::-1]
    artifact = {
        "schema_version": 3,
        "model": "GenDexGrasp PointNetCVAE SqrtFullRobots",
        "commit": _commit(repo),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "input_mesh": str(rod.metadata.get("file_path", "rod.obj")),
        "input_point_count": 2048,
        "output_finite": bool(np.all(np.isfinite(contact_map))),
        "output_min": float(np.min(contact_map)),
        "output_max": float(np.max(contact_map)),
        "top_contact_points_object": points[top].tolist(),
        "top_face_indices": face_indices[top].tolist(),
        "used_as_sampling_hint_only": True,
        "wuji_pose_directly_supported": False,
        "redistribution_allowed": False,
    }
    path = OUTPUT / "gendex_rod_contact_map_seed.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"name": "GenDexGrasp", "status": "REAL_CHECKPOINT_INFERENCE_PASS", "artifact": str(path), **artifact}


def _static_smoke(name: str) -> dict[str, object]:
    repo = EXTERNAL / name
    entry = repo / ("scripts/example_pretrain.py" if name == "DRO-Grasp" else "grasp_generation/main.py")
    result = subprocess.run(
        ["python", "-m", "py_compile", str(entry)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    checkpoints = sorted(str(path.relative_to(repo)) for path in repo.rglob("*.pth"))
    return {
        "name": name,
        "status": "ENTRYPOINT_COMPILE_PASS_CHECKPOINT_OR_DATA_REQUIRED" if result.returncode == 0 else "ENTRYPOINT_COMPILE_FAILED",
        "commit": _commit(repo),
        "entrypoint": str(entry),
        "checkpoint_files": checkpoints,
        "stdout": result.stdout[-2000:],
        "physical_success_evidence": False,
    }


def _commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    main()
