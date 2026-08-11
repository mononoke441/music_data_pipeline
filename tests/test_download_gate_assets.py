from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_gate_assets.sh"


def _run(tmp_path: Path, *arguments: str, extra_env=None):
    env = dict(os.environ)
    env.update({
        "GATE_ASSET_ROOT": str(tmp_path / "assets"),
        "HF_ENDPOINT": "https://hf.example.invalid",
    })
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _assets(tmp_path: Path):
    root = tmp_path / "assets"
    return {
        "MobileNetV1_mAP=0.389.pth": b"mobilenet-real-bytes",
    }, root


def test_preinstalled_assets_are_reused_and_real_sha256_manifest_is_written(tmp_path: Path):
    assets, root = _assets(tmp_path)
    root.mkdir()
    for name, content in assets.items():
        (root / name).write_bytes(content)

    completed = _run(tmp_path, "all", extra_env={
        "PANNS_MOBILENET_MD5": hashlib.md5(assets["MobileNetV1_mAP=0.389.pth"]).hexdigest(),
    })

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("reusing preinstalled file") == 1
    manifest = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert manifest == [
        f"{hashlib.sha256(content).hexdigest()}  {name}"
        for name, content in assets.items()
    ]


def test_verify_is_offline_and_rejects_trusted_sha_mismatch(tmp_path: Path):
    assets, root = _assets(tmp_path)
    root.mkdir()
    model_path = root / "MobileNetV1_mAP=0.389.pth"
    model_path.write_bytes(assets[model_path.name])

    fake_md5 = hashlib.md5(assets[model_path.name]).hexdigest()
    observed = _run(
        tmp_path, "--verify", "panns-mobilenet",
        extra_env={"PANNS_MOBILENET_MD5": fake_md5},
    )
    assert observed.returncode == 0, observed.stderr
    assert hashlib.sha256(assets[model_path.name]).hexdigest() in observed.stdout
    assert "no trusted expected SHA" in observed.stdout

    rejected = _run(
        tmp_path,
        "--verify",
        "panns-mobilenet",
        extra_env={
            "PANNS_MOBILENET_SHA256": "0" * 64,
            "PANNS_MOBILENET_MD5": fake_md5,
        },
    )
    assert rejected.returncode != 0
    assert "SHA256 mismatch" in rejected.stderr


def test_hf_source_uses_configured_hf_endpoint(tmp_path: Path):
    completed = _run(
        tmp_path,
        "--dry-run",
        "panns-mobilenet",
        extra_env={
            "PANNS_MOBILENET_HF_REPO": "PinnHe/ads",
            "PANNS_MOBILENET_HF_REPO_TYPE": "dataset",
            "PANNS_MOBILENET_HF_FILE": "ckpt/MobileNetV1_mAP=0.389.pth",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert (
        "https://hf.example.invalid/datasets/PinnHe/ads/resolve/main/"
        "ckpt/MobileNetV1_mAP=0.389.pth"
    ) in completed.stdout


def test_unreachable_source_fails_with_preinstall_recovery(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text("#!/usr/bin/env bash\nexit 22\n", encoding="utf-8")
    fake_curl.chmod(0o755)

    completed = _run(
        tmp_path,
        "panns-mobilenet",
        extra_env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert completed.returncode != 0
    assert "Official source is unreachable" in completed.stderr
    assert "Preinstall it" in completed.stderr
    assert "--verify" in completed.stderr


def test_panns_author_checksum_is_enforced_by_default(tmp_path: Path):
    assets, root = _assets(tmp_path)
    root.mkdir()
    (root / "MobileNetV1_mAP=0.389.pth").write_bytes(
        assets["MobileNetV1_mAP=0.389.pth"]
    )

    completed = _run(tmp_path, "--verify", "panns-mobilenet")

    assert completed.returncode != 0
    assert "MD5 mismatch" in completed.stderr
    assert "a419303e1c88aa1b9d2ac3811563d371" in completed.stderr
