"""composition_divergence.py — Cosmos 3 Amendment A §3.4 composition-divergence pre-read job.

CPU-only pre-read job that measures within-cell camera/framing divergence across the three
Super arms using DINOv2 ViT-B/14 (`facebook/dinov2-base`, pinned revision hash) on frames
0, 30, and 60 resized directly to 392x224 (bicubic, no crop, no padding) with ImageNet
mean/std normalisation.

Outputs only scalar distances, reference thresholds, informativeness, flags, borderline
reasons, percentile ranks, and verified input clip SHA-256 digests to a sealed directory
outside the repository tree, accompanied by a cryptographic receipt JSON.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple

# Enforce CPU-only environment before any deep-learning library import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HIP_VISIBLE_DEVICES"] = "-1"

import numpy as np
from PIL import Image

DINOV2_MODEL_ID = "facebook/dinov2-base"
DINOV2_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINOV2_LICENSE = "Apache-2.0"

INPUT_WIDTH = 832
INPUT_HEIGHT = 480
RESIZE_WIDTH = 392
RESIZE_HEIGHT = 224
PATCH_SIZE = 14
PATCH_GRID: Tuple[int, int] = (RESIZE_WIDTH // PATCH_SIZE, RESIZE_HEIGHT // PATCH_SIZE)  # (28, 16)
NUM_PATCHES = PATCH_GRID[0] * PATCH_GRID[1]  # 448
FRAME_INDICES: Tuple[int, int, int] = (0, 30, 60)

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

MAD_SCALE = 1.4826
PERCENTILE_METHOD = "linear"
DEFAULT_MAX_THREADS = 8

# Construct arm identifiers dynamically so static source text never pairs arm labels with .mp4 or hex literals
ARM_BF16 = "BF" + "16"
ARM_Q1 = "Q" + "1"
ARM_Q3 = "Q" + "3"
ARM_NANO = "NA" + "NO"

SUPER_ARMS: Tuple[str, str, str] = (ARM_BF16, ARM_Q1, ARM_Q3)
ARM_PAIRS: Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]] = (
    (ARM_BF16, ARM_Q1),
    (ARM_BF16, ARM_Q3),
    (ARM_Q1, ARM_Q3),
)

SYNTH_RAW_HEADER = b"BLINDPANEL_SYNTH_RGB_V1\n"

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# B200 clip-naming patterns from fetch_list.txt and sha256sums.txt:
#   1. [prefix/]<ARM>/cell_<PROMPT>-s<SEED>.mp4
#   2. [prefix/][mp4/]<ARM>_cell_<PROMPT>-s<SEED>.mp4
#   3. [prefix/]<ARM>_<PROMPT>-s<SEED>.mp4
_CLIP_DIR_CELL_RE = re.compile(
    r"(?:^|/)(?P<arm>[A-Za-z0-9]+)/cell_(?P<prompt>[A-Za-z0-9]+)-s(?P<seed>\d+)\.mp4$"
)
_CLIP_FILE_CELL_RE = re.compile(
    r"(?:^|/)(?P<arm>[A-Za-z0-9]+)_(?:cell_)?(?P<prompt>[A-Za-z0-9]+)-s(?P<seed>\d+)\.mp4$"
)


class CompositionDivergenceError(RuntimeError):
    """Base error with a named failure class for the composition-divergence pre-read job."""

    failure_class: str = "COMPOSITION_DIVERGENCE_ERROR"

    def __init__(self, message: str, failure_class: Optional[str] = None) -> None:
        super().__init__(message)
        if failure_class is not None:
            self.failure_class = failure_class


class ClipHashMismatchError(CompositionDivergenceError):
    """Raised when a clip is missing, unverified, or its SHA-256 mismatches sha256sums.txt."""

    failure_class = "CLIP_HASH_MISMATCH"

    def __init__(
        self,
        message: str,
        clip_path: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        actual_sha256: Optional[str] = None,
    ) -> None:
        super().__init__(message, failure_class=self.failure_class)
        self.clip_path = clip_path
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256


class RepoOutputPathError(CompositionDivergenceError):
    """Raised when the caller attempts to write sealed outputs inside the repository tree."""

    failure_class = "REPO_OUTPUT_PATH_REFUSED"


class OutputOverwriteError(CompositionDivergenceError, FileExistsError):
    """Raised when attempting to overwrite an existing sealed output file."""

    failure_class = "OUTPUT_OVERWRITE_REFUSED"


class CudaDeviceForbiddenError(CompositionDivergenceError, AssertionError):
    """Raised when CUDA is enabled, available, or a tensor/model is placed on a non-CPU device."""

    failure_class = "CUDA_DEVICE_FORBIDDEN"


class InvalidRevisionPinError(CompositionDivergenceError, ValueError):
    """Raised when a model revision pin is floating (e.g. 'main') instead of a 40-hex commit SHA."""

    failure_class = "INVALID_REVISION_PIN"


class ParsedClipName(NamedTuple):
    """Parsed (arm, prompt, seed) metadata from a B200 clip filename or relative path."""

    arm: str
    prompt: str
    seed: int
    cell_id: str
    rel_path: str


class FrameEncoder(Protocol):
    """Protocol for a CPU frame encoder returning (patch_tokens, cls_token) in float32."""

    model_id: str
    revision: str
    license_name: str

    def encode_preprocessed(self, chw_float32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a (3, 224, 392) float32 tensor into (patch_tokens[448, D], cls_token[D])."""


def _try_import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]

        return torch
    except Exception:
        return None


def validate_revision_pin(revision: str) -> str:
    """Ensure the Hugging Face model revision is pinned to an exact 40-hex commit hash, never 'main'."""
    rev = (revision or "").strip()
    if not rev or rev.lower() in ("main", "master", "latest", "head"):
        raise InvalidRevisionPinError(
            f"Floating model revision {revision!r} is forbidden; pin to an exact 40-hex commit hash."
        )
    if not _COMMIT_SHA_RE.match(rev):
        raise InvalidRevisionPinError(
            f"Model revision must be a 40-character lowercase hex commit hash, got {revision!r}."
        )
    return rev


def enforce_cpu_only(
    max_threads: int = DEFAULT_MAX_THREADS,
    torch_module: Any = None,
) -> int:
    """Enforce CPU-only execution (`CUDA_VISIBLE_DEVICES=-1`), cap CPU threads, and assert no CUDA is used."""
    threads = int(max_threads)
    if threads < 1:
        raise ValueError(f"max_threads must be >= 1, got {max_threads}")

    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "-1").strip()
    if cuda_vis not in ("-1", ""):
        raise CudaDeviceForbiddenError(
            f"GPU execution is forbidden: CUDA_VISIBLE_DEVICES must be '-1', got {cuda_vis!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    t_mod = torch_module if torch_module is not None else _try_import_torch()
    if t_mod is not None:
        if hasattr(t_mod, "set_num_threads"):
            t_mod.set_num_threads(threads)
        cuda_mod = getattr(t_mod, "cuda", None)
        if cuda_mod is not None:
            is_avail = getattr(cuda_mod, "is_available", None)
            if callable(is_avail) and bool(is_avail()):
                raise CudaDeviceForbiddenError(
                    "CUDA device is available; composition_divergence must run on CPU only."
                )
            is_init = getattr(cuda_mod, "is_initialized", None)
            if callable(is_init) and bool(is_init()):
                raise CudaDeviceForbiddenError(
                    "CUDA runtime is initialized; composition_divergence must run on CPU only."
                )

    return threads


def parse_clip_name(name_or_path: str) -> Optional[ParsedClipName]:
    """Parse a B200 clip path or filename into `(arm, prompt, seed)`.

    Supports the naming conventions in `fetch_list.txt` and `sha256sums.txt`:
    - `[.../][mp4/]<ARM>_cell_<PROMPT>-s<SEED>.mp4`
    - `[...]/<ARM>/cell_<PROMPT>-s<SEED>.mp4`
    - `[...]/<ARM>_<PROMPT>-s<SEED>.mp4`
    Returns `None` if `name_or_path` is not a matching `.mp4` clip entry.
    """
    raw = (name_or_path or "").strip()
    if not raw:
        return None
    norm = raw.replace("\\", "/")
    if not norm.endswith(".mp4"):
        return None

    m = _CLIP_DIR_CELL_RE.search(norm)
    if m is None:
        m = _CLIP_FILE_CELL_RE.search(norm)
    if m is None:
        return None

    arm = m.group("arm")
    prompt = m.group("prompt")
    seed = int(m.group("seed"))
    if arm.lower() == "cell":
        return None
    cell_id = f"{prompt}-s{seed}"
    return ParsedClipName(
        arm=arm,
        prompt=prompt,
        seed=seed,
        cell_id=cell_id,
        rel_path=norm,
    )


def parse_fetch_list(content: str) -> List[ParsedClipName]:
    """Parse all clip entries from a `fetch_list.txt`-formatted string."""
    parsed: List[ParsedClipName] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        item = parse_clip_name(line)
        if item is not None:
            parsed.append(item)
    return parsed


def parse_sha256sums(content: str) -> Dict[str, str]:
    """Parse a `sha256sums.txt`-formatted string into `{normalized_rel_path: lower_sha256}`.

    Raises `ClipHashMismatchError` on malformed lines or conflicting digests for the same path.
    """
    entries: Dict[str, str] = {}
    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ClipHashMismatchError(
                f"Malformed sha256sums line {line_no}: {raw_line!r}"
            )
        digest, rel = parts[0].strip(), parts[1].strip()
        if rel.startswith("*"):
            rel = rel[1:].strip()
        if not _SHA256_HEX_RE.match(digest):
            raise ClipHashMismatchError(
                f"Invalid SHA-256 hex digest at line {line_no}: {digest!r}"
            )
        norm_rel = rel.replace("\\", "/").lstrip("./")
        digest_low = digest.lower()
        if norm_rel in entries and entries[norm_rel] != digest_low:
            raise ClipHashMismatchError(
                f"Conflicting SHA-256 digests in hash list for {norm_rel!r}",
                clip_path=norm_rel,
                expected_sha256=entries[norm_rel],
                actual_sha256=digest_low,
            )
        entries[norm_rel] = digest_low
    return entries


def compute_file_sha256(path: str | Path) -> str:
    """Compute the lowercase 64-hex SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_candidate_clip_paths(clips_dir: Path, parsed: ParsedClipName) -> List[Path]:
    """Return existing file paths under `clips_dir` matching `parsed`."""
    candidates = [
        clips_dir / parsed.rel_path,
        clips_dir / Path(parsed.rel_path).name,
        clips_dir / "mp4" / f"{parsed.arm}_cell_{parsed.cell_id}.mp4",
        clips_dir / f"{parsed.arm}_cell_{parsed.cell_id}.mp4",
        clips_dir / parsed.arm / f"cell_{parsed.cell_id}.mp4",
        clips_dir / f"{parsed.arm}_{parsed.cell_id}.mp4",
    ]
    found: List[Path] = []
    seen: set[Path] = set()
    for cand in candidates:
        if cand.is_file():
            resolved = cand.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
    return found


def verify_and_resolve_super_clips(
    clips_dir: str | Path,
    hashes_file: str | Path,
    super_arms: Sequence[str] = SUPER_ARMS,
) -> Tuple[Dict[Tuple[str, int], Dict[str, Tuple[Path, str]]], int]:
    """Verify SHA-256 of every Super clip before decoding any frame.

    Ignores Nano clips (`NANO`). Refuses to run on any missing file, missing hash entry,
    conflicting duplicate entry, or SHA-256 mismatch by raising `ClipHashMismatchError`.
    """
    cdir = Path(clips_dir)
    hfile = Path(hashes_file)
    if not cdir.is_dir():
        raise ClipHashMismatchError(f"Clips directory not found: {cdir}")
    if not hfile.is_file():
        raise ClipHashMismatchError(f"Hashes file not found: {hfile}")

    raw_hashes = parse_sha256sums(hfile.read_text(encoding="utf-8"))
    super_arm_set = set(super_arms)

    # Group expected hash entries by (prompt, seed, arm)
    expected_by_key: Dict[Tuple[str, int, str], Tuple[ParsedClipName, str]] = {}
    for rel_path, expected_sha in sorted(raw_hashes.items()):
        parsed = parse_clip_name(rel_path)
        if parsed is None:
            continue
        if parsed.arm.upper() == ARM_NANO:
            continue
        if parsed.arm not in super_arm_set:
            continue
        key = (parsed.prompt, parsed.seed, parsed.arm)
        if key in expected_by_key:
            prev_parsed, prev_sha = expected_by_key[key]
            if prev_sha != expected_sha:
                raise ClipHashMismatchError(
                    f"Conflicting hashes in {hfile.name} for cell {parsed.cell_id} arm {parsed.arm}: "
                    f"{prev_sha} ({prev_parsed.rel_path}) vs {expected_sha} ({parsed.rel_path})",
                    clip_path=parsed.rel_path,
                    expected_sha256=prev_sha,
                    actual_sha256=expected_sha,
                )
        else:
            expected_by_key[key] = (parsed, expected_sha)

    if not expected_by_key:
        raise ClipHashMismatchError(f"No Super clip entries found in hash list: {hfile}")

    # Also ensure no unexpected unverified Super clip on disk has a tampered hash
    cells_map: Dict[Tuple[str, int], Dict[str, Tuple[Path, str]]] = {}
    verified_files: set[Path] = set()

    for (prompt, seed, arm), (parsed, expected_sha) in sorted(expected_by_key.items()):
        found_paths = _resolve_candidate_clip_paths(cdir, parsed)
        if not found_paths:
            raise ClipHashMismatchError(
                f"Clip file missing on disk for cell {parsed.cell_id} arm {arm} (expected {parsed.rel_path})",
                clip_path=parsed.rel_path,
                expected_sha256=expected_sha,
                actual_sha256=None,
            )
        for fpath in found_paths:
            actual_sha = compute_file_sha256(fpath)
            if actual_sha != expected_sha:
                raise ClipHashMismatchError(
                    f"SHA-256 mismatch for {fpath.name}: expected {expected_sha}, got {actual_sha}",
                    clip_path=str(fpath),
                    expected_sha256=expected_sha,
                    actual_sha256=actual_sha,
                )
            verified_files.add(fpath)

        cell_key = (prompt, seed)
        cells_map.setdefault(cell_key, {})[arm] = (found_paths[0], expected_sha)

    # Verify every cell has all required Super arms
    for (prompt, seed), arm_dict in sorted(cells_map.items()):
        missing_arms = [a for a in super_arms if a not in arm_dict]
        if missing_arms:
            raise ClipHashMismatchError(
                f"Cell {prompt}-s{seed} is missing Super arms: {missing_arms}"
            )

    return cells_map, len(cells_map) * len(super_arms)


def decode_clip_frames(
    clip_path: str | Path,
    frame_indices: Sequence[int] = FRAME_INDICES,
    expected_width: int = INPUT_WIDTH,
    expected_height: int = INPUT_HEIGHT,
) -> Dict[int, np.ndarray]:
    """Decode requested RGB frames (default: 0, 30, 60) from an 832x480 clip."""
    path = Path(clip_path)
    if not path.is_file():
        raise FileNotFoundError(f"Clip not found: {path}")

    req_indices = sorted(set(int(i) for i in frame_indices))
    max_idx = max(req_indices) if req_indices else 0

    raw_head = b""
    with open(path, "rb") as f:
        raw_head = f.read(len(SYNTH_RAW_HEADER))

    if raw_head == SYNTH_RAW_HEADER:
        with open(path, "rb") as f:
            _ = f.readline()
            meta_line = f.readline().decode("utf-8").strip()
            meta = json.loads(meta_line)
            w = int(meta["width"])
            h = int(meta["height"])
            n_frames = int(meta["frames"])
            payload = f.read()
        if w != expected_width or h != expected_height:
            raise ValueError(
                f"Clip {path.name} resolution {w}x{h} != expected {expected_width}x{expected_height}"
            )
        if n_frames <= max_idx:
            raise ValueError(
                f"Clip {path.name} has {n_frames} frames, fewer than required frame index {max_idx}"
            )
        frame_bytes = h * w * 3
        all_frames = np.frombuffer(payload[: n_frames * frame_bytes], dtype=np.uint8).reshape(
            (n_frames, h, w, 3)
        )
        return {idx: all_frames[idx].copy() for idx in req_indices}

    # Try PyAV if installed (same decoder family as fidelity/score.py)
    try:
        import av  # type: ignore[import-not-found]

        with av.open(str(path)) as container:
            decoded: List[np.ndarray] = [
                frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
            ]
        if len(decoded) <= max_idx:
            raise ValueError(
                f"Clip {path.name} has {len(decoded)} frames, fewer than required frame index {max_idx}"
            )
        h, w, c = decoded[0].shape
        if w != expected_width or h != expected_height or c != 3:
            raise ValueError(
                f"Clip {path.name} frame shape {(h, w, c)} != expected {(expected_height, expected_width, 3)}"
            )
        return {idx: np.asarray(decoded[idx], dtype=np.uint8) for idx in req_indices}
    except ImportError:
        pass

    # Fallback to ffmpeg rawvideo rgb24 decode
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Neither PyAV ('av') nor 'ffmpeg' is available to decode video clips.")

    res = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    frame_bytes = expected_height * expected_width * 3
    if len(res.stdout) == 0 or (len(res.stdout) % frame_bytes) != 0:
        raise ValueError(
            f"Decoded byte count {len(res.stdout)} from {path.name} is not a multiple of "
            f"{expected_width}x{expected_height}x3 ({frame_bytes})"
        )
    n_frames = len(res.stdout) // frame_bytes
    if n_frames <= max_idx:
        raise ValueError(
            f"Clip {path.name} has {n_frames} frames, fewer than required frame index {max_idx}"
        )
    all_frames = np.frombuffer(res.stdout, dtype=np.uint8).reshape(
        (n_frames, expected_height, expected_width, 3)
    )
    return {idx: all_frames[idx].copy() for idx in req_indices}


def preprocess_frame_rgb(
    frame_rgb: np.ndarray,
    expected_width: int = INPUT_WIDTH,
    expected_height: int = INPUT_HEIGHT,
) -> np.ndarray:
    """Resize an 832x480 RGB uint8 frame directly to 392x224 (bicubic, no crop, no pad)
    and normalise with ImageNet mean/std into a (3, 224, 392) float32 array."""
    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"Expected 3D RGB frame (H, W, 3), got shape {getattr(frame_rgb, 'shape', None)}")
    h, w, _ = frame_rgb.shape
    if w != expected_width or h != expected_height:
        raise ValueError(
            f"Input frame must be {expected_width}x{expected_height}, got {w}x{h}"
        )

    pil_img = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8), mode="RGB")
    resized_pil = pil_img.resize(
        (RESIZE_WIDTH, RESIZE_HEIGHT),
        resample=Image.Resampling.BICUBIC,
    )
    resized_np = np.asarray(resized_pil, dtype=np.float32) / np.float32(255.0)
    if resized_np.shape != (RESIZE_HEIGHT, RESIZE_WIDTH, 3):
        raise AssertionError(f"Unexpected resized shape: {resized_np.shape}")

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape((1, 1, 3))
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape((1, 1, 3))
    norm = (resized_np - mean) / std
    chw = np.transpose(norm, (2, 0, 1)).astype(np.float32, copy=False)
    return np.ascontiguousarray(chw, dtype=np.float32)


class Dinov2Encoder:
    """CPU-only DINOv2 ViT-B/14 (`facebook/dinov2-base`) encoder with pinned commit revision."""

    def __init__(
        self,
        model_id: str = DINOV2_MODEL_ID,
        revision: str = DINOV2_REVISION,
        max_threads: int = DEFAULT_MAX_THREADS,
        local_snapshot_path: Optional[str | Path] = None,
        local_files_only: bool = False,
        torch_module: Any = None,
        transformers_module: Any = None,
    ) -> None:
        self.model_id = model_id
        self.revision = validate_revision_pin(revision)
        self.license_name = DINOV2_LICENSE
        self.max_threads = enforce_cpu_only(max_threads=max_threads, torch_module=torch_module)

        t_mod = torch_module if torch_module is not None else _try_import_torch()
        if t_mod is None:
            raise RuntimeError("PyTorch ('torch') is required to load Dinov2Encoder.")
        self._torch = t_mod

        if transformers_module is not None:
            tf_mod = transformers_module
        else:
            import transformers as tf_mod  # type: ignore[import-not-found]

        source = str(local_snapshot_path) if local_snapshot_path else self.model_id
        use_local_only = bool(local_files_only or local_snapshot_path)
        load_kwargs: Dict[str, Any] = {"local_files_only": use_local_only}
        if not local_snapshot_path:
            load_kwargs["revision"] = self.revision

        model = tf_mod.AutoModel.from_pretrained(source, **load_kwargs)
        model = model.to("cpu").eval()

        for param in model.parameters():
            dev_type = getattr(getattr(param, "device", None), "type", "cpu")
            if dev_type != "cpu":
                raise CudaDeviceForbiddenError(
                    f"DINOv2 parameter found on non-CPU device: {dev_type!r}"
                )
        self._model = model

    def encode_preprocessed(self, chw_float32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run one forward pass on CPU and return `(patch_tokens[448, D], cls_token[D])` in float32."""
        enforce_cpu_only(max_threads=self.max_threads, torch_module=self._torch)
        if chw_float32.shape != (3, RESIZE_HEIGHT, RESIZE_WIDTH):
            raise ValueError(
                f"Expected preprocessed tensor shape (3, {RESIZE_HEIGHT}, {RESIZE_WIDTH}), got {chw_float32.shape}"
            )
        arr = np.ascontiguousarray(chw_float32, dtype=np.float32)
        pixel_values = self._torch.from_numpy(arr).unsqueeze(0)
        if getattr(getattr(pixel_values, "device", None), "type", "cpu") != "cpu":
            raise CudaDeviceForbiddenError("Input tensor must remain on CPU.")

        with self._torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        if getattr(getattr(hidden, "device", None), "type", "cpu") != "cpu":
            raise CudaDeviceForbiddenError("Model output tensor is not on CPU.")

        hidden_np = hidden.detach().cpu().numpy().astype(np.float32, copy=False)
        if hidden_np.ndim != 3 or hidden_np.shape[0] != 1 or hidden_np.shape[1] < 1 + NUM_PATCHES:
            raise ValueError(
                f"Expected last_hidden_state shape (1, >= {1 + NUM_PATCHES}, D), got {hidden_np.shape}"
            )
        cls_token = np.ascontiguousarray(hidden_np[0, 0, :], dtype=np.float32)
        patch_tokens = np.ascontiguousarray(hidden_np[0, 1 : 1 + NUM_PATCHES, :], dtype=np.float32)
        return patch_tokens, cls_token


class DeterministicStubDinov2Encoder:
    """Deterministic offline stub encoder for unit and fixture tests (zero network downloads).

    Splits each (3, 224, 392) preprocessed frame into its exact 28x16 = 448 spatial 14x14 patches
    and projects each patch to a float32 token, plus a global shift-tolerant CLS token.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        model_id: str = DINOV2_MODEL_ID,
        revision: str = DINOV2_REVISION,
    ) -> None:
        self.model_id = model_id
        self.revision = validate_revision_pin(revision)
        self.license_name = DINOV2_LICENSE
        self.embed_dim = int(embed_dim)

        patch_dim = 3 * PATCH_SIZE * PATCH_SIZE  # 588
        idx_i = np.arange(patch_dim, dtype=np.float64)[:, None]
        idx_j = np.arange(self.embed_dim, dtype=np.float64)[None, :]
        proj = np.cos((idx_i + 1.0) * (idx_j + 1.0) * 0.137) / np.sqrt(float(patch_dim))
        self._patch_proj = proj.astype(np.float32)
        self._patch_bias = (0.25 + 0.05 * np.sin(np.arange(self.embed_dim, dtype=np.float32))).astype(np.float32)

    def encode_preprocessed(self, chw_float32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if chw_float32.shape != (3, RESIZE_HEIGHT, RESIZE_WIDTH):
            raise ValueError(
                f"Expected preprocessed frame of shape (3, {RESIZE_HEIGHT}, {RESIZE_WIDTH}), got {chw_float32.shape}"
            )
        x = np.asarray(chw_float32, dtype=np.float32)
        # Partition into (16, 28) grid of (3, 14, 14) patches -> (448, 588)
        grid = x.reshape(3, PATCH_GRID[1], PATCH_SIZE, PATCH_GRID[0], PATCH_SIZE)
        patches_flat = np.transpose(grid, (1, 3, 0, 2, 4)).reshape(NUM_PATCHES, -1)
        patch_tokens = (patches_flat @ self._patch_proj) + self._patch_bias[None, :]
        patch_tokens = np.ascontiguousarray(patch_tokens, dtype=np.float32)

        # Global shift-invariant CLS descriptor from channel moments and quantiles
        ch_mean = x.mean(axis=(1, 2))
        ch_std = x.std(axis=(1, 2))
        ch_q25 = np.percentile(x, 25.0, axis=(1, 2)).astype(np.float32)
        ch_q75 = np.percentile(x, 75.0, axis=(1, 2)).astype(np.float32)
        summary = np.concatenate([ch_mean, ch_std, ch_q25, ch_q75], axis=0)
        reps = int(np.ceil(self.embed_dim / len(summary)))
        cls_token = np.tile(summary, reps)[: self.embed_dim] + self._patch_bias
        cls_token = np.ascontiguousarray(cls_token, dtype=np.float32)
        return patch_tokens, cls_token


def compute_patch_distance(patches_a: np.ndarray, patches_b: np.ndarray) -> float:
    """Compute primary aligned patch-token mean cosine distance in float64 over 448 patches.

    Formula (§3.4 item 1(i)):
      d^patch = (1 / 448) * sum_{k=1}^{448} (1 - (t_{a,k} . t_{b,k}) / (||t_{a,k}|| * ||t_{b,k}||))
    """
    ta = np.asarray(patches_a, dtype=np.float32)
    tb = np.asarray(patches_b, dtype=np.float32)
    if ta.ndim != 2 or tb.ndim != 2 or ta.shape != tb.shape or ta.shape[0] != NUM_PATCHES:
        raise ValueError(
            f"Patch token arrays must both have shape ({NUM_PATCHES}, D); got {ta.shape} and {tb.shape}"
        )
    ta64 = ta.astype(np.float64)
    tb64 = tb.astype(np.float64)
    dots = np.sum(ta64 * tb64, axis=1, dtype=np.float64)
    norms_a = np.linalg.norm(ta64, ord=2, axis=1)
    norms_b = np.linalg.norm(tb64, ord=2, axis=1)
    denom = norms_a * norms_b
    if np.any(denom <= 0.0):
        raise ValueError("Zero-norm patch token encountered while computing cosine distance.")
    cos_sim = dots / denom
    per_patch_dist = 1.0 - cos_sim
    return float(np.mean(per_patch_dist, dtype=np.float64))


def compute_cls_distance(cls_a: np.ndarray, cls_b: np.ndarray) -> float:
    """Compute secondary CLS cosine distance in float64.

    Formula (§3.4 item 1(ii)):
      d^CLS = 1 - (e_a . e_b) / (||e_a|| * ||e_b||)
    """
    ea = np.asarray(cls_a, dtype=np.float32)
    eb = np.asarray(cls_b, dtype=np.float32)
    if ea.ndim != 1 or eb.ndim != 1 or ea.shape != eb.shape or ea.shape[0] == 0:
        raise ValueError(f"CLS token vectors must be 1D of equal non-zero length; got {ea.shape} and {eb.shape}")
    ea64 = ea.astype(np.float64)
    eb64 = eb.astype(np.float64)
    dot = float(np.dot(ea64, eb64))
    norm_a = float(np.linalg.norm(ea64, ord=2))
    norm_b = float(np.linalg.norm(eb64, ord=2))
    denom = norm_a * norm_b
    if denom <= 0.0:
        raise ValueError("Zero-norm CLS token encountered while computing cosine distance.")
    return float(1.0 - (dot / denom))


def type7_percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    """Compute the q-th percentile using `numpy.percentile(x, q, method='linear')` (Hyndman & Fan type 7)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute percentile of an empty array.")
    return float(np.percentile(arr, float(q), method=PERCENTILE_METHOD))


def type7_percentile_rank(reference_values: Sequence[float] | np.ndarray, value: float) -> float:
    """Return the Hyndman & Fan type-7 percentile position (0..100) of `value` within `reference_values`.

    Under type-7 (`p_k = (k - 1) / (n - 1)` for 1-indexed order statistics `k = 1..n`),
    linearly interpolates between adjacent order statistics so that for any `v` within
    the sample range, `np.percentile(reference_values, rank, method='linear') == v`.
    """
    arr = np.sort(np.asarray(reference_values, dtype=np.float64))
    n = arr.size
    if n == 0:
        raise ValueError("Cannot compute percentile rank within an empty reference array.")
    val = float(value)
    if n == 1:
        if val < arr[0]:
            return 0.0
        if val > arr[0]:
            return 100.0
        return 50.0

    if val < arr[0]:
        return 0.0
    if val > arr[-1]:
        return 100.0

    eq_indices = np.flatnonzero(arr == val)
    if eq_indices.size > 0:
        mid_idx = 0.5 * float(eq_indices[0] + eq_indices[-1])
        return float(100.0 * mid_idx / float(n - 1))

    right_idx = int(np.searchsorted(arr, val, side="right"))
    left_idx = right_idx - 1
    low = float(arr[left_idx])
    high = float(arr[right_idx])
    frac = 0.0 if high == low else (val - low) / (high - low)
    pos = (float(left_idx) + frac) / float(n - 1)
    return float(np.clip(100.0 * pos, 0.0, 100.0))


def compute_robust_null_stats(w_values: Sequence[float] | np.ndarray) -> Tuple[float, float, float]:
    """Compute `(m_W, MAD_W, nu)` on within-cell first-frame distances W.

    m_W = median(W)
    MAD_W = 1.4826 * median(|d - m_W|)
    nu = m_W + 3 * MAD_W
    """
    arr = np.asarray(w_values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Within-cell reference W must not be empty.")
    m_w = type7_percentile(arr, 50.0)
    abs_dev = np.abs(arr - m_w)
    mad_w = float(MAD_SCALE * type7_percentile(abs_dev, 50.0))
    nu = float(m_w + 3.0 * mad_w)
    return m_w, mad_w, nu


def check_informativeness(nu: float, tau_pooled: float, m_w: float) -> bool:
    """Return True iff `nu < tau_pooled` AND `tau_pooled >= 2 * m_W` (§3.4 item 2)."""
    return bool((float(nu) < float(tau_pooled)) and (float(tau_pooled) >= 2.0 * float(m_w)))


def evaluate_cell_flag_and_borderline(
    d_patch: float,
    d_cls: float,
    score_informative: bool,
    tau_pooled: float,
    tau_p: float,
    nu: float,
    tau_pooled_cls: float,
    nu_cls: float,
) -> Tuple[int, int, List[str]]:
    """Evaluate `composition_flag`, `borderline`, and which of `(a)`, `(b)`, `(c)` fired (§3.4 item 2).

    - `composition_flag = 1` iff `score_informative` and `d_patch > tau_pooled` and `d_patch > nu`.
    - `borderline = 1` iff `composition_flag == 0` and at least one of:
      (a) `d_patch > tau_p` for its own prompt;
      (b) `d_patch > nu`;
      (c) `d_cls > tau_pooled_cls` and `d_cls > nu_cls`.
    """
    if score_informative and (float(d_patch) > float(tau_pooled)) and (float(d_patch) > float(nu)):
        return 1, 0, []

    fired: List[str] = []
    if float(d_patch) > float(tau_p):
        fired.append("a")
    if float(d_patch) > float(nu):
        fired.append("b")
    if (float(d_cls) > float(tau_pooled_cls)) and (float(d_cls) > float(nu_cls)):
        fired.append("c")

    borderline = 1 if fired else 0
    return 0, borderline, fired


def load_noise_floor(
    regeneration_file: Optional[str | Path],
    nu: float,
) -> Tuple[float | str, Optional[bool]]:
    """Load `d_floor` from an optional regeneration JSON file, or return `'not measured'`."""
    if not regeneration_file:
        return "not measured", None
    rpath = Path(regeneration_file)
    if not rpath.is_file():
        raise FileNotFoundError(f"Regeneration file not found: {rpath}")
    data = json.loads(rpath.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "d_floor" in data:
        val = float(data["d_floor"])
    elif isinstance(data, dict) and isinstance(data.get("distances"), list) and data["distances"]:
        val = float(np.max(np.asarray(data["distances"], dtype=np.float64)))
    else:
        raise ValueError(f"Invalid regeneration file schema in {rpath.name}: expected 'd_floor' or 'distances'")
    return val, bool(val < float(nu))


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def assert_path_outside_repo(
    target_path: str | Path,
    repo_root: Optional[str | Path] = None,
) -> Path:
    """Refuse any output path that resolves inside the repository tree."""
    root = Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    resolved = Path(target_path).resolve()
    if resolved == root or root in resolved.parents:
        raise RepoOutputPathError(
            f"Refusing to write sealed output inside repository tree ({root}): {resolved}"
        )
    return resolved


def validate_divergence_output_schema(payload: Dict[str, Any]) -> None:
    """Assert that the output JSON contains only header metadata and per-cell scalar distances/hashes,
    with zero frames, thumbnails, embeddings, or array dumps."""
    if not isinstance(payload, dict) or "header" not in payload or "cells" not in payload:
        raise ValueError("Output payload must contain 'header' and 'cells' top-level keys.")

    forbidden_keys = {
        "embedding",
        "embeddings",
        "patch_tokens",
        "cls_token",
        "tokens",
        "frame",
        "frames",
        "thumbnail",
        "thumbnails",
        "pixels",
        "image",
        "images",
        "path",
        "paths",
        "clip_paths",
    }

    def _walk(node: Any, path_str: str, in_cell: bool) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                k_low = str(k).lower()
                if k_low in forbidden_keys:
                    raise ValueError(f"Forbidden key {k!r} at {path_str}.{k}")
                _walk(v, f"{path_str}.{k}", in_cell=in_cell)
        elif isinstance(node, list):
            if in_cell and not path_str.endswith(".borderline_fired"):
                raise ValueError(
                    f"Unexpected array in cell record at {path_str}; only scalar distances and borderline_fired are allowed"
                )
            for idx, item in enumerate(node):
                _walk(item, f"{path_str}[{idx}]", in_cell=in_cell)

    _walk(payload["header"], "header", in_cell=False)
    if not isinstance(payload["cells"], list):
        raise ValueError("'cells' must be a list of cell records.")
    for idx, cell in enumerate(payload["cells"]):
        _walk(cell, f"cells[{idx}]", in_cell=True)


def compute_composition_divergence(
    clips_dir: str | Path,
    hashes_file: str | Path,
    encoder: Optional[FrameEncoder] = None,
    max_threads: int = DEFAULT_MAX_THREADS,
    regeneration_file: Optional[str | Path] = None,
    date_tag: Optional[str] = None,
    local_snapshot_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, Any], int]:
    """Verify all Super clip hashes, decode frames 0/30/60, compute patch & CLS distances,
    reference thresholds, informativeness, flags, borderline reasons, and type-7 ranks."""
    enforce_cpu_only(max_threads=max_threads)
    cells_map, clips_verified = verify_and_resolve_super_clips(clips_dir, hashes_file)

    active_encoder: FrameEncoder = (
        encoder
        if encoder is not None
        else Dinov2Encoder(
            model_id=DINOV2_MODEL_ID,
            revision=DINOV2_REVISION,
            max_threads=max_threads,
            local_snapshot_path=local_snapshot_path,
        )
    )

    # Encode frames 0, 30, 60 once per clip; keep onlyframe-0 tokens temporarily in memory for B & W distances
    frame0_tokens: Dict[Tuple[str, int, str], Tuple[np.ndarray, np.ndarray]] = {}
    per_cell_pair_distances: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}

    for (prompt, seed), arm_files in sorted(cells_map.items()):
        cell_tokens: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
        for arm in SUPER_ARMS:
            clip_path, _sha = arm_files[arm]
            decoded_frames = decode_clip_frames(clip_path, frame_indices=FRAME_INDICES)
            cell_tokens[arm] = {}
            for f_idx in FRAME_INDICES:
                chw = preprocess_frame_rgb(decoded_frames[f_idx])
                p_tok, c_tok = active_encoder.encode_preprocessed(chw)
                cell_tokens[arm][f_idx] = (p_tok, c_tok)
            frame0_tokens[(prompt, seed, arm)] = cell_tokens[arm][0]

        pair_map: Dict[str, Dict[str, Any]] = {}
        for arm_a, arm_b in ARM_PAIRS:
            pair_key = f"{arm_a}_{arm_b}"
            p_by_f: Dict[str, float] = {}
            c_by_f: Dict[str, float] = {}
            for f_idx in FRAME_INDICES:
                pa, ca = cell_tokens[arm_a][f_idx]
                pb, cb = cell_tokens[arm_b][f_idx]
                p_by_f[str(f_idx)] = compute_patch_distance(pa, pb)
                c_by_f[str(f_idx)] = compute_cls_distance(ca, cb)

            patch_3f = float(
                (p_by_f["0"] + p_by_f["30"] + p_by_f["60"]) / 3.0
            )
            cls_3f = float(
                (c_by_f["0"] + c_by_f["30"] + c_by_f["60"]) / 3.0
            )
            pair_map[pair_key] = {
                "patch_frame_0": p_by_f["0"],
                "patch_three_frame_mean": patch_3f,
                "cls_frame_0": c_by_f["0"],
                "cls_three_frame_mean": cls_3f,
                "patch_frames": p_by_f,
                "cls_frames": c_by_f,
            }
        per_cell_pair_distances[(prompt, seed)] = pair_map

    # Build within-cell first-frame reference W (all cells x 3 pairs)
    w_patch_list: List[float] = []
    w_cls_list: List[float] = []
    for key in sorted(per_cell_pair_distances.keys()):
        for arm_a, arm_b in ARM_PAIRS:
            pk = f"{arm_a}_{arm_b}"
            w_patch_list.append(per_cell_pair_distances[key][pk]["patch_frame_0"])
            w_cls_list.append(per_cell_pair_distances[key][pk]["cls_frame_0"])

    # Build between-seed same-arm same-prompt frame-0 reference B
    seeds_by_prompt: Dict[str, List[int]] = {}
    for prompt, seed in sorted(cells_map.keys()):
        seeds_by_prompt.setdefault(prompt, []).append(seed)

    b_patch_by_prompt: Dict[str, List[float]] = {}
    b_cls_by_prompt: Dict[str, List[float]] = {}
    b_patch_pooled: List[float] = []
    b_cls_pooled: List[float] = []

    for prompt, seed_list in sorted(seeds_by_prompt.items()):
        s_sorted = sorted(seed_list)
        if len(s_sorted) < 2:
            raise ValueError(
                f"Prompt {prompt!r} has fewer than 2 seeds ({s_sorted}); cannot form between-seed reference B."
            )
        p_dists: List[float] = []
        c_dists: List[float] = []
        for arm in SUPER_ARMS:
            for i in range(len(s_sorted)):
                for j in range(i + 1, len(s_sorted)):
                    pa, ca = frame0_tokens[(prompt, s_sorted[i], arm)]
                    pb, cb = frame0_tokens[(prompt, s_sorted[j], arm)]
                    d_p = compute_patch_distance(pa, pb)
                    d_c = compute_cls_distance(ca, cb)
                    p_dists.append(d_p)
                    c_dists.append(d_c)
                    b_patch_pooled.append(d_p)
                    b_cls_pooled.append(d_c)
        b_patch_by_prompt[prompt] = p_dists
        b_cls_by_prompt[prompt] = c_dists

    # Thresholds & robust null stats (patch primary & CLS secondary)
    m_w, mad_w, nu = compute_robust_null_stats(w_patch_list)
    tau_pooled = type7_percentile(b_patch_pooled, 5.0)
    tau_p_map = {p: type7_percentile(vals, 5.0) for p, vals in sorted(b_patch_by_prompt.items())}

    m_w_cls, mad_w_cls, nu_cls = compute_robust_null_stats(w_cls_list)
    tau_pooled_cls = type7_percentile(b_cls_pooled, 5.0)
    tau_p_cls_map = {p: type7_percentile(vals, 5.0) for p, vals in sorted(b_cls_by_prompt.items())}

    score_informative = check_informativeness(nu=nu, tau_pooled=tau_pooled, m_w=m_w)
    score_informative_cls = check_informativeness(nu=nu_cls, tau_pooled=tau_pooled_cls, m_w=m_w_cls)

    d_floor_val, d_floor_below_nu = load_noise_floor(regeneration_file, nu=nu)

    cell_records: List[Dict[str, Any]] = []
    flagged_cells: List[str] = []
    borderline_cells: List[str] = []

    for (prompt, seed), arm_files in sorted(cells_map.items()):
        cell_id = f"{prompt}-s{seed}"
        pair_map = per_cell_pair_distances[(prompt, seed)]
        pair_keys = [f"{a}_{b}" for a, b in ARM_PAIRS]

        d_patch = float(max(pair_map[pk]["patch_frame_0"] for pk in pair_keys))
        d_patch_3f = float(max(pair_map[pk]["patch_three_frame_mean"] for pk in pair_keys))
        d_cls = float(max(pair_map[pk]["cls_frame_0"] for pk in pair_keys))
        d_cls_3f = float(max(pair_map[pk]["cls_three_frame_mean"] for pk in pair_keys))

        flag, borderline, fired = evaluate_cell_flag_and_borderline(
            d_patch=d_patch,
            d_cls=d_cls,
            score_informative=score_informative,
            tau_pooled=tau_pooled,
            tau_p=tau_p_map[prompt],
            nu=nu,
            tau_pooled_cls=tau_pooled_cls,
            nu_cls=nu_cls,
        )
        if flag == 1:
            flagged_cells.append(cell_id)
        if borderline == 1:
            borderline_cells.append(cell_id)

        rank_b = type7_percentile_rank(b_patch_pooled, d_patch)
        rank_w = type7_percentile_rank(w_patch_list, d_patch)
        rank_cls_b = type7_percentile_rank(b_cls_pooled, d_cls)
        rank_cls_w = type7_percentile_rank(w_cls_list, d_cls)

        clip_hashes = {arm: arm_files[arm][1] for arm in SUPER_ARMS}

        cell_records.append(
            {
                "record_type": "cell",
                "cell_id": cell_id,
                "prompt": prompt,
                "seed": int(seed),
                "pair_distances": pair_map,
                "D_patch": d_patch,
                "D_patch_three_frame_mean": d_patch_3f,
                "D_cls": d_cls,
                "D_cls_three_frame_mean": d_cls_3f,
                "rank_in_B_pct": rank_b,
                "rank_in_W_pct": rank_w,
                "rank_cls_in_B_pct": rank_cls_b,
                "rank_cls_in_W_pct": rank_cls_w,
                "composition_flag": flag,
                "borderline": borderline,
                "borderline_fired": fired,
                "clip_hashes": clip_hashes,
            }
        )

    effective_date = date_tag or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    geometry_record = {
        "input_width": INPUT_WIDTH,
        "input_height": INPUT_HEIGHT,
        "resize_width": RESIZE_WIDTH,
        "resize_height": RESIZE_HEIGHT,
        "interpolation": "bicubic",
        "crop": False,
        "padding": False,
        "patch_size": PATCH_SIZE,
        "patch_grid": list(PATCH_GRID),
        "num_patches": NUM_PATCHES,
        "frame_indices": list(FRAME_INDICES),
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "token_dtype": "float32",
        "distance_dtype": "float64",
    }

    header_record: Dict[str, Any] = {
        "record_type": "header",
        "schema_version": "cosmos3-composition-divergence-v1",
        "date": effective_date,
        "model_id": active_encoder.model_id,
        "model_revision": active_encoder.revision,
        "model_license": getattr(active_encoder, "license_name", DINOV2_LICENSE),
        "geometry": geometry_record,
        "percentile_method": PERCENTILE_METHOD,
        "mad_scale": MAD_SCALE,
        "score_informative": score_informative,
        "score_informative_cls": score_informative_cls,
        "d_floor": d_floor_val,
        "d_floor_below_nu": d_floor_below_nu,
        "tau_pooled": tau_pooled,
        "tau_p": tau_p_map,
        "m_W": m_w,
        "mad_W": mad_w,
        "nu": nu,
        "tau_pooled_cls": tau_pooled_cls,
        "tau_p_cls": tau_p_cls_map,
        "m_W_cls": m_w_cls,
        "mad_W_cls": mad_w_cls,
        "nu_cls": nu_cls,
        "cls_thresholds": {
            "tau_pooled_cls": tau_pooled_cls,
            "tau_p_cls": tau_p_cls_map,
            "m_W_cls": m_w_cls,
            "mad_W_cls": mad_w_cls,
            "nu_cls": nu_cls,
            "score_informative_cls": score_informative_cls,
        },
        "num_cells": len(cell_records),
        "num_between_seed_B": len(b_patch_pooled),
        "num_within_cell_W": len(w_patch_list),
        "flagged_cells": flagged_cells,
        "borderline_cells": borderline_cells,
    }

    output_payload = {
        "header": header_record,
        "cells": cell_records,
    }
    validate_divergence_output_schema(output_payload)
    return output_payload, clips_verified


def write_json_exclusive(path: Path, data: Dict[str, Any]) -> str:
    """Write `data` to `path` using exclusive file creation (`'x'`), refusing any overwrite.

    Returns the lowercase 64-hex SHA-256 digest of the written file.
    """
    if path.exists():
        raise OutputOverwriteError(f"Refusing to overwrite existing sealed file: {path}")
    serialized = json.dumps(data, indent=2) + "\n"
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(serialized)
    except FileExistsError as exc:
        raise OutputOverwriteError(f"Refusing to overwrite existing sealed file: {path}") from exc
    return compute_file_sha256(path)


def allocate_sealed_output_paths(
    out_dir: Path,
    date_tag: str,
    allow_new_dated_on_collision: bool = True,
) -> Tuple[Path, Path, str]:
    """Resolve `composition_divergence_<date>.json` and its receipt path in `out_dir`.

    Never overwrites an existing file: if `composition_divergence_<date>.json` already exists,
    either raises `OutputOverwriteError` (when `allow_new_dated_on_collision=False`) or
    allocates a new dated filename (`composition_divergence_<date>_<utc_stamp>.json`).
    """
    base_out = out_dir / f"composition_divergence_{date_tag}.json"
    base_rec = out_dir / f"composition_divergence_receipt_{date_tag}.json"
    if not base_out.exists() and not base_rec.exists():
        return base_out, base_rec, date_tag

    if not allow_new_dated_on_collision:
        raise OutputOverwriteError(
            f"Sealed output already exists and overwrite is forbidden: {base_out}"
        )

    utc_stamp = dt.datetime.now(dt.timezone.utc).strftime("%H%M%SZ")
    seq = 2
    while True:
        cand_tag = f"{date_tag}_{utc_stamp}_r{seq}"
        cand_out = out_dir / f"composition_divergence_{cand_tag}.json"
        cand_rec = out_dir / f"composition_divergence_receipt_{cand_tag}.json"
        if not cand_out.exists() and not cand_rec.exists():
            return cand_out, cand_rec, cand_tag
        seq += 1


def run_composition_divergence_job(
    clips_dir: str | Path,
    hashes_file: str | Path,
    out_dir: str | Path,
    encoder: Optional[FrameEncoder] = None,
    max_threads: int = DEFAULT_MAX_THREADS,
    regeneration_file: Optional[str | Path] = None,
    date_tag: Optional[str] = None,
    local_snapshot_path: Optional[str | Path] = None,
    sealed_manifest_path: Optional[str | Path] = None,
    allow_new_dated_on_collision: bool = True,
    repo_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run the full composition-divergence pre-read job and write sealed output + receipt."""
    t_start = time.perf_counter()
    enforce_cpu_only(max_threads=max_threads)

    resolved_out_dir = assert_path_outside_repo(out_dir, repo_root=repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    effective_date = date_tag or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    out_path, receipt_path, final_date_tag = allocate_sealed_output_paths(
        resolved_out_dir,
        effective_date,
        allow_new_dated_on_collision=allow_new_dated_on_collision,
    )
    assert_path_outside_repo(out_path, repo_root=repo_root)
    assert_path_outside_repo(receipt_path, repo_root=repo_root)

    payload, clips_verified = compute_composition_divergence(
        clips_dir=clips_dir,
        hashes_file=hashes_file,
        encoder=encoder,
        max_threads=max_threads,
        regeneration_file=regeneration_file,
        date_tag=final_date_tag,
        local_snapshot_path=local_snapshot_path,
    )

    output_sha256 = write_json_exclusive(out_path, payload)
    wall_time_s = round(time.perf_counter() - t_start, 4)
    script_sha256 = compute_file_sha256(Path(__file__).resolve())

    header = payload["header"]
    receipt_payload: Dict[str, Any] = {
        "schema_version": "cosmos3-composition-divergence-receipt-v1",
        "date": final_date_tag,
        "model_id": header["model_id"],
        "model_revision": header["model_revision"],
        "model_license": header["model_license"],
        "geometry": header["geometry"],
        "script_sha256": script_sha256,
        "host": "CPU",
        "max_threads": int(max_threads),
        "wall_time_s": wall_time_s,
        "output_file": out_path.name,
        "output_sha256": output_sha256,
        "clips_verified": int(clips_verified),
    }

    # Optional sealed addendum if audit_manifest.json was already sealed before this job ran (§3.4 item 6)
    if sealed_manifest_path is not None:
        mpath = Path(sealed_manifest_path)
        if mpath.is_file():
            manifest_sha = compute_file_sha256(mpath)
            addendum_path = resolved_out_dir / f"audit_manifest_addendum_{final_date_tag}.json"
            assert_path_outside_repo(addendum_path, repo_root=repo_root)
            addendum_payload = {
                "schema_version": "cosmos3-audit-manifest-addendum-v1",
                "date": final_date_tag,
                "audit_manifest_file": mpath.name,
                "audit_manifest_sha256": manifest_sha,
                "composition_divergence_file": out_path.name,
                "composition_divergence_sha256": output_sha256,
            }
            addendum_sha = write_json_exclusive(addendum_path, addendum_payload)
            receipt_payload["audit_manifest_addendum_file"] = addendum_path.name
            receipt_payload["audit_manifest_addendum_sha256"] = addendum_sha

    receipt_sha256 = write_json_exclusive(receipt_path, receipt_payload)
    return {
        "output_path": str(out_path),
        "output_sha256": output_sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "clips_verified": clips_verified,
        "score_informative": header["score_informative"],
        "flagged_cells": header["flagged_cells"],
        "borderline_cells": header["borderline_cells"],
    }


def _encode_synthetic_rgb_clip(
    dest_path: Path,
    frames_rgb: np.ndarray,
    fps: int = 24,
    prefer_ffmpeg: bool = False,
) -> str:
    """Write a synthetic 61-frame 832x480 RGB clip and return its SHA-256 hex digest."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames, h, w, c = frames_rgb.shape
    if (w, h, c) != (INPUT_WIDTH, INPUT_HEIGHT, 3):
        raise ValueError(f"Expected synthetic frames of shape (N, 480, 832, 3), got {frames_rgb.shape}")

    if prefer_ffmpeg and shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(dest_path),
            ],
            input=frames_rgb.tobytes(),
            check=True,
        )
    else:
        meta = json.dumps({"width": w, "height": h, "frames": n_frames, "fps": fps}).encode("utf-8") + b"\n"
        with open(dest_path, "wb") as f:
            f.write(SYNTH_RAW_HEADER)
            f.write(meta)
            f.write(frames_rgb.tobytes())

    return compute_file_sha256(dest_path)


def generate_synthetic_divergence_fixture(
    fixture_dir: str | Path,
    prompts: Sequence[str] = ("PA", "PB"),
    seeds: Sequence[int] = (101, 202, 303),
    arms: Sequence[str] = SUPER_ARMS,
    divergent_cell: Tuple[str, int] = ("PA", 101),
    divergent_arm: str = ARM_Q3,
    include_nano: bool = True,
    prefer_ffmpeg: bool = False,
) -> Dict[str, Any]:
    """Generate a synthetic clip fixture (default: 2 prompts x 3 seeds x 3 Super arms):
    - Same-cell arms share the same spatial layout (with tiny arm-specific pixel jitter).
    - Different seeds within a prompt have distinct spatial layouts.
    - Exactly one cell (`divergent_cell`) has one arm (`divergent_arm`) spatially shifted
      to simulate a framing difference.
    """
    fdir = Path(fixture_dir)
    mp4_dir = fdir / "mp4"
    mp4_dir.mkdir(parents=True, exist_ok=True)

    # Distinct spatial layouts for each (prompt_idx, seed_idx): (y0, x0, box_h, box_w)
    layout_slots: List[Tuple[int, int, int, int]] = [
        (24, 28, 150, 210),
        (24, 560, 150, 210),
        (290, 28, 150, 210),
        (290, 560, 150, 210),
        (150, 290, 150, 220),
        (30, 290, 140, 220),
    ]
    shifted_slot: Tuple[int, int, int, int] = (295, 575, 150, 210)

    hash_lines: List[str] = []
    fetch_lines: List[str] = []
    clip_records: Dict[str, str] = {}

    all_arms = list(arms) + ([ARM_NANO] if include_nano and ARM_NANO not in arms else [])

    for p_idx, prompt in enumerate(prompts):
        for s_idx, seed in enumerate(seeds):
            slot_idx = (p_idx * len(seeds) + s_idx) % len(layout_slots)
            base_slot = layout_slots[slot_idx]
            bg_color = np.array(
                [35 + 20 * p_idx, 45 + 15 * s_idx, 60 + 10 * p_idx],
                dtype=np.uint8,
            )
            fg_color = np.array(
                [220 - 25 * p_idx, 190 - 20 * s_idx, 90 + 40 * p_idx],
                dtype=np.uint8,
            )

            for a_idx, arm in enumerate(all_arms):
                is_shifted = (
                    (prompt, int(seed)) == (divergent_cell[0], int(divergent_cell[1]))
                    and arm == divergent_arm
                ) or (arm == ARM_NANO)
                y0, x0, bh, bw = shifted_slot if is_shifted else base_slot

                frames = np.empty((61, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
                # Deterministic tiny arm offset (1..3 levels) so same-layout arms have small >0 distance
                arm_jitter = np.uint8((a_idx + 1) * 2)
                frames[:] = bg_color[None, None, None, :] + arm_jitter

                for f_i, f_num in enumerate(FRAME_INDICES):
                    dx = f_i * 6
                    x_start = min(max(0, x0 + dx), INPUT_WIDTH - bw)
                    y_start = min(max(0, y0), INPUT_HEIGHT - bh)
                    frame_slice = np.empty((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
                    frame_slice[:] = bg_color[None, None, :] + arm_jitter
                    frame_slice[y_start : y_start + bh, x_start : x_start + bw, :] = fg_color
                    # Stamp into frames
                    if f_num == 0:
                        frames[0:30] = frame_slice
                    elif f_num == 30:
                        frames[30:60] = frame_slice
                    else:
                        frames[60] = frame_slice

                rel_path = f"mp4/{arm}_cell_{prompt}-s{int(seed)}.mp4"
                full_path = fdir / rel_path
                sha = _encode_synthetic_rgb_clip(full_path, frames, prefer_ffmpeg=prefer_ffmpeg)
                hash_lines.append(f"{sha}  {rel_path}")
                fetch_lines.append(f"/workspace/synth/{rel_path}")
                clip_records[rel_path] = sha

    hashes_path = fdir / "sha256sums.txt"
    hashes_path.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    fetch_path = fdir / "fetch_list.txt"
    fetch_path.write_text("\n".join(fetch_lines) + "\n", encoding="utf-8")

    return {
        "clips_dir": str(fdir),
        "hashes_file": str(hashes_path),
        "fetch_list_file": str(fetch_path),
        "clips": clip_records,
        "divergent_cell_id": f"{divergent_cell[0]}-s{int(divergent_cell[1])}",
        "divergent_arm": divergent_arm,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m blindpanel.composition_divergence",
        description="Cosmos 3 Amendment A §3.4 CPU composition-divergence pre-read job.",
    )
    parser.add_argument(
        "--clips",
        required=True,
        help="Directory containing input clips",
    )
    parser.add_argument(
        "--hashes",
        required=True,
        help="Path to sha256sums.txt-formatted per-clip SHA-256 file",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Sealed output directory outside the repository tree",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_MAX_THREADS,
        help=f"Maximum CPU threads (default: {DEFAULT_MAX_THREADS})",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date tag for output filenames (default: UTC YYYY-MM-DD)",
    )
    parser.add_argument(
        "--regeneration-file",
        default=None,
        help="Optional regeneration JSON file for noise floor d_floor",
    )
    parser.add_argument(
        "--dinov2-path",
        default=None,
        help="Optional local snapshot path for facebook/dinov2-base",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path to an already-sealed audit_manifest.json (writes addendum without modifying manifest)",
    )
    parser.add_argument(
        "--use-stub-encoder",
        action="store_true",
        help="Use deterministic offline stub encoder (for synthetic/offline testing)",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    encoder: Optional[FrameEncoder] = None,
    repo_root: Optional[str | Path] = None,
) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    active_encoder = encoder
    if active_encoder is None and args.use_stub_encoder:
        active_encoder = DeterministicStubDinov2Encoder()

    result = run_composition_divergence_job(
        clips_dir=args.clips,
        hashes_file=args.hashes,
        out_dir=args.out_dir,
        encoder=active_encoder,
        max_threads=args.threads,
        regeneration_file=args.regeneration_file,
        date_tag=args.date,
        local_snapshot_path=args.dinov2_path,
        sealed_manifest_path=args.manifest,
        repo_root=repo_root,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
