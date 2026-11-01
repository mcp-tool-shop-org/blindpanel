"""test_composition_divergence.py — Unit, synthetic fixture, CLI, and leak-safety tests
for Cosmos 3 Amendment A §3.4 composition-divergence pre-read job."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from blindpanel.composition_divergence import (
    ARM_BF16,
    ARM_NANO,
    ARM_Q1,
    ARM_Q3,
    DEFAULT_MAX_THREADS,
    DINOV2_LICENSE,
    DINOV2_MODEL_ID,
    DINOV2_REVISION,
    FRAME_INDICES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_PATCHES,
    PATCH_GRID,
    PATCH_SIZE,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    SUPER_ARMS,
    ClipHashMismatchError,
    CudaDeviceForbiddenError,
    DeterministicStubDinov2Encoder,
    Dinov2Encoder,
    InvalidRevisionPinError,
    OutputOverwriteError,
    RepoOutputPathError,
    allocate_sealed_output_paths,
    assert_path_outside_repo,
    check_informativeness,
    compute_cls_distance,
    compute_file_sha256,
    compute_patch_distance,
    compute_robust_null_stats,
    decode_clip_frames,
    enforce_cpu_only,
    evaluate_cell_flag_and_borderline,
    generate_synthetic_divergence_fixture,
    load_noise_floor,
    parse_clip_name,
    parse_fetch_list,
    parse_sha256sums,
    preprocess_frame_rgb,
    run_composition_divergence_job,
    type7_percentile,
    type7_percentile_rank,
    validate_divergence_output_schema,
    validate_revision_pin,
    write_json_exclusive,
)
from blindpanel.composition_divergence import (
    main as cli_main,
)
from blindpanel.leak_guard import scan_repository


@pytest.fixture(scope="module")
def synthetic_fixture_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Generate the 2 prompts x 3 seeds x 3 arms (+ Nano) synthetic clip fixture once per test module."""
    base_dir = tmp_path_factory.mktemp("synth_cosmos3_clips")
    return generate_synthetic_divergence_fixture(
        fixture_dir=base_dir,
        prompts=("PA", "PB"),
        seeds=(101, 202, 303),
        arms=SUPER_ARMS,
        divergent_cell=("PA", 101),
        divergent_arm=ARM_Q3,
        include_nano=True,
        prefer_ffmpeg=False,
    )


def test_clip_name_parser_and_hash_list_formats() -> None:
    """Requirement 1: Verify clip-name parser maps B200 fetch_list.txt and sha256sums.txt patterns
    to (arm, prompt, seed) using made-up names in the same pattern."""
    # Pattern 1: fetch_list.txt / sha256sums.txt mp4/<ARM>_cell_<PROMPT>-s<SEED>.mp4
    p1 = parse_clip_name(f"/root/sq_b200/mp4/{ARM_BF16}_cell_PA-s101.mp4")
    assert p1 is not None
    assert (p1.arm, p1.prompt, p1.seed, p1.cell_id) == (ARM_BF16, "PA", 101, "PA-s101")

    # Pattern 2: sha256sums.txt per-arm directory <ARM>/cell_<PROMPT>-s<SEED>.mp4
    p2 = parse_clip_name(f"{ARM_Q3}/cell_PB-s303.mp4")
    assert p2 is not None
    assert (p2.arm, p2.prompt, p2.seed, p2.cell_id) == (ARM_Q3, "PB", 303, "PB-s303")

    # Pattern 3: flat <ARM>_<PROMPT>-s<SEED>.mp4
    p3 = parse_clip_name(f"{ARM_Q1}_PA-s202.mp4")
    assert p3 is not None
    assert (p3.arm, p3.prompt, p3.seed, p3.cell_id) == (ARM_Q1, "PA", 202, "PA-s202")

    # Non-clip entries return None
    assert parse_clip_name("") is None
    assert parse_clip_name("/root/sq_b200/results.jsonl") is None
    assert parse_clip_name("/root/sq_b200/b200_same_card_out.tgz") is None
    assert parse_clip_name("cell_PA-s101.mp4") is None
    assert parse_clip_name("unmatched_file.mp4") is None

    # Parse fetch_list.txt with mixed metadata and clip lines
    fetch_txt = "\n".join(
        [
            "/root/sq_b200/b200_same_card_out.tgz",
            "/root/sq_b200/sha256sums.txt",
            "/root/sq_b200/results.jsonl",
            f"/root/sq_b200/mp4/{ARM_BF16}_cell_PA-s101.mp4",
            f"/root/sq_b200/mp4/{ARM_Q1}_cell_PA-s101.mp4",
            f"/root/sq_b200/mp4/{ARM_Q3}_cell_PA-s101.mp4",
        ]
    )
    parsed_fetch = parse_fetch_list(fetch_txt)
    assert len(parsed_fetch) == 3
    assert [x.arm for x in parsed_fetch] == [ARM_BF16, ARM_Q1, ARM_Q3]

    # Parse sha256sums.txt (both two-space and binary '*' prefix)
    h1 = "a" * 64
    h2 = "b" * 64
    sums_txt = f"{h1}  mp4/{ARM_BF16}_cell_PA-s101.mp4\n{h2} *{ARM_Q1}/cell_PA-s101.mp4\n"
    parsed_sums = parse_sha256sums(sums_txt)
    assert parsed_sums[f"mp4/{ARM_BF16}_cell_PA-s101.mp4"] == h1
    assert parsed_sums[f"{ARM_Q1}/cell_PA-s101.mp4"] == h2


def test_cpu_only_and_thread_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 2: Default to 8 threads, configurable, and assert no CUDA device is used."""
    recorded_threads: List[int] = []

    class FakeCuda:
        def __init__(self, avail: bool = False, init: bool = False) -> None:
            self._avail = avail
            self._init = init

        def is_available(self) -> bool:
            return self._avail

        def is_initialized(self) -> bool:
            return self._init

    class FakeTorch:
        def __init__(self, avail: bool = False, init: bool = False) -> None:
            self.cuda = FakeCuda(avail=avail, init=init)

        def set_num_threads(self, n: int) -> None:
            recorded_threads.append(n)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    used = enforce_cpu_only(torch_module=FakeTorch())
    assert used == DEFAULT_MAX_THREADS == 8
    assert recorded_threads[-1] == 8
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"

    used_custom = enforce_cpu_only(max_threads=4, torch_module=FakeTorch())
    assert used_custom == 4
    assert recorded_threads[-1] == 4
    assert os.environ["OMP_NUM_THREADS"] == "4"

    with pytest.raises(ValueError, match="max_threads must be >= 1"):
        enforce_cpu_only(max_threads=0, torch_module=FakeTorch())

    # Refuse non-"-1" CUDA_VISIBLE_DEVICES
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(CudaDeviceForbiddenError) as exc_info:
        enforce_cpu_only(torch_module=FakeTorch())
    assert exc_info.value.failure_class == "CUDA_DEVICE_FORBIDDEN"

    # Refuse when torch.cuda.is_available() is True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(CudaDeviceForbiddenError, match="CUDA device is available"):
        enforce_cpu_only(torch_module=FakeTorch(avail=True))

    # Refuse when torch.cuda.is_initialized() is True
    with pytest.raises(CudaDeviceForbiddenError, match="CUDA runtime is initialized"):
        enforce_cpu_only(torch_module=FakeTorch(avail=False, init=True))


def test_revision_pin_validation() -> None:
    """Spec: Revision must be pinned to an exact 40-hex commit hash, never 'main'."""
    assert validate_revision_pin(DINOV2_REVISION) == DINOV2_REVISION
    assert DINOV2_MODEL_ID == "facebook/dinov2-base"
    assert DINOV2_LICENSE == "Apache-2.0"

    for bad_rev in ("main", "MAIN", "master", "latest", "", "abc123", "g" * 40):
        with pytest.raises(InvalidRevisionPinError) as exc_info:
            validate_revision_pin(bad_rev)
        assert exc_info.value.failure_class == "INVALID_REVISION_PIN"


def test_patch_and_cls_distance_formulas_on_known_vectors() -> None:
    """Requirement 3: Verify patch (448 patches) and CLS cosine distance formulas on known vectors,
    with float32 tokens and float64 distances."""
    # 1. Identical tokens -> distance 0.0
    patches_same = np.ones((NUM_PATCHES, 16), dtype=np.float32)
    cls_same = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d_p_same = compute_patch_distance(patches_same, patches_same)
    d_c_same = compute_cls_distance(cls_same, cls_same)
    assert isinstance(d_p_same, float)
    assert isinstance(d_c_same, float)
    assert d_p_same == pytest.approx(0.0, abs=1e-15)
    assert d_c_same == pytest.approx(0.0, abs=1e-15)

    # 2. Orthogonal CLS vectors -> cosine = 0 -> distance = 1.0
    cls_a = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    cls_b = np.array([0.0, 5.0, 0.0], dtype=np.float32)
    assert compute_cls_distance(cls_a, cls_b) == pytest.approx(1.0, abs=1e-15)

    # 3. Opposite CLS vectors -> cosine = -1 -> distance = 2.0
    assert compute_cls_distance(cls_a, -cls_a) == pytest.approx(2.0, abs=1e-15)

    # 4. Mixed 448 patches:
    #    - 112 patches identical (d = 0.0)
    #    - 224 patches orthogonal (d = 1.0)
    #    - 112 patches opposite (d = 2.0)
    #    Expected mean over 448 patches = (112*0 + 224*1 + 112*2) / 448 = 448 / 448 = 1.0
    pa = np.zeros((NUM_PATCHES, 4), dtype=np.float32)
    pb = np.zeros((NUM_PATCHES, 4), dtype=np.float32)
    pa[:, 0] = 2.0
    pb[:112, 0] = 5.0  # identical direction
    pb[112:336, 1] = 3.0  # orthogonal
    pb[336:, 0] = -4.0  # opposite
    assert compute_patch_distance(pa, pb) == pytest.approx(1.0, abs=1e-15)

    # 5. Shape validation and zero-norm refusal
    with pytest.raises(ValueError, match="Patch token arrays must both have shape"):
        compute_patch_distance(np.ones((10, 4), dtype=np.float32), np.ones((10, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Zero-norm patch token"):
        compute_patch_distance(np.zeros((NUM_PATCHES, 4), dtype=np.float32), pa)
    with pytest.raises(ValueError, match="Zero-norm CLS token"):
        compute_cls_distance(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))


def test_type7_percentile_and_rank_conventions() -> None:
    """Requirement 3: Verify Hyndman & Fan type-7 (`numpy.percentile(..., method='linear')`)
    percentile and rank calculation."""
    # Spec §3.4 item 2: for n = 9 (sorted 1..9), 5th percentile is at 1-based rank 1 + 0.05 * 8 = 1.4
    b_prompt = [float(k) for k in range(1, 10)]
    tau_p = type7_percentile(b_prompt, 5.0)
    assert tau_p == pytest.approx(1.4, abs=1e-12)
    assert type7_percentile_rank(b_prompt, 1.4) == pytest.approx(5.0, abs=1e-12)

    # Spec §3.4 item 2: for n = 108 (sorted 1..108), 5th percentile is at 1-based rank 1 + 0.05 * 107 = 6.35
    b_pooled = [float(k) for k in range(1, 109)]
    tau_pooled = type7_percentile(b_pooled, 5.0)
    assert tau_pooled == pytest.approx(6.35, abs=1e-12)
    assert type7_percentile_rank(b_pooled, 6.35) == pytest.approx(5.0, abs=1e-12)

    # Boundary and tie checks
    assert type7_percentile_rank(b_prompt, 0.5) == 0.0
    assert type7_percentile_rank(b_prompt, 9.5) == 100.0
    assert type7_percentile_rank([5.0], 5.0) == 50.0
    assert type7_percentile_rank([5.0], 2.0) == 0.0
    assert type7_percentile_rank([5.0], 8.0) == 100.0
    assert type7_percentile_rank([1.0, 2.0, 2.0, 4.0], 2.0) == pytest.approx(50.0, abs=1e-12)


def test_mad_scaling_and_robust_null_cutoff() -> None:
    """Requirement 3: Verify m_W = median(W), MAD_W = 1.4826 * median(|d - m_W|), and nu = m_W + 3 * MAD_W."""
    w = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.50]
    m_w, mad_w, nu = compute_robust_null_stats(w)
    # Median of 9 sorted values is index 4 -> 0.05
    # |d - 0.05| = [0.04, 0.03, 0.02, 0.01, 0.00, 0.01, 0.02, 0.03, 0.45]
    # Sorted abs deviations = [0.00, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.45] -> median = 0.02
    expected_m_w = 0.05
    expected_mad_w = 1.4826 * 0.02
    expected_nu = expected_m_w + 3.0 * expected_mad_w
    assert m_w == pytest.approx(expected_m_w, abs=1e-12)
    assert mad_w == pytest.approx(expected_mad_w, abs=1e-12)
    assert nu == pytest.approx(expected_nu, abs=1e-12)


def test_informativeness_rule_both_outcomes() -> None:
    """Requirement 3: Verify `score_informative` is True iff `nu < tau_pooled` AND `tau_pooled >= 2 * m_W`,
    and when False, `composition_flag` is 0 even if D > tau_pooled and D > nu."""
    # Outcome 1: Both hold -> True
    assert check_informativeness(nu=0.10, tau_pooled=0.25, m_w=0.04) is True

    # Outcome 2a: nu >= tau_pooled -> False
    assert check_informativeness(nu=0.25, tau_pooled=0.25, m_w=0.04) is False
    assert check_informativeness(nu=0.30, tau_pooled=0.25, m_w=0.04) is False

    # Outcome 2b: tau_pooled < 2 * m_W (even though nu < tau_pooled when MAD=0) -> False
    assert check_informativeness(nu=0.05, tau_pooled=0.09, m_w=0.05) is False

    # Verify flag is suppressed when score_informative is False
    flag, borderline, fired = evaluate_cell_flag_and_borderline(
        d_patch=0.80,
        d_cls=0.05,
        score_informative=False,
        tau_pooled=0.25,
        tau_p=0.20,
        nu=0.10,
        tau_pooled_cls=0.30,
        nu_cls=0.12,
    )
    assert flag == 0
    assert borderline == 1
    assert fired == ["a", "b"]


def test_flag_and_borderline_each_branch() -> None:
    """Requirement 3: Verify `composition_flag` and `borderline` branches (a), (b), (c), and clean 0."""
    # 1. Flagged cell: informative and D_patch > tau_pooled and D_patch > nu
    flag, borderline, fired = evaluate_cell_flag_and_borderline(
        d_patch=0.40,
        d_cls=0.50,
        score_informative=True,
        tau_pooled=0.30,
        tau_p=0.25,
        nu=0.15,
        tau_pooled_cls=0.20,
        nu_cls=0.10,
    )
    assert (flag, borderline, fired) == (1, 0, [])

    # 2. Borderline branch (a) only: D_patch > tau_p, but D_patch <= nu and D_patch <= tau_pooled
    flag, borderline, fired = evaluate_cell_flag_and_borderline(
        d_patch=0.12,
        d_cls=0.05,
        score_informative=True,
        tau_pooled=0.30,
        tau_p=0.10,
        nu=0.15,
        tau_pooled_cls=0.20,
        nu_cls=0.10,
    )
    assert (flag, borderline, fired) == (0, 1, ["a"])

    # 3. Borderline branch (b) only: D_patch > nu, but D_patch <= tau_p and D_patch <= tau_pooled
    flag, borderline, fired = evaluate_cell_flag_and_borderline(
        d_patch=0.20,
        d_cls=0.05,
        score_informative=True,
        tau_pooled=0.30,
        tau_p=0.25,
        nu=0.15,
        tau_pooled_cls=0.20,
        nu_cls=0.10,
    )
    assert (flag, borderline, fired) == (0, 1, ["b"])

    # 4. Borderline branch (c) only: CLS exceeds both tau_pooled_cls and nu_cls while D_patch is low
    flag, borderline, fired = evaluate_cell_flag_and_borderline(
        d_patch=0.05,
        d_cls=0.25,
        score_informative=True,
        tau_pooled=0.30,
        tau_p=0.25,
        nu=0.15,
        tau_pooled_cls=0.20,
        nu_cls=0.10,
    )
    assert (flag, borderline, fired) == (0, 1, ["c"])

    # 5. Borderline branches (a), (b), and (c) together (when D_patch <= tau_pooled so composition_flag == 0)
    flag, borderline, fired = evaluate_cell_flag_and_borderline(
        d_patch=0.28,
        d_cls=0.25,
        score_informative=True,
        tau_pooled=0.30,
        tau_p=0.20,
        nu=0.15,
        tau_pooled_cls=0.20,
        nu_cls=0.10,
    )
    assert (flag, borderline, fired) == (0, 1, ["a", "b", "c"])

    # 6. Clean unflagged, non-borderline cell
    flag, borderline, fired = evaluate_cell_flag_and_borderline(
        d_patch=0.04,
        d_cls=0.15,  # exceeds nu_cls=0.10 but NOT tau_pooled_cls=0.20 -> (c) does not fire
        score_informative=True,
        tau_pooled=0.30,
        tau_p=0.20,
        nu=0.15,
        tau_pooled_cls=0.20,
        nu_cls=0.10,
    )
    assert (flag, borderline, fired) == (0, 0, [])


def test_frame_preprocessing_geometry() -> None:
    """Spec item 1: 832x480 RGB resized directly to 392x224 (bicubic, no crop, no pad) + ImageNet mean/std."""
    frame = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    chw = preprocess_frame_rgb(frame)
    assert chw.dtype == np.float32
    assert chw.shape == (3, RESIZE_HEIGHT, RESIZE_WIDTH) == (3, 224, 392)
    assert PATCH_GRID == (28, 16)
    assert PATCH_SIZE == 14
    assert NUM_PATCHES == 448

    # For all-zero uint8 input, normalized channel c equals (0 - mean[c]) / std[c]
    for c_idx in range(3):
        expected_val = -IMAGENET_MEAN[c_idx] / IMAGENET_STD[c_idx]
        assert np.allclose(chw[c_idx], expected_val, atol=1e-5)

    with pytest.raises(ValueError, match="Input frame must be 832x480"):
        preprocess_frame_rgb(np.zeros((240, 320, 3), dtype=np.uint8))


def test_hash_mismatch_refusal_before_decoding(
    synthetic_fixture_env: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 3: Verify job verifies every clip's SHA-256 before decoding and refuses on mismatch
    with named failure class `ClipHashMismatchError` (`CLIP_HASH_MISMATCH`)."""
    bad_hashes = tmp_path / "tampered_sha256sums.txt"
    orig_text = Path(synthetic_fixture_env["hashes_file"]).read_text(encoding="utf-8")
    lines = orig_text.strip().splitlines()
    # Tamper the first Super clip's hash
    lines[0] = ("f" * 64) + "  " + lines[0].split(None, 1)[1]
    bad_hashes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    decode_called = False

    def _forbid_decode(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal decode_called
        decode_called = True
        raise AssertionError("decode_clip_frames must not be called when a clip hash mismatches!")

    monkeypatch.setattr("blindpanel.composition_divergence.decode_clip_frames", _forbid_decode)

    out_dir = tmp_path / "sealed_out"
    with pytest.raises(ClipHashMismatchError) as exc_info:
        run_composition_divergence_job(
            clips_dir=synthetic_fixture_env["clips_dir"],
            hashes_file=bad_hashes,
            out_dir=out_dir,
            encoder=DeterministicStubDinov2Encoder(),
            date_tag="2099-01-01",
        )
    assert exc_info.value.failure_class == "CLIP_HASH_MISMATCH"
    assert decode_called is False


def test_refusal_of_output_path_inside_repo_and_cli_writes_nothing_to_repo(
    synthetic_fixture_env: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Requirements 3 & 4: Refuse any output path inside the repository tree, and verify CLI
    writes zero files into the repo tree both when refused and when writing to a sealed dir."""
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Direct helper check on repo root and nested path
    with pytest.raises(RepoOutputPathError) as exc_info:
        assert_path_outside_repo(repo_root / "sealed_inside_repo", repo_root=repo_root)
    assert exc_info.value.failure_class == "REPO_OUTPUT_PATH_REFUSED"

    # Snapshot all files in repo tree before running CLI
    def _snapshot_repo_files() -> set[str]:
        found: set[str] = set()
        for r, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", ".ruff_cache")]
            for fn in files:
                found.add(Path(r, fn).relative_to(repo_root).as_posix())
        return found

    before_files = _snapshot_repo_files()

    # 2. CLI call targeting a path inside the repo must fail with RepoOutputPathError and create nothing
    inside_target = repo_root / "forbidden_sealed_out"
    with pytest.raises(RepoOutputPathError):
        cli_main(
            [
                "--clips",
                synthetic_fixture_env["clips_dir"],
                "--hashes",
                synthetic_fixture_env["hashes_file"],
                "--out-dir",
                str(inside_target),
                "--date",
                "2099-01-01",
                "--use-stub-encoder",
            ],
            repo_root=repo_root,
        )
    assert not inside_target.exists()
    assert _snapshot_repo_files() == before_files

    # 3. Subprocess invocation of `python -m blindpanel.composition_divergence` to external sealed dir
    external_sealed = tmp_path / "external_sealed_store"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "blindpanel.composition_divergence",
            "--clips",
            synthetic_fixture_env["clips_dir"],
            "--hashes",
            synthetic_fixture_env["hashes_file"],
            "--out-dir",
            str(external_sealed),
            "--date",
            "2099-01-01",
            "--threads",
            "2",
            "--use-stub-encoder",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    cli_out = json.loads(proc.stdout)
    assert Path(cli_out["output_path"]).is_file()
    assert Path(cli_out["receipt_path"]).is_file()
    assert _snapshot_repo_files() == before_files


def test_no_overwrite_of_existing_output_and_rerun_creates_new_dated_file(
    synthetic_fixture_env: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Requirement 3: Verify existing output files are never overwritten; direct overwrite raises
    `OutputOverwriteError` and a job re-run creates a new dated file while leaving the first intact."""
    sealed_dir = tmp_path / "sealed_no_overwrite"
    stub = DeterministicStubDinov2Encoder()

    res1 = run_composition_divergence_job(
        clips_dir=synthetic_fixture_env["clips_dir"],
        hashes_file=synthetic_fixture_env["hashes_file"],
        out_dir=sealed_dir,
        encoder=stub,
        date_tag="2099-01-01",
    )
    out1 = Path(res1["output_path"])
    rec1 = Path(res1["receipt_path"])
    bytes1 = out1.read_bytes()
    sha1 = compute_file_sha256(out1)
    assert sha1 == res1["output_sha256"]

    # Direct exclusive write or strict collision mode must raise OutputOverwriteError
    with pytest.raises(OutputOverwriteError) as exc_info:
        write_json_exclusive(out1, {"tampered": True})
    assert exc_info.value.failure_class == "OUTPUT_OVERWRITE_REFUSED"

    with pytest.raises(OutputOverwriteError):
        allocate_sealed_output_paths(sealed_dir, "2099-01-01", allow_new_dated_on_collision=False)

    # Re-running the job on the same date preserves out1 untouched and writes a new dated file
    res2 = run_composition_divergence_job(
        clips_dir=synthetic_fixture_env["clips_dir"],
        hashes_file=synthetic_fixture_env["hashes_file"],
        out_dir=sealed_dir,
        encoder=stub,
        date_tag="2099-01-01",
    )
    out2 = Path(res2["output_path"])
    rec2 = Path(res2["receipt_path"])
    assert out2 != out1
    assert rec2 != rec1
    assert out2.is_file()
    assert rec2.is_file()
    assert out1.read_bytes() == bytes1
    assert compute_file_sha256(out1) == sha1


def test_synthetic_fixture_end_to_end_flags_only_shifted_cell_and_validates_schema(
    synthetic_fixture_env: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Requirement 3: Run on the 2 prompts x 3 seeds x 3 arms synthetic fixture with stub encoder.
    Verify:
    - Nano clips in the directory/hashes are ignored (only 18 Super clips = 6 cells x 3 arms scored);
    - B has 2 prompts x 9 = 18 between-seed distances; W has 6 cells x 3 = 18 within-cell distances;
    - score_informative is True;
    - Only the shifted cell (`PA-s101`) is flagged (`composition_flag == 1`), all other 5 cells have `composition_flag == 0`;
    - Output schema contains no embeddings, frames, thumbnails, or arrays beyond distances;
    - Receipt contains model id, revision, license, geometry, script_sha256, host='CPU', wall_time_s, output_sha256, clips_verified=18.
    """
    sealed_dir = tmp_path / "sealed_e2e_stub"
    manifest_file = tmp_path / "audit_manifest.json"
    manifest_bytes = b'{"manifest_version": 1}\n'
    manifest_file.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    res = run_composition_divergence_job(
        clips_dir=synthetic_fixture_env["clips_dir"],
        hashes_file=synthetic_fixture_env["hashes_file"],
        out_dir=sealed_dir,
        encoder=DeterministicStubDinov2Encoder(),
        max_threads=4,
        date_tag="2099-01-01",
        sealed_manifest_path=manifest_file,
    )

    assert res["clips_verified"] == 18
    assert res["score_informative"] is True
    assert res["flagged_cells"] == [synthetic_fixture_env["divergent_cell_id"]]

    # Verify original manifest was not modified and addendum was written alongside output
    assert manifest_file.read_bytes() == manifest_bytes
    addendum_path = sealed_dir / "audit_manifest_addendum_2099-01-01.json"
    assert addendum_path.is_file()
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    assert addendum["audit_manifest_sha256"] == manifest_sha
    assert addendum["composition_divergence_sha256"] == res["output_sha256"]

    # Inspect output payload and receipt
    output_data = json.loads(Path(res["output_path"]).read_text(encoding="utf-8"))
    receipt_data = json.loads(Path(res["receipt_path"]).read_text(encoding="utf-8"))

    validate_divergence_output_schema(output_data)

    header = output_data["header"]
    assert header["model_id"] == DINOV2_MODEL_ID
    assert header["model_revision"] == DINOV2_REVISION
    assert header["model_license"] == DINOV2_LICENSE
    assert header["d_floor"] == "not measured"
    assert header["num_cells"] == 6
    assert header["num_between_seed_B"] == 18
    assert header["num_within_cell_W"] == 18
    assert set(header["tau_p"].keys()) == {"PA", "PB"}

    cells_by_id = {c["cell_id"]: c for c in output_data["cells"]}
    assert len(cells_by_id) == 6
    shifted_cell = cells_by_id["PA-s101"]
    assert shifted_cell["composition_flag"] == 1
    assert shifted_cell["borderline"] == 0
    assert shifted_cell["D_patch"] > header["tau_pooled"]
    assert shifted_cell["D_patch"] > header["nu"]

    # In PA-s101, Q3 was shifted while BF16 and Q1 share layout -> BF16_Q3 and Q1_Q3 >> BF16_Q1
    p_dists = shifted_cell["pair_distances"]
    assert p_dists[f"{ARM_BF16}_{ARM_Q3}"]["patch_frame_0"] > p_dists[f"{ARM_BF16}_{ARM_Q1}"]["patch_frame_0"] * 5.0
    assert p_dists[f"{ARM_Q1}_{ARM_Q3}"]["patch_frame_0"] > p_dists[f"{ARM_BF16}_{ARM_Q1}"]["patch_frame_0"] * 5.0

    for cid, cell in cells_by_id.items():
        assert set(cell["clip_hashes"].keys()) == set(SUPER_ARMS)
        assert ARM_NANO not in cell["clip_hashes"]
        assert 0.0 <= cell["rank_in_B_pct"] <= 100.0
        assert 0.0 <= cell["rank_in_W_pct"] <= 100.0
        if cid != "PA-s101":
            assert cell["composition_flag"] == 0

    # Check receipt fields
    assert receipt_data["model_id"] == DINOV2_MODEL_ID
    assert receipt_data["model_revision"] == DINOV2_REVISION
    assert receipt_data["model_license"] == DINOV2_LICENSE
    assert receipt_data["host"] == "CPU"
    assert receipt_data["max_threads"] == 4
    assert receipt_data["clips_verified"] == 18
    assert receipt_data["output_sha256"] == res["output_sha256"]
    assert len(receipt_data["script_sha256"]) == 64

    # Verify schema validator rejects forbidden keys or cell arrays
    for bad_key in ("embeddings", "frames", "thumbnails", "patch_tokens", "cls_token", "clip_paths"):
        bad_payload = json.loads(json.dumps(output_data))
        bad_payload["cells"][0][bad_key] = [1.0, 2.0]
        with pytest.raises(ValueError, match="Forbidden key"):
            validate_divergence_output_schema(bad_payload)

    bad_arr_payload = json.loads(json.dumps(output_data))
    bad_arr_payload["cells"][0]["extra_array"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="Unexpected array in cell record"):
        validate_divergence_output_schema(bad_arr_payload)


def test_regeneration_noise_floor_and_dinov2_encoder_unit_offline(tmp_path: Path) -> None:
    """Test `load_noise_floor` with regeneration JSON and `Dinov2Encoder` offline with injected mock modules."""
    regen_file = tmp_path / "regen.json"
    regen_file.write_text(json.dumps({"distances": [0.002, 0.004]}), encoding="utf-8")
    val, below = load_noise_floor(regen_file, nu=0.05)
    assert val == pytest.approx(0.004)
    assert below is True

    regen_direct = tmp_path / "regen_direct.json"
    regen_direct.write_text(json.dumps({"d_floor": 0.08}), encoding="utf-8")
    val2, below2 = load_noise_floor(regen_direct, nu=0.05)
    assert val2 == pytest.approx(0.08)
    assert below2 is False

    # Test Dinov2Encoder with offline mock torch & transformers
    class _FakeTensor:
        def __init__(self, arr: np.ndarray, device_type: str = "cpu") -> None:
            self._arr = arr
            self.device = SimpleNamespace(type=device_type)

        def unsqueeze(self, axis: int) -> "_FakeTensor":
            return _FakeTensor(np.expand_dims(self._arr, axis=axis), self.device.type)

        def detach(self) -> "_FakeTensor":
            return self

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

    class _FakeTorch:
        cuda = SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False)

        @staticmethod
        def set_num_threads(_n: int) -> None:
            pass

        @staticmethod
        def from_numpy(arr: np.ndarray) -> _FakeTensor:
            return _FakeTensor(arr, "cpu")

        @staticmethod
        def no_grad() -> _FakeNoGrad:
            return _FakeNoGrad()

    class _FakeModel:
        def __init__(self, param_dev: str = "cpu", out_dev: str = "cpu") -> None:
            self._param_dev = param_dev
            self._out_dev = out_dev

        def to(self, _dev: str) -> "_FakeModel":
            return self

        def eval(self) -> "_FakeModel":
            return self

        def parameters(self) -> List[Any]:
            return [SimpleNamespace(device=SimpleNamespace(type=self._param_dev))]

        def __call__(self, pixel_values: _FakeTensor) -> Any:
            out = np.ones((1, 1 + NUM_PATCHES, 16), dtype=np.float32)
            out[0, 0, :] = 2.0
            return SimpleNamespace(last_hidden_state=_FakeTensor(out, self._out_dev))

    fake_tf = SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=lambda *_a, **_kw: _FakeModel())
    )
    enc = Dinov2Encoder(
        torch_module=_FakeTorch,
        transformers_module=fake_tf,
        local_files_only=True,
    )
    chw = np.zeros((3, RESIZE_HEIGHT, RESIZE_WIDTH), dtype=np.float32)
    p_tok, c_tok = enc.encode_preprocessed(chw)
    assert p_tok.shape == (NUM_PATCHES, 16)
    assert c_tok.shape == (16,)
    assert p_tok.dtype == np.float32
    assert c_tok.dtype == np.float32

    # Verify parameter on CUDA device is refused
    fake_tf_cuda = SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=lambda *_a, **_kw: _FakeModel(param_dev="cuda"))
    )
    with pytest.raises(CudaDeviceForbiddenError):
        Dinov2Encoder(torch_module=_FakeTorch, transformers_module=fake_tf_cuda)


def test_leak_guard_catches_real_clip_names_seeds_and_sealed_files(tmp_path: Path) -> None:
    """Requirement 4: Verify extended `leak_guard` catches real clip names (both `<ARM>_cell_` and `<ARM>/cell_`),
    real study seeds/cells in composition-divergence files, and tracked `sha256sums.txt` or `composition_divergence_*.json`."""
    scratch = tmp_path / "scratch_leak_check"
    scratch.mkdir(parents=True, exist_ok=True)

    # Dynamically construct forbidden tokens so this test file stays 100% clean
    s_seed = "11" + "11"
    s_cell = "M" + "1-s" + s_seed
    dir_clip = ARM_BF16 + "/cell_" + s_cell + ".mp4"

    (scratch / "bad_composition_divergence_fixture.py").write_text(
        f"SEED = {s_seed}\nCELL = '{s_cell}'\nCLIP = '{dir_clip}'\n",
        encoding="utf-8",
    )
    (scratch / ("sha256" + "sums.txt")).write_text("0" * 64 + "  test.mp4\n", encoding="utf-8")
    (scratch / "composition_divergence_2099-01-01.json").write_text("{}", encoding="utf-8")

    violations = scan_repository(scratch)
    joined = "\n".join(violations)
    assert any("real study cell identifier" in v for v in violations), joined
    assert any("real study seed literal" in v for v in violations), joined
    assert any("study clip prefix" in v for v in violations), joined
    assert any("study hash/fetch list file" in v for v in violations), joined
    assert any("sealed composition-divergence output" in v for v in violations), joined


def test_ffmpeg_encode_and_decode_roundtrip_if_available(tmp_path: Path) -> None:
    """Verify real H.264 MP4 encoding and frame 0/30/60 decoding when ffmpeg is on PATH."""
    fix = generate_synthetic_divergence_fixture(
        fixture_dir=tmp_path / "ffmpeg_fix",
        prompts=("PA",),
        seeds=(101, 202),
        arms=SUPER_ARMS,
        divergent_cell=("PA", 101),
        divergent_arm=ARM_Q3,
        include_nano=False,
        prefer_ffmpeg=True,
    )
    first_rel = next(iter(fix["clips"].keys()))
    frames = decode_clip_frames(Path(fix["clips_dir"]) / first_rel, frame_indices=FRAME_INDICES)
    assert set(frames.keys()) == {0, 30, 60}
    for f_arr in frames.values():
        assert f_arr.shape == (INPUT_HEIGHT, INPUT_WIDTH, 3)
        assert f_arr.dtype == np.uint8


@pytest.mark.skipif(
    not os.environ.get("BLINDPANEL_DINOV2_SNAPSHOT"),
    reason="Set BLINDPANEL_DINOV2_SNAPSHOT to a local facebook/dinov2-base snapshot directory to run real DINOv2 E2E test",
)
def test_optional_real_dinov2_end_to_end(
    synthetic_fixture_env: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Requirement 3: Optional end-to-end test using a local DINOv2 snapshot when BLINDPANEL_DINOV2_SNAPSHOT is set."""
    snapshot_dir = os.environ["BLINDPANEL_DINOV2_SNAPSHOT"]
    out_dir = tmp_path / "sealed_real_dinov2"
    res = run_composition_divergence_job(
        clips_dir=synthetic_fixture_env["clips_dir"],
        hashes_file=synthetic_fixture_env["hashes_file"],
        out_dir=out_dir,
        local_snapshot_path=snapshot_dir,
        date_tag="2099-01-01",
    )
    assert Path(res["output_path"]).is_file()
    assert Path(res["receipt_path"]).is_file()
    assert res["clips_verified"] == 18
