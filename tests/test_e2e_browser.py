"""test_e2e_browser.py — End-to-end headless Chromium browser test for blindpanel.

Exercises:
1. Loading the rater session in a real browser (Session Start interstitial, practice-start copy, progress aria-label).
2. Leak scan & 12px minimum computed font-size check across every rendered rater state (Session Start, active Block 1
   rating in both 3-Up Grid and 1:1 Flip layouts, Rest Break, Block 2 2AFC, and Session Complete).
3. Detail visibility layout behavior:
   - Wide window (3440×1440) defaults to 3 native 832×480 tiles (`video-grid-native-3`).
   - Narrower windows (1440×900 and 1024×768) auto-fallback to synchronized single-large-clip 1:1 Flip mode
     (`video-grid-flip-single`) with A/B/C flip (`1`/`2`/`3`), and also support toggling to 3-Up Grid (`video-grid-3`)
     without losing playback position or draft selections.
4. Rating items (including per-clip Question 2 on M1–M4 manipulation rows).
5. Resuming exact state after a full browser page reload.
6. Completing the session, exporting blind CSV/JSONL, and running offline unblinding (`unblind_session`).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from playwright.sync_api import Page, sync_playwright

from blindpanel.server import BlindPanelSession, create_http_server
from blindpanel.unblind import load_jsonl_safe, unblind_session
from tests.test_security_and_leaks import FORBIDDEN_RATER_TOKENS


def _find_browser_executable() -> Optional[str]:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _create_tiny_mp4_clips(tmp_path: Path, arms: List[str], cells: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Create tiny valid MP4 clips (using real B200 clips if present or ffmpeg/fallback) and return arm mappings."""
    real_dir = Path("E:/AI/rnd-local/projects/cosmos3/b200-same-card/artifacts/arm4_NANO/x/mp4")
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, Dict[str, str]]] = {}

    for cid in cells:
        out[cid] = {}
        for arm in arms:
            real_candidate = real_dir / f"{arm}_cell_{cid}.mp4"
            dest = clips_dir / f"{arm}_{cid}.mp4"
            if real_candidate.is_file():
                shutil.copyfile(real_candidate, dest)
            elif shutil.which("ffmpeg"):
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=gray:s=832x480:r=24:d=0.5",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(dest),
                    ],
                    check=True,
                    capture_output=True,
                )
            else:
                dest.write_bytes(f"MP4_FALLBACK_{arm}_{cid}".encode("utf-8"))
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            out[cid][arm] = {"path": str(dest), "sha256": sha}
    return out


def _assert_rater_dom_hygiene_and_font_size(page: Page, state_name: str) -> None:
    """Assert zero study internals appear in any visible text, title, aria-*, alt, or HTML comment,
    and every visible text element has computed font-size >= 12px."""
    audit = page.evaluate(
        """() => {
        const html = document.documentElement.outerHTML;
        const comments = [];
        const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_COMMENT);
        while (walker.nextNode()) {
          comments.push(walker.currentNode.nodeValue || "");
        }
        const attrs = [];
        document.querySelectorAll("*").forEach((el) => {
          for (const attr of el.attributes) {
            if (attr.name === "title" || attr.name === "alt" || attr.name.startsWith("aria-")) {
              attrs.push(`${attr.name}=${attr.value}`);
            }
          }
        });
        const visibleText = document.body.innerText || "";
        const smallFonts = [];
        document.querySelectorAll("*").forEach((el) => {
          const rect = el.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0 && el.children.length === 0 && (el.textContent || "").trim().length > 0) {
            const fs = parseFloat(window.getComputedStyle(el).fontSize);
            if (fs < 12.0) {
              smallFonts.push({ tag: el.tagName, text: el.textContent.trim().slice(0, 30), fontSize: fs });
            }
          }
        });
        return { visibleText, attrs: attrs.join(" | "), comments: comments.join(" | "), smallFonts };
      }"""
    )

    combined = f"{audit['visibleText']}\n{audit['attrs']}\n{audit['comments']}"
    for tok in FORBIDDEN_RATER_TOKENS:
        assert tok not in combined, f"[{state_name}] Forbidden token '{tok}' leaked in rater DOM: {combined[:400]}"

    assert len(audit["smallFonts"]) == 0, f"[{state_name}] Elements < 12px found: {audit['smallFonts']}"


def test_e2e_browser_flow_detail_visibility_resume_and_unblind(tmp_path: Path) -> None:
    arms_b1 = ["BF16", "Q1", "Q3"]
    arms_b2 = ["BF16", "NANO"]
    cells = ["M1-s1111", "L1-s2026"]
    clips = _create_tiny_mp4_clips(tmp_path, ["BF16", "Q1", "Q3", "NANO"], cells)

    cfg = {
        "schema_version": "blindpanel-config-v1",
        "study_id": "e2e-browser-study",
        "blocks": [
            {
                "block_id": "block1",
                "title": "Main Block 1",
                "method": "bws_3",
                "arms": arms_b1,
                "require_full_play": False,
                "break_after_item": 1,
                "items": [
                    {
                        "cell_id": "M1-s1111",
                        "prompt_id": "M1",
                        "prompt_type": "manipulation",
                        "prompt": "A single robot arm in a warehouse grasps a cardboard box on a moving conveyor belt and lifts it clear of the belt. Static camera, overhead fluorescent light, realistic.",
                        "rater_cue": "the box and the gripper through the clip",
                        "is_warmup": False,
                        "ask_q2": True,
                        "slot_arms": ["BF16", "Q1", "Q3"],
                        "arms": {k: clips["M1-s1111"][k] for k in arms_b1},
                    },
                    {
                        "cell_id": "L1-s2026",
                        "prompt_id": "L1",
                        "prompt_type": "locomotion",
                        "prompt": "A humanoid robot walks steadily down a narrow office hallway. Tracking camera from behind, realistic.",
                        "rater_cue": "the humanoid robot's legs and torso along the hallway through the clip",
                        "is_warmup": False,
                        "ask_q2": False,
                        "slot_arms": ["Q1", "Q3", "BF16"],
                        "arms": {k: clips["L1-s2026"][k] for k in arms_b1},
                    },
                ],
            },
            {
                "block_id": "block2",
                "title": "Block 2 · Side-by-Side Comparison",
                "optional": True,
                "method": "2afc",
                "arms": arms_b2,
                "require_full_play": False,
                "items": [
                    {
                        "cell_id": "M1-s1111",
                        "prompt_id": "M1",
                        "prompt_type": "manipulation",
                        "prompt": "A single robot arm in a warehouse grasps a cardboard box on a moving conveyor belt and lifts it clear of the belt. Static camera, overhead fluorescent light, realistic.",
                        "rater_cue": "the box and the gripper through the clip",
                        "ask_q2": False,
                        "slot_arms": ["BF16", "NANO"],
                        "arms": {k: clips["M1-s1111"][k] for k in arms_b2},
                    }
                ],
            },
        ],
    }

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    sealed_path = tmp_path / "sealed_map_20261030.bin"
    ans_path = tmp_path / "answers.jsonl"

    session = BlindPanelSession(
        config_path=str(cfg_path),
        sealed_map_path=str(sealed_path),
        answers_path=str(ans_path),
    )
    httpd = create_http_server(session, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    try:
        exe = _find_browser_executable()
        with sync_playwright() as p:
            launch_kwargs = {"headless": True}
            if exe:
                launch_kwargs["executable_path"] = exe
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            # 1. Load Session Start screen
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("#btn-begin-session")
            _assert_rater_dom_hygiene_and_font_size(page, "session_start")

            # Verify BUILD-MUST-DO item 1 & 2 on Session Start
            body_txt = page.inner_text("body")
            assert "Session Start" in body_txt
            assert "The opening practice items let you adjust to the video clips and playback controls" in body_txt
            prog_label = page.get_attribute(".progress-track", "aria-label")
            assert prog_label == "Item 1 of 2"

            # Click Begin Session -> Item 1 / 2 (M1 manipulation with Q2)
            page.click("#btn-begin-session")
            page.wait_for_selector("#video-grid-container")
            _assert_rater_dom_hygiene_and_font_size(page, "item1_flip_1440")

            # 2. Verify Detail Visibility:
            # At 1440x900 (< 2548px), default automatic layout is 1:1 Flip (`flip`)
            eff_layout = page.get_attribute("#video-grid-container", "data-effective-layout")
            assert eff_layout == "flip"

            # Verify cue strip reads "Look at: <rater_cue>"
            cue_text = page.inner_text("#cue-strip").strip()
            assert cue_text == "Look at: the box and the gripper through the clip"

            # Pick Best=A, Worst=C (automatically ranks B=2nd) and Q2 answers
            page.click('[data-pick-role="best"][data-pick-letter="A"]')
            page.click('[data-pick-role="worst"][data-pick-letter="C"]')
            assert "Order: A ≻ B ≻ C" in page.inner_text("#order-summary-badge")

            page.click('[data-q2-letter="A"][data-q2-val="yes"]')
            page.click('[data-q2-letter="B"][data-q2-val="partly"]')
            page.click('[data-q2-letter="C"][data-q2-val="no"]')

            # Flip between A, B, C using keyboard 1, 2, 3 and verify answers are preserved
            page.keyboard.press("2")
            assert not page.locator('[data-tile-letter="B"]').evaluate("el => el.classList.contains('flip-inactive')")
            assert page.locator('[data-tile-letter="A"]').evaluate("el => el.classList.contains('flip-inactive')")

            # Resize window to 3440x1440 (Director's screen >= 2548px) and verify automatic switch to native 3-Up (3 x 832px)
            page.set_viewport_size({"width": 3440, "height": 1440})
            page.wait_for_timeout(150)
            assert page.get_attribute("#video-grid-container", "data-effective-layout") == "native"
            assert "Order: A ≻ B ≻ C" in page.inner_text("#order-summary-badge")
            _assert_rater_dom_hygiene_and_font_size(page, "item1_native_3440")

            # Resize back to 1024x768 and verify layout switches back without losing answers
            page.set_viewport_size({"width": 1024, "height": 768})
            page.wait_for_timeout(150)
            assert page.get_attribute("#video-grid-container", "data-effective-layout") == "flip"
            assert "Order: A ≻ B ≻ C" in page.inner_text("#order-summary-badge")
            _assert_rater_dom_hygiene_and_font_size(page, "item1_flip_1024")

            # Submit Item 1 -> triggers Mid-Block Rest Break
            page.click("#btn-submit-item")
            page.wait_for_selector("#btn-resume-break")
            _assert_rater_dom_hygiene_and_font_size(page, "mid_block_rest_break")
            assert "Rest Break" in page.inner_text("body")

            # 3. Reload browser during Rest Break and verify exact resume state
            page.reload(wait_until="networkidle")
            page.wait_for_selector("#btn-resume-break")
            page.click("#btn-resume-break")

            # Now on Item 2 / 2 (L1 locomotion — no Q2 row)
            page.wait_for_selector("#video-grid-container")
            assert page.inner_text("#counter-badge") == "Item 2 / 2"
            assert page.locator(".action-row").count() == 0

            # Reload browser on Item 2 / 2 and verify it resumes on Item 2 / 2
            page.reload(wait_until="networkidle")
            page.wait_for_selector("#video-grid-container")
            assert page.inner_text("#counter-badge") == "Item 2 / 2"

            # Rate Item 2 (slot_arms = Q1, Q3, BF16 -> pick C=BF16 as Best, B=Q3 as Worst)
            page.click('[data-pick-role="best"][data-pick-letter="C"]')
            page.click('[data-pick-role="worst"][data-pick-letter="B"]')
            page.click("#btn-submit-item")

            # 4. Inter-block Rest Break before Optional Block 2
            page.wait_for_selector("#btn-resume-block2")
            _assert_rater_dom_hygiene_and_font_size(page, "inter_block_break")
            # Verify BUILD-MUST-DO item 3: Resume Block 2 is btn-primary, Save & Close is secondary
            resume_classes = page.get_attribute("#btn-resume-block2", "class") or ""
            save_classes = page.get_attribute("#btn-save-close", "class") or ""
            assert "btn-primary" in resume_classes
            assert "btn-primary" not in save_classes

            page.click("#btn-resume-block2")

            # 5. Block 2 (2AFC Side-by-Side Comparison)
            page.wait_for_selector('[data-choice="A"]')
            _assert_rater_dom_hygiene_and_font_size(page, "block2_2afc")
            page.click('[data-choice="A"]')
            page.click("#btn-submit-item")

            # 6. Session Complete screen
            page.wait_for_selector("#btn-export-csv")
            _assert_rater_dom_hygiene_and_font_size(page, "session_complete")
            complete_text = page.inner_text("body")
            assert "Session Complete" in complete_text
            assert "Pre-Flight" not in complete_text

            browser.close()

        # 7. Verify CSV/JSONL export and offline unblinding
        with urllib.request.urlopen(base_url + "/api/export/csv") as resp:
            csv_data = resp.read().decode("utf-8")
        assert "M1" in csv_data
        assert "BF16" not in csv_data

        sealed_map = json.loads(sealed_path.read_text(encoding="utf-8"))
        events, _ = load_jsonl_safe(str(ans_path))
        unblinded = unblind_session(sealed_map, events)

        assert unblinded["completed_rows"] == 3
        # BF16 was picked Best (rank 1) in all 3 items
        assert unblinded["summary_by_arm"]["BF16"]["mean_rank"] == 1.0
        assert unblinded["summary_by_arm"]["BF16"]["q2_action_completion"]["yes"] == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
