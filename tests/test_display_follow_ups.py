"""test_display_follow_ups.py — Display fixes found on the first full-session screens (mock clips only).

1. Counts on the cards. On the public study design (Block 1: 43 items, Block 2: 42 optional pairs), the
   block-transition card and the Session Complete card show the same total the session counter used for the
   finished block ("43 / 43"), never the next block's total (the old card read "42 / 42") and never a hardcoded 43.
   The rater sees no warm-up or repeat counts anywhere.
2. Tile size. At 3440×1440 the three <video> elements render at exactly 832 CSS px wide (device scale 1) in the
   default layout, with `?layout=3up`, and after clicking the 3-Up toggle; the zoom buttons show 1:1 as active there.
   At 1440×900 the default is the single-clip flip, whose active <video> is also 832 px wide.
3. Text size. At 3440×1440 every visible text element is at least 14 px; at 1440×900 the sizes are unchanged
   (body 13 px, nothing under 12 px).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
from playwright.sync_api import Page, sync_playwright

from blindpanel.cosmos3_study import build_public_study_design_config
from blindpanel.mock_clips import generate_mock_study_clips
from blindpanel.server import BlindPanelSession, create_http_server
from tests.test_security_and_leaks import FORBIDDEN_RATER_TOKENS

WIDE = {"width": 3440, "height": 1440}
LAPTOP = {"width": 1440, "height": 900}

FONT_AUDIT_JS = """() => {
  const out = [];
  document.querySelectorAll("*").forEach((el) => {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0 && el.children.length === 0 && (el.textContent || "").trim().length > 0) {
      const cs = window.getComputedStyle(el);
      if (cs.visibility === "hidden" || cs.display === "none") return;
      out.push({ tag: el.tagName, id: el.id || "", cls: String(el.className || ""), text: el.textContent.trim().slice(0, 40),
                 fontSize: parseFloat(cs.fontSize) });
    }
  });
  return out;
}"""

VIDEO_WIDTHS_JS = """() => Array.from(document.querySelectorAll("video[data-clip-letter]")).map((v) => {
  const r = v.getBoundingClientRect();
  return { letter: v.getAttribute("data-clip-letter"), width: r.width, height: r.height };
})"""


def _find_browser_executable() -> Optional[str]:
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("msedge"),
    ]
    for env_key in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
        base = os.environ.get(env_key)
        if base:
            candidates.append(os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"))
            candidates.append(os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _launch(p):
    kwargs = {"headless": True}
    exe = _find_browser_executable()
    if exe:
        kwargs["executable_path"] = exe
    return p.chromium.launch(**kwargs)


def _serve(session: BlindPanelSession) -> Tuple[object, str]:
    httpd = create_http_server(session, host="127.0.0.1", port=0)
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, base_url


def _public_study_session(tmp_path: Path, with_mock_mp4: bool) -> Tuple[BlindPanelSession, Dict]:
    """The public cosmos3 design (43 + 42 items) on mock clips. Real MP4s only when a browser must play them."""
    cfg = build_public_study_design_config()
    cfg_path = tmp_path / "study_design.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    cells = sorted({it["base_cell_id"] for blk in cfg["blocks"] for it in blk["items"]})
    arms = sorted({arm for blk in cfg["blocks"] for arm in blk["arms"]})
    clips_dir = tmp_path / "mock_clips"
    if with_mock_mp4:
        generate_mock_study_clips(clips_dir, arms=arms, base_cells=cells)
    else:
        clips_dir.mkdir(parents=True, exist_ok=True)
        for cid in cells:
            for arm in arms:
                (clips_dir / f"{arm}_{cid}.mp4").write_bytes(f"MOCK_MP4_{arm}_{cid}".encode("utf-8"))
    session = BlindPanelSession(
        config_path=str(cfg_path),
        clips_dir=str(clips_dir),
        sealed_map_path=str(tmp_path / "sealed_store" / "sealed_map_session.bin"),
        answers_path=str(tmp_path / "answers.jsonl"),
    )
    return session, cfg


def _rate_items_server_side(session: BlindPanelSession, global_indices: List[int], auto_resume: bool = True) -> None:
    """Any valid answer per item, submitted through the session engine (no browser, no clip playback)."""
    for gi in global_indices:
        st = session.get_public_state(mode="live", requested_global_index=gi)
        letters = st["letters"]
        payload: Dict = {"mode": "live", "global_index": gi}
        if st["method"] in ("bws_3", "bws"):
            payload["best_letter"] = letters[0]
            payload["worst_letter"] = letters[-1]
        else:
            payload["choice"] = letters[0]
        if st["ask_q2"]:
            payload["q2_by_letter"] = {letter: "yes" for letter in letters}
        session.submit_rating(payload)
        after = session.get_public_state(mode="live")
        if auto_resume and after["interstitial"] == "rest_break":
            session.resume_after_break(mode="live")


def _assert_no_internals(page: Page, label: str) -> str:
    text = page.inner_text("body")
    for tok in FORBIDDEN_RATER_TOKENS:
        assert tok not in text, f"[{label}] forbidden token {tok!r} on a rater card"
    return text


def test_block_transition_and_session_complete_cards_count_the_finished_block(tmp_path: Path) -> None:
    session, cfg = _public_study_session(tmp_path, with_mock_mp4=False)
    b1 = cfg["blocks"][0]
    b2 = cfg["blocks"][1]
    n_b1 = len(b1["items"])
    n_b2 = len(b2["items"])
    assert (n_b1, n_b2) == (43, 42)
    n_warm_b1 = sum(1 for it in b1["items"] if it["is_warmup"])
    n_rep_b1 = sum(1 for it in b1["items"] if it["is_repeat"])
    assert (n_warm_b1, n_rep_b1, n_b1 - n_warm_b1) == (3, 4, 40)

    session.start_session(mode="live")
    _rate_items_server_side(session, list(range(n_b1)))

    st_break = session.get_public_state(mode="live")
    assert st_break["interstitial"] == "block_break"
    # The active item at the block break is the first Block 2 item (total 42); the finished block's total is reported apart.
    assert st_break["block_total_items"] == n_b2
    assert st_break["prev_block_total_items"] == n_b1
    assert st_break["completed_blocks"] == [{"block_number": 1, "total_items": n_b1}]

    httpd, base_url = _serve(session)
    try:
        with sync_playwright() as p:
            browser = _launch(p)
            page = browser.new_context(viewport=LAPTOP).new_page()

            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("#btn-resume-block2")
            text = _assert_no_internals(page, "block_break")
            assert "Item 43 / 43" in text
            assert "All 43 items of Block 1 are complete and saved on disk" in text
            assert "Block 2 (42 side-by-side pairs) is optional" in text
            assert "42 / 42" not in text
            assert "Save & Close" in text and "Resume Block 2" in text
            assert page.inner_text("#break-timer-display").strip() == "10 : 00" or page.inner_text(
                "#break-timer-display"
            ).strip().startswith("09 : 5")

            page.click("#btn-save-close")
            page.wait_for_selector("#session-saved-message")
            st_done = session.get_public_state(mode="live")
            assert st_done["interstitial"] == "session_complete"
            assert st_done["completed_blocks"] == [{"block_number": 1, "total_items": n_b1}]

            text = _assert_no_internals(page, "session_complete")
            assert "Session Complete — 43 / 43 items rated" in text
            assert "100% Complete (Item 43 / 43)" in text
            assert "All 43 items saved and verified" in text
            assert "42" not in text
            assert page.inner_text("#counter-badge-complete").strip() == "Item 43 / 43"

            # A fresh load of the completed session shows the same card.
            page.reload(wait_until="networkidle")
            page.wait_for_selector("#session-saved-message")
            assert "Session Complete — 43 / 43 items rated" in page.inner_text("body")
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_session_complete_lists_every_finished_block_when_block_2_is_rated(tmp_path: Path) -> None:
    session, cfg = _public_study_session(tmp_path, with_mock_mp4=False)
    n_b1 = len(cfg["blocks"][0]["items"])
    n_b2 = len(cfg["blocks"][1]["items"])
    session.start_session(mode="live")
    _rate_items_server_side(session, list(range(n_b1)))
    session.start_block2(mode="live")
    _rate_items_server_side(session, list(range(n_b1, n_b1 + n_b2)))
    st = session.get_public_state(mode="live")
    assert st["interstitial"] == "session_complete"
    assert st["completed_blocks"] == [
        {"block_number": 1, "total_items": n_b1},
        {"block_number": 2, "total_items": n_b2},
    ]

    httpd, base_url = _serve(session)
    try:
        with sync_playwright() as p:
            browser = _launch(p)
            page = browser.new_context(viewport=LAPTOP).new_page()
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("#session-saved-message")
            text = _assert_no_internals(page, "session_complete_both_blocks")
            assert "Block 1: 43 / 43 · Block 2: 42 / 42 items rated" in text
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_native_832px_tiles_are_the_default_at_3440_and_flip_at_1440(tmp_path: Path) -> None:
    session, cfg = _public_study_session(tmp_path, with_mock_mp4=True)
    session.start_session(mode="live")
    httpd, base_url = _serve(session)
    try:
        with sync_playwright() as p:
            browser = _launch(p)

            def open_item(viewport: Dict[str, int], query: str) -> Page:
                ctx = browser.new_context(viewport=viewport, device_scale_factor=1)
                page = ctx.new_page()
                page.goto(f"{base_url}/{query}", wait_until="networkidle")
                page.wait_for_selector("video[data-clip-letter='A']")
                page.wait_for_timeout(200)
                return page

            # 3440x1440: default, explicit ?layout=3up, and the 3-Up toggle all give three 832 px videos and zoom 1:1.
            for query in ("", "?layout=3up"):
                page = open_item(WIDE, query)
                assert page.get_attribute("#video-grid-container", "data-effective-layout") == "native", query
                vids = page.evaluate(VIDEO_WIDTHS_JS)
                assert [v["letter"] for v in vids] == ["A", "B", "C"]
                assert all(v["width"] == 832 and v["height"] == 480 for v in vids), (query, vids)
                active_zoom = page.evaluate("() => Array.from(document.querySelectorAll('[data-zoom].active')).map(b => b.dataset.zoom)")
                assert active_zoom == ["1:1"], (query, active_zoom)
                page.context.close()

            page = open_item(WIDE, "?layout=flip")
            assert page.get_attribute("#video-grid-container", "data-effective-layout") == "flip"
            page.click("[data-layout-mode='3up']")
            page.wait_for_timeout(200)
            assert page.get_attribute("#video-grid-container", "data-effective-layout") == "native"
            vids = page.evaluate(VIDEO_WIDTHS_JS)
            assert all(v["width"] == 832 and v["height"] == 480 for v in vids), vids
            page.context.close()

            # 1440x900: the flip is the default; its active video is 832 px wide; Fit is the zoom shown.
            page = open_item(LAPTOP, "")
            assert page.get_attribute("#video-grid-container", "data-effective-layout") == "flip"
            vids = page.evaluate(VIDEO_WIDTHS_JS)
            shown = [v for v in vids if v["width"] > 1]
            assert len(shown) == 1 and shown[0]["letter"] == "A" and shown[0]["width"] == 832, vids
            active_zoom = page.evaluate("() => Array.from(document.querySelectorAll('[data-zoom].active')).map(b => b.dataset.zoom)")
            assert active_zoom == ["fit"]
            page.context.close()

            # 1440x900 with 3-Up asked for: the scaled grid (the window cannot fit three native tiles).
            page = open_item(LAPTOP, "?layout=3up")
            assert page.get_attribute("#video-grid-container", "data-effective-layout") == "3up"
            vids = page.evaluate(VIDEO_WIDTHS_JS)
            assert len(vids) == 3 and all(0 < v["width"] < 832 for v in vids), vids
            page.context.close()
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize(
    "viewport,min_font,body_font",
    [
        (WIDE, 14.0, 16.25),
        (LAPTOP, 12.0, 13.0),
    ],
)
def test_text_size_floor_per_viewport(tmp_path: Path, viewport: Dict[str, int], min_font: float, body_font: float) -> None:
    session, cfg = _public_study_session(tmp_path, with_mock_mp4=True)
    n_b1 = len(cfg["blocks"][0]["items"])
    m1_index = next(i for i, it in enumerate(session.flat_items) if it["prompt_id"] == "M1" and not it["is_warmup"])
    httpd, base_url = _serve(session)
    try:
        with sync_playwright() as p:
            browser = _launch(p)
            ctx = browser.new_context(viewport=viewport, device_scale_factor=1)
            page = ctx.new_page()

            def audit(label: str) -> None:
                assert page.evaluate("() => parseFloat(getComputedStyle(document.body).fontSize)") == body_font, label
                rows = page.evaluate(FONT_AUDIT_JS)
                assert rows, label
                small = [r for r in rows if r["fontSize"] < min_font]
                assert not small, f"[{label}] text under {min_font}px at {viewport['width']}x{viewport['height']}: {small}"

            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("#btn-begin-session")
            audit("session_start")

            session.start_session(mode="live")
            page.goto(f"{base_url}/?item={m1_index}", wait_until="networkidle")
            page.wait_for_selector("video[data-clip-letter='A']")
            audit("active_item_with_q2")

            _rate_items_server_side(session, list(range(23)), auto_resume=False)
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("#btn-resume-break")
            audit("rest_break")
            session.resume_after_break(mode="live")

            _rate_items_server_side(session, list(range(23, n_b1)))
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("#btn-resume-block2")
            audit("block_break")

            session.decline_block2(mode="live")
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("#session-saved-message")
            audit("session_complete")
            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
