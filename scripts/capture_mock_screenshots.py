"""capture_mock_screenshots.py — Capture 1440×900 and 1024×768 screenshots of Block 1 using synthetic mock clips only.

Enforces Rule 4 & Rule 5:
- Generates synthetic mock clips via `blindpanel.mock_clips` in a temporary scratch directory.
- Captures 1440×900 and 1024×768 screenshots in both 3-Up and 1:1 Flip layouts under `docs/screenshots/`.
- Stamps every PNG with the `blindpanel-mock-clip-generator-v1` metadata marker and writes
  `docs/screenshots/mock_manifest.json`.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

from playwright.sync_api import sync_playwright

from blindpanel.cosmos3_study import build_public_study_design_config
from blindpanel.mock_clips import (
    generate_mock_study_clips,
    stamp_png_with_mock_marker,
    write_mock_screenshots_manifest,
)
from blindpanel.server import BlindPanelSession, create_http_server


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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "docs" / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg = build_public_study_design_config()
        cfg_path = tmp / "study_design.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        all_cells = sorted({it["base_cell_id"] for blk in cfg["blocks"] for it in blk["items"]})
        all_arms = sorted({arm for blk in cfg["blocks"] for arm in blk["arms"]})
        clips_dir = tmp / "mock_clips"
        generate_mock_study_clips(clips_dir, arms=all_arms, base_cells=all_cells)

        session = BlindPanelSession(
            config_path=str(cfg_path),
            clips_dir=str(clips_dir),
            sealed_map_path=str(tmp / "sealed_store" / "sealed_map_session.bin"),
            answers_path=str(tmp / "answers.jsonl"),
        )
        session.start_session(mode="live")

        httpd = create_http_server(session, host="127.0.0.1", port=0)
        port = httpd.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        captured_names: List[str] = []
        try:
            exe = _find_browser_executable()
            with sync_playwright() as p:
                launch_kwargs = {"headless": True}
                if exe:
                    launch_kwargs["executable_path"] = exe
                browser = p.chromium.launch(**launch_kwargs)

                # Item index 5 in Block 1 is Item 6 / 43 (M1 manipulation prompt with Q2)
                for width, height in [(1440, 900), (1024, 768)]:
                    for layout_mode in ["3up", "flip"]:
                        ctx = browser.new_context(viewport={"width": width, "height": height})
                        page = ctx.new_page()
                        url = f"{base_url}/?item=5&layout={layout_mode}&flip=B&unlock=1&t=0.5"
                        page.goto(url, wait_until="networkidle")
                        page.wait_for_selector("video[data-clip-letter='A']")

                        page.evaluate(
                            """async () => {
                            const vids = Array.from(document.querySelectorAll('video'));
                            for (const v of vids) {
                              v.pause();
                              v.currentTime = 0.5;
                            }
                            await new Promise((resolve) => setTimeout(resolve, 400));
                            window.__blindpanel.setSelectionsForTest(
                              'B',
                              'C',
                              { A: 'partly', B: 'yes', C: 'no' },
                              'Mock clip verification preview'
                            );
                          }"""
                        )
                        page.wait_for_timeout(300)

                        fname = f"block1-mock-clips-{layout_mode}-{width}x{height}.png"
                        out_path = out_dir / fname
                        page.screenshot(path=str(out_path), full_page=False)
                        stamp_png_with_mock_marker(out_path)
                        captured_names.append(fname)
                        print(f"Captured and stamped mock screenshot: {out_path}")
                        ctx.close()

                browser.close()
        finally:
            httpd.shutdown()
            httpd.server_close()

    manifest_path = write_mock_screenshots_manifest(out_dir, captured_names)
    print(f"Wrote mock screenshot manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
