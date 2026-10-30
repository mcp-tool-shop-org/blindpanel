"""capture_real_clip_screenshots.py — Capture 1440×900 and 1024×768 screenshots of Block 1 with real B200 clips in both layouts."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

from blindpanel.cosmos3_study import DEFAULT_CLIPS_DIR, build_cosmos3_amendment_a_config
from blindpanel.server import BlindPanelSession, create_http_server


def _find_browser_executable() -> Optional[str]:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
    ]
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
        cfg = build_cosmos3_amendment_a_config(clips_dir=DEFAULT_CLIPS_DIR, verify_files_exist=True)
        cfg_path = tmp / "cosmos3_amendment_a.json"
        import json

        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        session = BlindPanelSession(
            config_path=str(cfg_path),
            sealed_map_path=str(tmp / "sealed_map_20261030.bin"),
            answers_path=str(tmp / "answers.jsonl"),
            verify_clip_shape=True,
        )
        session.start_session(mode="live")

        httpd = create_http_server(session, host="127.0.0.1", port=0)
        port = httpd.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        try:
            exe = _find_browser_executable()
            with sync_playwright() as p:
                launch_kwargs = {"headless": True}
                if exe:
                    launch_kwargs["executable_path"] = exe
                browser = p.chromium.launch(**launch_kwargs)

                # Item index 5 in Block 1 is Item 6 / 43 (M1-s2026 manipulation prompt with Q2)
                for width, height in [(1440, 900), (1024, 768)]:
                    for layout_mode in ["3up", "flip"]:
                        ctx = browser.new_context(viewport={"width": width, "height": height})
                        page = ctx.new_page()
                        url = f"{base_url}/?item=5&layout={layout_mode}&flip=B&unlock=1&t=1.25"
                        page.goto(url, wait_until="networkidle")
                        page.wait_for_selector("video[data-clip-letter='A']")

                        # Wait for all 3 real B200 MP4 videos to decode a frame at t=1.25s
                        page.evaluate(
                            """async () => {
                            const vids = Array.from(document.querySelectorAll('video'));
                            for (const v of vids) {
                              v.pause();
                              v.currentTime = 1.25;
                            }
                            await new Promise((resolve) => setTimeout(resolve, 600));
                            window.__blindpanel.setSelectionsForTest(
                              'B',
                              'C',
                              { A: 'partly', B: 'yes', C: 'no' },
                              'B lifts box clear; A partial; C stays on belt'
                            );
                          }"""
                        )
                        page.wait_for_timeout(400)

                        fname = f"block1-real-clips-{layout_mode}-{width}x{height}.png"
                        out_path = out_dir / fname
                        page.screenshot(path=str(out_path), full_page=False)
                        print(f"Captured {out_path}")
                        ctx.close()

                browser.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
