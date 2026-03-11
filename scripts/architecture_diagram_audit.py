#!/usr/bin/env python3
"""
Architecture diagram audit - check text labels fit within boxes.
"""

import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SITE_ROOT = REPO_ROOT.parent / "josueortc.github.io"
OUTPUT_DIR = REPO_ROOT / "docs" / "audit-screenshots"
PORT = 9876


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        ["python3", "-m", "http.server", str(PORT)],
        cwd=str(SITE_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1400, "height": 900},
                device_scale_factor=2,
            )
            page = context.new_page()

            url = f"http://localhost:{PORT}/projects/prismt/index.html"
            page.goto(url, wait_until="networkidle", timeout=15000)
            page.wait_for_timeout(1000)

            # 1. Scroll to Architecture section
            page.locator("#architecture").scroll_into_view_if_needed()
            page.wait_for_timeout(600)

            # 2. Screenshot PRISMTransformer (Classification) - first tab active by default
            page.screenshot(path=str(OUTPUT_DIR / "arch-01-prismt-classification.png"))
            print("1. Screenshot: arch-01-prismt-classification.png (Classification)")

            # 3. Click PRISMTransformer tab (second tab)
            page.locator('.arch-tab[data-tab="reconstruction"]').click()
            page.wait_for_timeout(500)

            # 4. Screenshot PRISMTransformer diagram
            page.screenshot(path=str(OUTPUT_DIR / "arch-02-prism-transformer.png"))
            print("2. Screenshot: arch-02-prism-transformer.png (PRISMTransformer)")

            # Zoom in on the architecture diagram area for closer inspection
            arch_diagram = page.locator("#arch-reconstruction")
            arch_diagram.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            arch_diagram.screenshot(path=str(OUTPUT_DIR / "arch-03-prism-transformer-closeup.png"))
            print("3. Screenshot: arch-03-prism-transformer-closeup.png (diagram only)")

            browser.close()
    finally:
        server.kill()

    print(f"\nScreenshots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
