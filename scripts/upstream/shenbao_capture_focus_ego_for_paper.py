#!/usr/bin/env python3
"""Capture the paper's global strict-w10 exact-core ego network from the dashboard."""

from __future__ import annotations

import argparse
import base64
import json
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
import websocket


ROOT = Path(__file__).resolve().parent
DEFAULT_HTML = (
    ROOT
    / "shenbao"
    / "shenbao_interpretation"
    / "focus_anchor_dashboard_ver2"
    / "html"
    / "focus_multi_ego_network_dashboard.html"
)
DEFAULT_OUTPUT = (
    ROOT
    / "shenbao"
    / "shenbao_interpretation"
    / "focus_anchor_dashboard_ver2"
    / "paper_figure"
    / "focus_multi_ego_strict_w10_global_exact_core_top100.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", default=str(DEFAULT_HTML))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--browser",
        choices=("cdp-chrome", "chrome", "edge"),
        default="cdp-chrome",
    )
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def select_value(driver: webdriver.Remote, element_id: str, value: str) -> None:
    element = driver.find_element(By.ID, element_id)
    Select(element).select_by_value(value)


def ensure_checked(driver: webdriver.Remote, element_id: str) -> None:
    element = driver.find_element(By.ID, element_id)
    if not element.is_selected():
        element.click()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def cdp_call(connection: websocket.WebSocket, call_id: int, method: str, params: dict | None = None) -> dict:
    connection.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
    while True:
        message = json.loads(connection.recv())
        if message.get("id") == call_id:
            if "error" in message:
                raise RuntimeError(f"CDP {method} failed: {message['error']}")
            return message.get("result", {})


def capture_with_cdp(html_path: Path, raw_path: Path, timeout: int) -> None:
    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    if not chrome.is_file():
        raise FileNotFoundError(chrome)
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="shenbao_chrome_") as user_data:
        process = subprocess.Popen(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--allow-file-access-from-files",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={user_data}",
                "--window-size=1400,1200",
                "--force-device-scale-factor=1",
                html_path.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        connection = None
        try:
            deadline = time.monotonic() + timeout
            target = None
            while time.monotonic() < deadline:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as response:
                        targets = json.load(response)
                    target = next((item for item in targets if item.get("type") == "page"), None)
                    if target:
                        break
                except OSError:
                    pass
                time.sleep(0.25)
            if not target:
                raise TimeoutError("Chrome DevTools page target did not become available")

            connection = websocket.create_connection(
                target["webSocketDebuggerUrl"], timeout=30, suppress_origin=True
            )
            call_id = 1

            def evaluate(expression: str):
                nonlocal call_id
                result = cdp_call(
                    connection,
                    call_id,
                    "Runtime.evaluate",
                    {"expression": expression, "returnByValue": True},
                )
                call_id += 1
                value = result.get("result", {})
                if value.get("subtype") == "error":
                    raise RuntimeError(value.get("description", "JavaScript evaluation failed"))
                return value.get("value")

            while time.monotonic() < deadline:
                ready = evaluate(
                    "document.readyState === 'complete' && "
                    "document.getElementById('profile') !== null && "
                    "document.getElementById('profile').options.length >= 3"
                )
                if ready:
                    break
                time.sleep(0.5)
            else:
                raise TimeoutError("Dashboard controls did not become available")

            for element_id, value in (
                ("profile", "strict"),
                ("window", "10"),
                ("centerMode", "exact_core_only"),
                ("periodSet", "global"),
                ("period", "global"),
                ("topN", "100"),
            ):
                changed = evaluate(
                    "(() => {"
                    f"const e=document.getElementById({json.dumps(element_id)});"
                    f"const v={json.dumps(value)};"
                    "if(!e || !Array.from(e.options).some(o => o.value===v)) return false;"
                    "e.value=v; e.dispatchEvent(new Event('change',{bubbles:true})); return true;"
                    "})()"
                )
                if not changed:
                    raise RuntimeError(f"Dashboard option is unavailable: {element_id}={value}")

            evaluate("document.getElementById('selectMajor').click()")
            evaluate("document.getElementById('selectAllPos').click()")
            for element_id in ("markAmbiguousPos", "shapeByPos"):
                evaluate(
                    "(() => {"
                    f"const e=document.getElementById({json.dumps(element_id)});"
                    "if(!e.checked){e.click();} return e.checked;"
                    "})()"
                )

            while time.monotonic() < deadline:
                if int(evaluate("document.querySelectorAll('#graph g.node').length") or 0) >= 100:
                    break
                time.sleep(0.5)
            else:
                raise TimeoutError("Network nodes did not render")
            time.sleep(2)

            rect = evaluate(
                "(() => {const r=document.getElementById('graph').getBoundingClientRect();"
                "return {x:r.x,y:r.y,width:r.width,height:r.height};})()"
            )
            result = cdp_call(
                connection,
                call_id,
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "captureBeyondViewport": True,
                    "clip": {**rect, "scale": 1},
                },
            )
            raw_path.write_bytes(base64.b64decode(result["data"]))
        finally:
            if connection is not None:
                connection.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> None:
    args = parse_args()
    html_path = Path(args.html).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not html_path.is_file():
        raise FileNotFoundError(html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(f".{output_path.stem}.raw.png")

    if args.browser == "cdp-chrome":
        capture_with_cdp(html_path, raw_path, args.timeout)
        with Image.open(raw_path) as source:
            image = source.convert("RGBA")
            image = image.resize((991, 768), Image.Resampling.LANCZOS)
            image.save(output_path, format="PNG")
        raw_path.unlink(missing_ok=True)
        print(f"Wrote PNG: {output_path}")
        return

    options = webdriver.ChromeOptions() if args.browser == "chrome" else webdriver.EdgeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--allow-file-access-from-files")
    options.add_argument("--window-size=1400,1200")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--force-device-scale-factor=1")
    driver_class = webdriver.Chrome if args.browser == "chrome" else webdriver.Edge

    driver = driver_class(options=options)
    try:
        driver.set_page_load_timeout(args.timeout)
        driver.get(html_path.as_uri())
        wait = WebDriverWait(driver, args.timeout)
        wait.until(lambda d: len(Select(d.find_element(By.ID, "profile")).options) >= 3)

        select_value(driver, "profile", "strict")
        select_value(driver, "window", "10")
        select_value(driver, "centerMode", "exact_core_only")
        select_value(driver, "periodSet", "global")
        select_value(driver, "period", "global")
        select_value(driver, "topN", "100")
        driver.find_element(By.ID, "selectMajor").click()
        driver.find_element(By.ID, "selectAllPos").click()
        ensure_checked(driver, "markAmbiguousPos")
        ensure_checked(driver, "shapeByPos")

        wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#graph g.node")) >= 100)
        time.sleep(2)
        graph = driver.find_element(By.ID, "graph")
        graph.screenshot(str(raw_path))
    finally:
        driver.quit()

    with Image.open(raw_path) as source:
        image = source.convert("RGBA")
        image = image.resize((991, 768), Image.Resampling.LANCZOS)
        image.save(output_path, format="PNG")
    raw_path.unlink(missing_ok=True)
    print(f"Wrote PNG: {output_path}")


if __name__ == "__main__":
    main()
