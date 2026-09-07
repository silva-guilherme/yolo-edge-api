#!/usr/bin/env python3
import argparse
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import subprocess

OUTPUT_DIR = Path("dataset/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_snapshot(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = resp.read()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame is not None, frame


def parse_args():
    p = argparse.ArgumentParser(description="Captura de frames para dataset")
    p.add_argument("--source", default="0")
    p.add_argument("--url", default="http://localhost:5001/stream")
    p.add_argument("--total", type=int, default=200)
    p.add_argument("--interval", type=float, default=1.5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--manual", action="store_true")
    p.add_argument("--snapshot-url", default="http://localhost:5001/snapshot")
    return p.parse_args()


class RpicamCapture:
    def __init__(self, device: int, width: int, height: int, fps: int = 15):
        cmd = [
            "rpicam-vid", "-t", "0", "-n", "--codec", "mjpeg",
            "--camera", str(device),
            "--width", str(width), "--height", str(height),
            "--framerate", str(fps),
            "-o", "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self._buf = b""

    def read(self):
        while True:
            start = self._buf.find(b"\xff\xd8")
            end = self._buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
            if start != -1 and end != -1:
                jpg = self._buf[start:end + 2]
                self._buf = self._buf[end + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return frame is not None, frame
            chunk = self._proc.stdout.read(4096)
            if not chunk:
                return False, None
            self._buf += chunk

    def isOpened(self):
        return self._proc.poll() is None

    def release(self):
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def open_capture(args):
    if args.source == "mjpeg":
        cap = cv2.VideoCapture(args.url)
    else:
        cap = RpicamCapture(int(args.source), args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte: {args.source}")
    return cap


def is_sharp_enough(frame: np.ndarray, threshold: float = 20.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= threshold


def main():
    args = parse_args()
    cap = open_capture(args)

    saved = 0
    skipped = 0
    last_saved = 0.0

    print(f"[INFO] Iniciando captura: {args.total} frames | intervalo: {args.interval}s")
    print(f"[INFO] Salvando em: {OUTPUT_DIR.resolve()}")
    print("[INFO] Pressione Ctrl+C para encerrar antecipadamente.")

    try:
        while saved < args.total:
            if args.manual:
                input(f"  [{saved:>3}/{args.total}] Pressione ENTER para capturar...")
                if args.source == "mjpeg":
                    ret, frame = fetch_snapshot(args.snapshot_url)
                else:
                    flush_until = time.time() + 0.5
                    while time.time() < flush_until:
                        cap.read()
                    ret, frame = cap.read()
            else:
                ret, frame = cap.read()

            if not ret:
                print("[AVISO] Frame invalido, tentando novamente...")
                time.sleep(0.1)
                continue

            if not args.manual:
                now = time.time()
                if now - last_saved < args.interval:
                    continue

            if not is_sharp_enough(frame):
                skipped += 1
                if args.manual:
                    print("  [AVISO] Frame borrado, descartado.")
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = OUTPUT_DIR / f"frame_{ts}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
            last_saved = time.time()
            print(f"  [{saved:>3}/{args.total}] Salvo: {path.name} (descartados: {skipped})")

    except KeyboardInterrupt:
        print("\n[INFO] Captura interrompida pelo usuario.")
    finally:
        cap.release()
        print(f"\n[OK] {saved} frames salvos em {OUTPUT_DIR}")
        print(f"[OK] {skipped} frames borrados descartados automaticamente")


if __name__ == "__main__":
    main()
