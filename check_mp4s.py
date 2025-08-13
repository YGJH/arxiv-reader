#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan a folder for unplayable .mp4 files and remove/move them safely.
Requires: ffprobe/ffmpeg (from FFmpeg) for best results.
Author: ChatGPT
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

def run_ffprobe(file: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Quick structural check using ffprobe.
    Returns (ok, info). ok=False means the file looks bad/unreadable.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(file),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout or "{}")
        streams = data.get("streams", [])
        fmt = data.get("format", {}) or {}
        duration_val = fmt.get("duration")
        try:
            duration = float(duration_val) if duration_val not in (None, "N/A", "") else 0.0
        except Exception:
            duration = 0.0
        has_av = any(s.get("codec_type") in {"video", "audio"} for s in streams)
        ok = bool(has_av) and duration >= 0.5  # accept very short clips only if >= 0.5s
        return ok, {"duration": duration, "has_av": has_av, "streams": len(streams)}
    except subprocess.CalledProcessError as e:
        # ffprobe failed (file likely corrupt or not a real media file)
        return False, {"error": "ffprobe_failed", "detail": e.stderr.strip()[:300] if e.stderr else ""}
    except Exception as e:
        return False, {"error": "ffprobe_exception", "detail": str(e)[:300]}

def deep_decode(file: Path, seconds: int) -> Tuple[bool, Dict[str, Any]]:
    """
    Deep check: try decoding the first N seconds with ffmpeg and surface decode errors.
    Returns (ok, info). ok=False if decode errors occur.
    """
    try:
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-xerror",              # treat decode warnings as errors
            "-t", str(seconds),
            "-i", str(file),
            "-f", "null",
            "-"                      # discard output
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ok = (res.returncode == 0)
        tail = (res.stderr.decode(errors="ignore") if res.stderr else "")[-500:]
        return ok, {"returncode": res.returncode, "stderr_tail": tail}
    except FileNotFoundError:
        return False, {"error": "ffmpeg_not_found"}
    except Exception as e:
        return False, {"error": "ffmpeg_exception", "detail": str(e)[:300]}

def is_bad(file: Path, args) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Heuristic:
      - If ffprobe not present: we can't be confident. Mark as bad only if deep mode fails.
      - If ffprobe present: bad if probe fails, or no A/V streams, or duration < min_duration.
      - If --deep: also require ffmpeg decode to pass for the first --seconds seconds.
    """
    info_all: Dict[str, Any] = {}
    # ffprobe check
    if shutil.which("ffprobe"):
        ok_probe, info_probe = run_ffprobe(file)
        info_all.update({"probe": info_probe})
        if not ok_probe:
            return True, "probe_failed_or_invalid", info_all
        if info_probe.get("duration", 0.0) < args.min_duration:
            return True, "too_short", info_all
    else:
        info_all["probe"] = {"warning": "ffprobe_not_found"}

    # deep decode (optional but recommended)
    if args.deep:
        if not shutil.which("ffmpeg"):
            return True, "deep_required_but_ffmpeg_missing", info_all
        ok_decode, info_decode = deep_decode(file, args.seconds)
        info_all["decode"] = info_decode
        if not ok_decode:
            return True, "decode_error", info_all

    return False, "ok", info_all

def main():
    ap = argparse.ArgumentParser(
        description="Scan a folder for unplayable .mp4 files and delete/move them. "
                    "By default it's a dry-run; use --delete to actually remove, or --move-to-trash to move."
    )
    ap.add_argument("folder", help="Target folder to scan")
    ap.add_argument("--pattern", default="*.mp4", help="Glob pattern (default: *.mp4)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--deep", action="store_true", help="Decode the first N seconds to catch silent corruption")
    ap.add_argument("--seconds", type=int, default=8, help="Seconds to decode in --deep mode (default: 8)")
    ap.add_argument("--min-duration", type=float, default=0.5, help="Minimum duration in seconds to consider valid (default: 0.5)")
    ap.add_argument("--delete", action="store_true", help="Permanently delete bad files (dangerous)")
    ap.add_argument("--move-to-trash", action="store_true", help="Move bad files into a .trash folder under the root (safer)")
    ap.add_argument("--verbose", action="store_true", help="Print detailed info for each file")
    args = ap.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists():
        print(f"[!] Folder not found: {root}", file=sys.stderr)
        sys.exit(2)

    files = list(root.rglob(args.pattern) if args.recursive else root.glob(args.pattern))
    total = len(files)
    print(f"Scanning {total} file(s) in: {root}")
    if not files:
        return

    bad = []
    for f in files:
        isbad, reason, info = is_bad(f, args)
        if isbad:
            bad.append((f, reason, info))
        if args.verbose:
            print(f"[{'BAD' if isbad else 'OK '}] {f} :: {reason} :: {info}")

    print(f"\nSummary: checked {total} file(s), found {len(bad)} bad file(s).")

    # handle removal
    if bad:
        trash_dir = root / ".trash"
        for f, reason, info in bad:
            if args.delete:
                try:
                    f.unlink()
                    print(f"DELETED: {f} [{reason}]")
                except Exception as e:
                    print(f"[!] FAILED to delete {f}: {e}", file=sys.stderr)
            elif args.move_to_trash:
                try:
                    trash_dir.mkdir(exist_ok=True)
                    dest = trash_dir / f.name
                    # ensure unique name in trash
                    if dest.exists():
                        base = dest.stem
                        suf = dest.suffix
                        i = 1
                        while dest.exists():
                            dest = trash_dir / f"{base}.{i}{suf}"
                            i += 1
                    f.rename(dest)
                    print(f"MOVED -> {dest} (from {f}) [{reason}]")
                except Exception as e:
                    print(f"[!] FAILED to move {f} to trash: {e}", file=sys.stderr)
            else:
                print(f"DRY-RUN: would remove {f} [{reason}]")

    # final hints
    if not shutil.which("ffprobe"):
        print("[!] Hint: ffprobe not found; install FFmpeg for more reliable checks.", file=sys.stderr)
    if args.deep and not shutil.which("ffmpeg"):
        print("[!] Hint: ffmpeg not found; deep decode not available.", file=sys.stderr)

if __name__ == "__main__":
    main()
