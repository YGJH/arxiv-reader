#!/usr/bin/env python3
import argparse, os, sys, itertools, math
import cv2
from PIL import Image
import imagehash
from collections import defaultdict
import subprocess, json

# 抑制 OpenCV 日誌
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

def get_duration(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if fps > 0 and frames > 0:
        return frames / fps
    return None

def sample_timestamps(duration, n=8):
    # 均勻抽樣，避開開頭/結尾
    if not duration or duration <= 0:
        return [0.0]
    return [duration * (i + 1) / (n + 1) for i in range(n)]

def grab_frame_at(cap, t_sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    # BGR -> RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def phash_image(arr):
    img = Image.fromarray(arr)
    return imagehash.phash(img)  # 64-bit

def video_signature(path, samples=8, max_fail_ratio=0.5):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    # 先讀取首幀驗證可讀性
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        ok0, f0 = cap.read()
        if not ok0 or f0 is None:
            cap.release()
            return None
        # 重置到開頭
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    except Exception:
        cap.release()
        return None

    duration = get_duration(cap)
    ts = sample_timestamps(duration, samples)
    hashes = []
    wh = None
    fail = 0
    for t in ts:
        frame = grab_frame_at(cap, t)
        if frame is None:
            fail += 1
            continue
        if wh is None:
            h, w = frame.shape[:2]
            wh = (w, h)
        hashes.append(phash_image(frame))
    cap.release()
    if not hashes or fail / max(1, len(ts)) > max_fail_ratio:
        return None
    return {
        "duration": duration or 0.0,
        "hashes": hashes,
        "wh": wh
    }

def phash_distance(sigA, sigB):
    # 以對齊索引取最短長度計算平均距離
    a, b = sigA["hashes"], sigB["hashes"]
    n = min(len(a), len(b))
    if n == 0:
        return math.inf
    dists = [a[i] - b[i] for i in range(n)]
    return sum(dists) / n

def group_by_duration(paths, sigs, tol=0.5):
    buckets = defaultdict(list)
    for p in paths:
        sig = sigs.get(p)
        if not sig:
            continue
        key = round((sig["duration"] or 0.0) / tol)
        buckets[key].append(p)
    return buckets

def pick_keep(paths, sigs):
    # 優先解析度高，其次檔案大，其次字典序
    scored = []
    for p in paths:
        sig = sigs[p]
        w, h = (sig["wh"] or (0, 0))
        size = os.path.getsize(p) if os.path.exists(p) else 0
        scored.append((w*h, size, p))
    scored.sort(reverse=True)
    return scored[0][2], [p for _,_,p in scored[1:]]

# 以 ffprobe 驗證影片是否可讀
def ffprobe_ok(path: str) -> bool:
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type,width,height",
                "-show_entries", "format=duration",
                "-of", "json",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return False
        data = json.loads(proc.stdout or "{}")
        # 有影片流且有寬高，且 duration 合理即可視為可讀
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        dur = None
        try:
            dur = float(fmt.get("duration", 0))
        except Exception:
            dur = 0.0
        has_video = any(s.get("codec_type") == "video" and s.get("width") and s.get("height") for s in streams)
        return has_video and (dur is None or dur >= 0)
    except Exception:
        return False

def find_near_duplicates(root, samples=8, duration_tol=0.5, phash_thr=8, ffprobe_validate=False):
    # 掃描影片檔
    exts = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn.lower())[1] in exts:
                paths.append(os.path.join(dp, fn))

    bads = []  # 記錄不可讀 / 壞檔

    # 建立簽名
    sigs = {}
    for p in paths:
        # 先用 ffprobe 驗證（可選）
        if ffprobe_validate and not ffprobe_ok(p):
            bads.append(p)
            continue
        try:
            sig = video_signature(p, samples=samples)
            if sig:
                sigs[p] = sig
            else:
                bads.append(p)
        except Exception:
            bads.append(p)

    # 依時長分桶後，做近似比對
    dup_groups = []  # 每組為近似重複的一群檔案
    visited = set()
    duration_buckets = group_by_duration(paths, sigs, tol=duration_tol)

    for _, bucket in duration_buckets.items():
        cand = [p for p in bucket if p in sigs]
        for p in cand:
            if p in visited:
                continue
            group = [p]
            for q in cand:
                if q == p or q in visited:
                    continue
                if abs((sigs[p]["duration"] or 0) - (sigs[q]["duration"] or 0)) > duration_tol:
                    continue
                dist = phash_distance(sigs[p], sigs[q])
                if dist <= phash_thr:
                    group.append(q)
            if len(group) > 1:
                for x in group:
                    visited.add(x)
                dup_groups.append(group)

    # 轉成 (keep, dup) 清單
    decisions = []
    for group in dup_groups:
        keep, dups = pick_keep(group, sigs)
        for d in dups:
            decisions.append((keep, d))
    return decisions, bads

def main():
    import argparse
    ap = argparse.ArgumentParser(description="刪除近似重複影片（以感知雜湊判斷，保留高解析度）")
    ap.add_argument("folder")
    ap.add_argument("--delete", action="store_true", help="實際刪除（預設試跑）")
    ap.add_argument("--samples", type=int, default=8, help="每部片抽樣影格數")
    ap.add_argument("--duration-tol", type=float, default=0.5, help="時長容忍秒數")
    ap.add_argument("--phash-thr", type=int, default=8, help="平均 pHash 漢明距離閾值")
    ap.add_argument("--ffprobe-validate", action="store_true", help="使用 ffprobe 驗證影片，無法解析者將跳過")
    args = ap.parse_args()

    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        print(f"資料夾不存在: {root}", file=sys.stderr); sys.exit(1)

    decs, bads = find_near_duplicates(root, samples=args.samples,
                                duration_tol=args.duration_tol,
                                phash_thr=args.phash_thr,
                                ffprobe_validate=args.ffprobe_validate)
    if bads:
        print(f"有 {len(bads)} 個不可讀/壞檔已跳過：")
        for p in bads:
            print(f"SKIP: {p}")

    if not decs:
        print("未找到近似重複影片。"); return

    print(f"發現 {len(decs)} 個可刪項目（保留左側、刪除右側）:")
    for keep, dup in decs:
        print(f"KEEP: {keep}\nDEL : {dup}\n")

    if not args.delete:
        print("試跑模式。加入 --delete 以實際刪除。"); return

    removed = 0
    for _, dup in decs:
        try:
            os.remove(dup); removed += 1
        except Exception as e:
            print(f"刪除失敗: {dup} - {e}", file=sys.stderr)
    print(f"完成，已刪除 {removed} 個檔案。")

if __name__ == "__main__":
    main()