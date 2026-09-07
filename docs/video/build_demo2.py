#!/usr/bin/env python3
"""Stack demo2 camera pairs at normal speed, matching the stamp-paper cards."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


METHODS = (
    ("joint-wam", "block-joint", 60),
    ("distilled-wam-1v2a", "block-1v2a", 31),
    ("streaming-wam", "block-1v2a-rtc", 26),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source_dir, output_dir, selected_method=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, stem, duration in METHODS:
        if selected_method is not None and method != selected_method:
            continue
        sources = [source_dir / f"{stem}-{view}.mp4" for view in (1, 2)]
        output = output_dir / f"{method}.mp4"
        poster = output.with_suffix(".jpg")
        manifest = output.with_suffix(".json")
        for path in (output, poster, manifest):
            if path.exists():
                raise FileExistsError(path)
        for source in sources:
            probe = subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "json", str(source),
            ], text=True)
            if float(json.loads(probe)["format"]["duration"]) < duration:
                raise ValueError(f"Source shorter than requested duration: {source}")
        filters = ";".join([
            f"[{i}:v]setpts=PTS-STARTPTS,scale=640:360,setsar=1,fps=24[v{i}]"
            for i in (0, 1)
        ]) + ";[v0][v1]vstack=inputs=2:shortest=1[out]"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-threads", "2", "-i", str(sources[0]),
            "-threads", "2", "-i", str(sources[1]),
            "-filter_complex_threads", "2", "-filter_complex", filters,
            "-map", "[out]", "-an", "-t", str(duration),
            "-c:v", "libx264", "-threads", "2", "-preset", "medium",
            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-ss", "0", "-i", str(output), "-frames:v", "1", "-q:v", "2", str(poster),
        ], check=True)
        manifest.write_text(json.dumps({
            "layout": "vertical", "duration_seconds": duration, "playback_speed": 1.0,
            "views": [
                {"position": position, "source": f"{source_dir.name}/{source.name}",
                 "source_sha256": sha256(source), "start_seconds": 0.0}
                for position, source in zip(("top", "bottom"), sources)
            ],
            "output_sha256": sha256(output), "poster_sha256": sha256(poster),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Built {method}: {duration}s, 640x720, 24 fps", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--method", choices=[method for method, _, _ in METHODS])
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "assets/real-robot/block")
    args = parser.parse_args()
    build(args.source_dir, args.output_dir, args.method)
