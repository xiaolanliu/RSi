#!/usr/bin/env python3
"""Concatenate real action-video clips and stack three selected camera views.

Offline only. Preserve each clip's recorded timing and exclude the unrecorded
intervals between calls. Inputs are recording.py clip.json / MP4 pairs.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def clips(source, cameras, excluded):
    rows, omitted = [], []
    for file in source.glob("*/clip.json"):
        record = json.loads(file.read_text())
        if record["identity"] in excluded:
            omitted.append({"identity": record["identity"], "reason": "explicit exclusion"})
            continue
        if record["status"] not in {"completed", "failed", "duration_limit"} or record["frames"] < 1:
            raise ValueError(f"Clip has no finalized playable frames: {file}")
        for camera in cameras:
            name = record["cameras"][camera]["file"]
            if Path(name).name != name or not (file.parent / name).is_file():
                raise ValueError(f"Missing/invalid camera video: {file}: {camera}")
        rows.append((record["started_unix"], file.parent, record))
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise ValueError("No finalized action clips found")
    # Preserve identity/FPS. Geometry changes are decoded in separate runs below;
    # MPEG-4 concat demuxing can retain the first run's dimensions and corrupt frames.
    for camera in cameras:
        fingerprints = {(r[2]["cameras"][camera]["serial"], r[2]["fps"]) for r in rows}
        if len(fingerprints) != 1:
            raise ValueError(f"Camera identity or FPS changed: {camera}")
        if any(not isinstance(r[2]["cameras"][camera][key], int) or
               r[2]["cameras"][camera][key] < 2 or r[2]["cameras"][camera][key] % 2
               for r in rows for key in ("width", "height")):
            raise ValueError(f"Positive even camera dimensions required: {camera}")
    return rows, omitted


def concat_line(path):
    value = str(path.resolve())
    if "\n" in value or "\r" in value:
        raise ValueError("Newlines in video paths are unsupported")
    return "file '" + value.replace("'", "'\\''") + "'"


def stage_header(rows, output, width, ffmpeg, font_file=None):
    """Render public stage summaries above, without modifying camera footage."""
    if not any(record.get("stage") for _, _, record in rows):
        return None
    from PIL import Image, ImageDraw, ImageFont

    candidates = [font_file] if font_file else [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    font_path = next((str(p) for p in candidates if p and Path(p).is_file()), None)
    if not font_path:
        raise ValueError("Stage summaries require a CJK-capable font; use --font")
    padding = max(12, round(width / 40))
    body_size = max(14, round(width / 40))
    title_size = max(18, round(width / 32))
    body = ImageFont.truetype(font_path, body_size)
    title = ImageFont.truetype(font_path, title_size)
    line_height = round(body_size * 1.4)

    def wrap(text, font):
        lines, line = [], ""
        for char in text:
            if char == "\n" or (line and font.getlength(line + char) > width - 2 * padding):
                lines.append(line)
                line = "" if char == "\n" else char
            else:
                line += char
        return lines + ([line] if line else [])

    cards = []
    for index, (_, _, record) in enumerate(rows):
        stage = record.get("stage")
        if stage:
            if set(stage) != {"title", "action", "reason"} or not all(isinstance(v, str) for v in stage.values()):
                raise ValueError("Malformed stage summary in clip metadata")
            heading = f"ASTRA  |  {index + 1:02d}  {stage['title']}"
            lines = wrap("行动：" + stage["action"], body) + wrap("依据：" + stage["reason"], body)
        else:
            heading, lines = f"ASTRA  |  {index + 1:02d}", ["本阶段未附行动摘要"]
        cards.append((wrap(heading, title), lines, record["frames"]))
    header_height = max(2 * padding + len(head) * (title_size + 8) + 10 + len(lines) * line_height
                        for head, lines, _ in cards)
    header_height += header_height % 2
    card_dir = output / "stage-cards"
    card_dir.mkdir()
    header_file = output / "stage-header.mp4"
    fps = rows[0][2]["fps"]
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{header_height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(header_file)]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as process:
        try:
            for index, (head, lines, frames) in enumerate(cards):
                card = Image.new("RGB", (width, header_height), "#142232")
                draw = ImageDraw.Draw(card)
                y = padding
                for line in head:
                    draw.text((padding, y), line, font=title, fill="#81d4f2")
                    y += title_size + 8
                y += 10
                for line in lines:
                    draw.text((padding, y), line, font=body, fill="#f1f5f8")
                    y += line_height
                card.save(card_dir / f"stage-{index + 1:03d}.png")
                data = card.tobytes()
                for _ in range(frames):
                    process.stdin.write(data)
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError("Stage header encoding failed")
    return dict(file=header_file.name, height=header_height, font=font_path,
                meaning="Public action and rationale summaries written before execution")


def export(source, output, cameras, excluded, ffmpeg, width=960, encoder="libx264", font_file=None):
    rows, omitted = clips(source, cameras, excluded)
    output.mkdir(parents=True, exist_ok=False)
    if encoder == "libx264":
        codec = ["-c:v", encoder, "-preset", "medium", "-crf", "20", "-threads", "2"]
    elif encoder == "h264_videotoolbox":
        codec = ["-c:v", encoder, "-b:v", "6M"]
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")
    codec += ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-nostdin", "-n"]
    header = stage_header(rows, output, width, ffmpeg, font_file)
    canvases = {camera: {key: max(record["cameras"][camera][key] for _, _, record in rows)
                         for key in ("width", "height")} for camera in cameras}
    with tempfile.TemporaryDirectory(prefix="agilex-action-export-") as temporary:
        for index, camera in enumerate(cameras):
            destination = output / f"{camera}.mp4"
            canvas_width, canvas_height = canvases[camera]["width"], canvases[camera]["height"]
            normalize = (f"scale={canvas_width}:{canvas_height}:force_original_aspect_ratio=decrease:"
                         f"force_divisible_by=2,pad={canvas_width}:{canvas_height}:"
                         "(ow-iw)/2:(oh-ih)/2:eval=frame,setsar=1")
            groups = []
            for row in rows:
                profile = tuple(row[2]["cameras"][camera][key] for key in ("width", "height"))
                if not groups or groups[-1][0] != profile:
                    groups.append((profile, []))
                groups[-1][1].append(row)
            normalized = []
            for group_index, (_, group) in enumerate(groups):
                listing = Path(temporary) / f"{index}-{group_index}.ffconcat"
                listing.write_text("ffconcat version 1.0\n" + "\n".join(
                    concat_line(directory / record["cameras"][camera]["file"])
                    for _, directory, record in group) + "\n")
                encoded = destination if len(groups) == 1 else Path(temporary) / f"{index}-{group_index}.mp4"
                subprocess.run(common + ["-f", "concat", "-safe", "0", "-i", str(listing),
                    "-an", "-vf", normalize] + codec + [str(encoded)], check=True)
                normalized.append(encoded)
            if len(normalized) > 1:
                listing = Path(temporary) / f"{index}-normalized.ffconcat"
                listing.write_text("ffconcat version 1.0\n" + "\n".join(
                    concat_line(path) for path in normalized) + "\n")
                subprocess.run(common + ["-f", "concat", "-safe", "0", "-i", str(listing),
                    "-map", "0:v:0", "-an", "-c:v", "copy", "-movflags", "+faststart",
                    str(destination)], check=True)
            print(f"Encoded {camera}", flush=True)
        height = width * 9 // 16
        filters = [f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]" for i in range(3)]
        inputs = [arg for camera in cameras for arg in ("-i", str(output / f"{camera}.mp4"))]
        if header:
            inputs += ["-i", str(output / header["file"])]
            filters += ["[3:v]setsar=1[header]", "[header][v0][v1][v2]vstack=inputs=4[out]"]
        else:
            filters.append("[v0][v1][v2]vstack=inputs=3[out]")
        subprocess.run(common + inputs + ["-filter_complex", ";".join(filters),
            "-map", "[out]", "-an"] + codec + [str(output / "stacked.mp4")], check=True)
    cursor, timeline = 0.0, []
    for _, directory, record in rows:
        duration = record["frames"] / record["fps"]
        timeline.append(dict(identity=record["identity"], source=str(directory.resolve()),
            playback_start_s=cursor, playback_seconds=duration, status=record["status"],
            started_unix=record["started_unix"], finished_unix=record["finished_unix"],
            padded_frames=record["padded_frames"], repeated_samples=record["repeated_samples"],
            max_sample_gap_s=record["max_sample_gap_s"], stage=record.get("stage"),
            source_cameras=record["cameras"]))
        cursor += duration
    manifest = dict(kind="real action clips concatenated without time between calls",
        stack_order_top_to_bottom=list(cameras), cross_camera_hardware_sync=False,
        playback_seconds=cursor, clips=timeline, excluded=omitted, encoder=encoder, stage_header=header,
        output_camera_canvases=canvases,
        geometry="Contiguous runs of each native geometry are decoded separately, normalized to a common "
                 "canvas and joined as H.264. Centered black padding preserves the full image and aspect ratio. "
                 "Native profiles are retained per clip; decoder errors abort export.",
        timing="Each clip retains recorded CFR timing; late samples may hold a prior frame. "
               "Phase planning, settling and observation remain; model reasoning between phases is absent.")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cameras", nargs=3, default=["front", "left_wrist", "right_wrist"])
    parser.add_argument("--exclude-identity", action="append", default=[])
    parser.add_argument("--ffmpeg")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--encoder", choices=["libx264", "h264_videotoolbox"], default="libx264")
    parser.add_argument("--font", type=Path, help="CJK-capable font for optional stage summaries")
    args = parser.parse_args()
    if len(set(args.cameras)) != 3 or any(Path(c).name != c or c in {".", ".."} for c in args.cameras):
        parser.error("Choose three distinct plain camera names")
    if args.width < 320 or args.width % 32:
        parser.error("Width must be >=320 and divisible by 32")
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = export(args.source, args.output, args.cameras, set(args.exclude_identity), ffmpeg, args.width, args.encoder, args.font)
    print(json.dumps({"clips": len(result["clips"]), "seconds": result["playback_seconds"],
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
