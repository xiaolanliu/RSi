"""Subscribe to the front RGB topic and write an annotated process MP4."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from PIL import Image, ImageDraw, ImageFont
from sensor_msgs.msg import Image as RosImage


def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size=size, index=0)
    except OSError:
        return ImageFont.load_default()


def _text_width(draw, text, font):
    if hasattr(draw, "textlength"):
        return draw.textlength(text, font=font)
    if hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)[0]
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _wrap(draw, text, font, width):
    if not text:
        return []
    lines, current = [], ""
    for char in text.replace("\n", " ").strip():
        trial = current + char
        if _text_width(draw, trial, font) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines[:8]


def annotate(bgr, caption, font_path):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    title_font = _load_font(font_path, max(16, height // 22))
    body_font = _load_font(font_path, max(13, height // 28))
    pad = 10
    box_height = min(height // 3, 160)
    draw.rectangle((0, 0, width, box_height), fill=(12, 12, 10, 195))
    status = (caption.get("status") or "live").upper()
    decision = caption.get("decision") or "idle"
    title = "%s  ·  %s" % (status, decision)
    draw.text((pad, 6), title, font=title_font, fill=(244, 239, 230, 255))
    y = 34
    for line in _wrap(draw, caption.get("reason") or "", body_font, width - 2 * pad):
        draw.text((pad, y), line, font=body_font, fill=(232, 214, 176, 255))
        y += 18
    command = caption.get("command") or ""
    if command:
        compact = " ".join(command.split())
        for line in _wrap(draw, compact, body_font, width - 2 * pad)[:3]:
            draw.text((pad, y), line, font=body_font, fill=(196, 214, 198, 255))
            y += 16
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class Recorder:
    def __init__(self, args):
        self.args = args
        self.overlay_path = Path(args.overlay)
        self.preview_path = Path(args.preview)
        self.stop_path = Path(args.stop)
        self.caption = {}
        self.writer = None
        self.frames = 0
        self.bridge = CvBridge()
        self.font = args.font

    def _caption(self):
        try:
            self.caption = json.loads(self.overlay_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        return self.caption

    def _write_preview(self, bgr):
        tmp = self.preview_path.with_suffix(".tmp.jpg")
        cv2.imwrite(str(tmp), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        tmp.replace(self.preview_path)

    def callback(self, message):
        if self.stop_path.exists():
            rospy.signal_shutdown("stop")
            return
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        labeled = annotate(frame, self._caption(), self.font)
        if self.writer is None:
            height, width = labeled.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.args.output, fourcc, self.args.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError("Could not open process video writer")
        self.writer.write(labeled)
        self.frames += 1
        if self.frames % 2 == 0:
            self._write_preview(labeled)

    def close(self):
        if self.writer is not None:
            self.writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--stop", required=True)
    parser.add_argument("--topic", default="/camera_f/color/image_raw")
    parser.add_argument("--font", required=True)
    parser.add_argument("--fps", type=float, default=15)
    args = parser.parse_args()
    Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
    rospy.init_node("agilex_front_record", anonymous=True, disable_signals=True)
    recorder = Recorder(args)
    rospy.Subscriber(args.topic, RosImage, recorder.callback, queue_size=1)
    rate = rospy.Rate(2)
    try:
        while not rospy.is_shutdown():
            if Path(args.stop).exists():
                break
            rate.sleep()
    finally:
        recorder.close()
        print(json.dumps({"frames": recorder.frames, "output": args.output}))


if __name__ == "__main__":
    main()
