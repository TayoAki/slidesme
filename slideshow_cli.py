"""Command-line entry point for generating social-media slideshows.

Examples:
    uv run python slideshow_cli.py "5 money habits that changed my life"
    uv run python slideshow_cli.py "morning routine" --style bold --aspect 9:16 --video
    uv run python slideshow_cli.py "gym tips" --images none --handle mygym
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

root_dir = os.path.dirname(os.path.realpath(__file__))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from app.models.slideshow import (
        MAX_SLIDES,
        MIN_SLIDES,
        SlideAspect,
        SlideImageSource,
        SlideStyle,
    )

    p = argparse.ArgumentParser(
        description="Generate a viral-style photo carousel / slideshow from a topic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("topic", help="What the slideshow is about")
    p.add_argument("--slides", type=int, default=7, help=f"Number of slides ({MIN_SLIDES}-{MAX_SLIDES})")
    p.add_argument("--language", default="", help="Output language, e.g. en, es, zh-CN (default: topic language)")
    p.add_argument("--tone", default="", help="Extra copywriting instructions, e.g. 'funny, gen-z voice'")
    p.add_argument("--script-file", help="JSON file with {slides:[{heading,body,image_query}],caption,hashtags}; skips the LLM")
    p.add_argument("--style", choices=[s.value for s in SlideStyle], default=SlideStyle.tiktok.value)
    p.add_argument("--aspect", choices=[a.value for a in SlideAspect], default=SlideAspect.portrait_4_5.value)
    p.add_argument("--images", choices=[s.value for s in SlideImageSource], default=SlideImageSource.pexels.value, help="Background photo source")
    p.add_argument("--local-image", action="append", default=[], help="Local background image (repeatable, used in slide order)")
    p.add_argument("--font", default="", help="Font file name inside resource/fonts")
    p.add_argument("--accent", default="#FFD400", help="Accent colour for the 'bold' style")
    p.add_argument("--handle", default="", help="@handle watermark on each slide")
    p.add_argument("--no-numbers", action="store_true", help="Hide the 2/7 slide counter")
    p.add_argument("--video", action="store_true", help="Also export an MP4 slideshow with music")
    p.add_argument("--seconds", type=float, default=3.0, help="Seconds per slide in the MP4")
    p.add_argument("--no-bgm", action="store_true", help="MP4 without background music")
    p.add_argument("--task-id", default="", help="Output folder name under storage/tasks")
    args = p.parse_args(argv)
    if not MIN_SLIDES <= args.slides <= MAX_SLIDES:
        p.error(f"--slides must be between {MIN_SLIDES} and {MAX_SLIDES}")
    return args


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    from app.models.slideshow import SlideshowParams, SlideshowScript
    from app.services import slideshow
    from app.utils import utils

    script = None
    if args.script_file:
        with open(args.script_file, encoding="utf-8") as f:
            script = SlideshowScript.model_validate(json.load(f))

    params = SlideshowParams(
        topic=args.topic,
        language=args.language,
        slide_count=args.slides,
        tone=args.tone,
        script=script,
        aspect=args.aspect,
        style=args.style,
        image_source=args.images,
        font_name=args.font,
        accent_color=args.accent,
        handle=args.handle,
        show_slide_numbers=not args.no_numbers,
        export_video=args.video,
        seconds_per_slide=args.seconds,
        bgm_enabled=not args.no_bgm,
    )
    task_id = args.task_id or utils.get_uuid()
    result = slideshow.generate(task_id, params, local_images=[os.path.abspath(p) for p in args.local_image])
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
