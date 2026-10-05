"""Turn a carousel that already performs into a reusable template.

The slide text is read with local OCR (RapidOCR, runs on CPU, nothing leaves
the server). From the OCR boxes we also measure the visual format: where the
text sits, how big it is, and how dark the photos are. The configured LLM then
(optionally) names the hook formula and photo vibe. Only the *format* is kept:
the source photos are never reused in generated slides.
"""

from __future__ import annotations

import io
import json
import os
import re
import statistics
import threading
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import requests
from loguru import logger
from PIL import Image, ImageOps

from app.config import config
from app.models.slideshow import (
    CarouselTemplate,
    SlideAspect,
    SlideStyle,
    TemplateSlide,
    TemplateVisual,
)
from app.services import llm
from app.services.scrapecreators import CarouselPost
from app.utils import utils

MAX_SOURCE_IMAGE_BYTES = 15 * 1024 * 1024
_NUMBERED_RE = re.compile(
    r"^\s*(?:(?:rule|step|tip|day|hack|sign|reason|no\.?|#)\s*)?(\d{1,2})\s*[.)\-:]\s*\S",
    re.IGNORECASE,
)
_CTA_RE = re.compile(
    r"\b(save (this|it|for)|follow|share (this|with)|comment|part 2|link in bio|send this|tag (a|someone))\b",
    re.IGNORECASE,
)
_PLUG_RE = re.compile(
    r"(\b[\w-]+\.(com|app|ai|co|io|net|org)\b|\blink in (my )?bio\b|\bapp called\b|\bdownload(ed)?\b|\bi use(d)?\b|\busing [A-Z])",
    re.IGNORECASE,
)
_WATERMARK_RE = re.compile(r"^(@\S+|tiktok)$", re.IGNORECASE)

_ocr_engine = None
_ocr_lock = threading.Lock()


# --------------------------------------------------------------------------- storage


def templates_dir(sub: str = "") -> str:
    d = utils.storage_dir("templates", create=True)
    return os.path.join(d, sub) if sub else d


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "template"


def save_template(template: CarouselTemplate) -> CarouselTemplate:
    with open(templates_dir(f"{template.id}.json"), "w", encoding="utf-8") as f:
        f.write(template.model_dump_json(indent=2))
    return template


def load_template(template_id: str) -> CarouselTemplate:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", template_id or ""):
        raise ValueError("invalid template id")
    path = templates_dir(f"{template_id}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"template not found: {template_id}")
    with open(path, encoding="utf-8") as f:
        return CarouselTemplate.model_validate_json(f.read())


def list_templates() -> List[CarouselTemplate]:
    out = []
    for name in sorted(os.listdir(templates_dir())):
        if name.endswith(".json"):
            try:
                out.append(load_template(name[:-5]))
            except Exception as e:
                logger.warning(f"skipping unreadable template {name}: {e}")
    return sorted(out, key=lambda t: t.created_at, reverse=True)


def delete_template(template_id: str) -> None:
    load_template(template_id)  # validates the id
    os.remove(templates_dir(f"{template_id}.json"))
    thumbs = templates_dir(template_id)
    if os.path.isdir(thumbs):
        for name in os.listdir(thumbs):
            os.remove(os.path.join(thumbs, name))
        os.rmdir(thumbs)


def template_thumbnails(template_id: str) -> List[str]:
    d = templates_dir(template_id)
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.endswith(".jpg")]


# --------------------------------------------------------------------------- OCR


def _engine():
    global _ocr_engine
    with _ocr_lock:
        if _ocr_engine is None:
            from rapidocr import RapidOCR

            _ocr_engine = RapidOCR(params={"Global.log_level": "warning"})
        return _ocr_engine


class OcrLine:
    __slots__ = ("text", "x0", "y0", "x1", "y1")

    def __init__(self, text: str, x0: float, y0: float, x1: float, y1: float):
        self.text, self.x0, self.y0, self.x1, self.y1 = text, x0, y0, x1, y1

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


def ocr_lines(image: Image.Image, min_score: float = 0.6) -> List[OcrLine]:
    result = _engine()(np.array(image.convert("RGB")))
    if result is None or result.boxes is None:
        return []
    lines = []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        text = (text or "").strip()
        if score < min_score or not text or _WATERMARK_RE.match(text):
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        lines.append(OcrLine(text, min(xs), min(ys), max(xs), max(ys)))
    return sorted(lines, key=lambda line: (line.y0, line.x0))


def group_paragraphs(lines: List[OcrLine]) -> List[List[OcrLine]]:
    """Split lines into paragraphs where the vertical gap is larger than usual."""
    if not lines:
        return []
    median_h = statistics.median(line.h for line in lines)
    paragraphs = [[lines[0]]]
    for prev, line in zip(lines, lines[1:]):
        gap = line.y0 - prev.y1
        if gap > 0.45 * median_h:
            paragraphs.append([line])
        else:
            paragraphs[-1].append(line)
    return paragraphs


def _join(lines: List[OcrLine]) -> str:
    return re.sub(r"\s+", " ", " ".join(line.text for line in lines)).strip()


def slide_text(lines: List[OcrLine]) -> tuple[str, str]:
    paragraphs = group_paragraphs(lines)
    if not paragraphs:
        return "", ""
    heading = _join(paragraphs[0])
    body = " ".join(_join(p) for p in paragraphs[1:])
    return heading, body


def classify(index: int, total: int, heading: str, body: str) -> str:
    text = f"{heading} {body}".strip()
    if not text:
        return "photo"
    if index == 0:
        return "hook"
    if _PLUG_RE.search(text):
        return "plug"
    if index == total - 1 and _CTA_RE.search(text):
        return "cta"
    if _NUMBERED_RE.match(heading):
        return "item"
    return "text"


def _luminance(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def closest_aspect(width: int, height: int) -> SlideAspect:
    ratio = width / max(height, 1)
    options = {SlideAspect.portrait_9_16: 9 / 16, SlideAspect.portrait_4_5: 4 / 5, SlideAspect.square: 1.0}
    return min(options, key=lambda a: abs(options[a] - ratio))


def measure_visual(images: List[Image.Image], slides_lines: List[List[OcrLine]]) -> TemplateVisual:
    text_ys, font_ratios, offsets, bg_lums, box_lums = [], [], [], [], []
    for image, lines in zip(images, slides_lines):
        w, h = image.size
        lum = _luminance(image)
        mask = np.ones_like(lum, dtype=bool)
        for line in lines:
            y0, y1 = max(0, int(line.y0)), min(h, int(line.y1))
            x0, x1 = max(0, int(line.x0)), min(w, int(line.x1))
            mask[y0:y1, x0:x1] = False
            if y1 > y0 and x1 > x0:
                box_lums.append(float(lum[y0:y1, x0:x1].mean()))
        bg_lums.append(float(lum[mask].mean()) if mask.any() else float(lum.mean()))
        if lines:
            top, bottom = min(line.y0 for line in lines), max(line.y1 for line in lines)
            text_ys.append(((top + bottom) / 2) / h)
            # OCR boxes add ~30% padding around the glyphs.
            font_ratios.append(statistics.median(line.h for line in lines) * 0.68 / w)
            offsets.extend(abs(line.cx - w / 2) / w for line in lines)

    bg = statistics.mean(bg_lums) if bg_lums else 0.4
    # Bright, uniform boxes behind the text mean TikTok's "highlight" text style.
    boxed = bool(box_lums) and statistics.median(box_lums) > 0.72 and bg < 0.65
    return TemplateVisual(
        style=SlideStyle.highlight if boxed else SlideStyle.native,
        text_y=min(0.85, max(0.15, statistics.median(text_ys))) if text_ys else 0.42,
        font_ratio=min(0.1, max(0.025, statistics.median(font_ratios))) if font_ratios else 0.046,
        align="center" if not offsets or statistics.median(offsets) < 0.06 else "left",
        # Dark source photos -> darken ours as well so white text stays legible.
        overlay=round(0.4 if bg < 0.3 else 0.28 if bg < 0.5 else 0.18, 2),
        bg_luminance=round(bg, 3),
    )


# --------------------------------------------------------------------------- LLM notes


FORMAT_PROMPT = """
You are a social-media strategist. Below is the slide-by-slide text (read by OCR) of a TikTok
photo carousel that got {plays:,} views, {saves:,} saves and {shares:,} shares.
Explain the reusable FORMAT, not the topic.

Respond ONLY with one minified JSON object:
{{"name":"3-6 word template name","hook_pattern":"the hook formula with [placeholders], e.g. 'Some weird [topic] hacks from my [authority] (that actually work)'","format_notes":"2-4 sentences: slide structure, length per slide, tone, why it gets saved","photo_vibe":"2-4 words describing the background photos to search for, e.g. 'dark moody aesthetic'"}}

Caption: {caption}
Background brightness (0 dark - 1 bright): {bg:.2f}

Slides:
{slides}
""".strip()


def describe_format(template: CarouselTemplate) -> dict:
    slides = "\n".join(
        f"{i + 1}. [{s.role}] {s.heading}" + (f" // {s.body}" if s.body else "")
        for i, s in enumerate(template.slides)
    )
    prompt = FORMAT_PROMPT.format(
        plays=template.plays,
        saves=template.saves,
        shares=template.shares,
        caption=template.source_caption[:300],
        bg=template.visual.bg_luminance,
        slides=slides,
    )
    response = llm._generate_response(prompt)
    if not isinstance(response, str) or response.startswith("Error: "):
        raise ValueError(response)
    text = llm._strip_code_fence(response)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(match.group() if match else text)
    return {k: str(data.get(k) or "").strip()[:400] for k in ("name", "hook_pattern", "format_notes", "photo_vibe")}


def heuristic_notes(template: CarouselTemplate) -> dict:
    roles = [s.role for s in template.slides]
    items = roles.count("item")
    words = [s.words for s in template.slides if s.role in ("item", "text", "plug")]
    avg = round(statistics.mean(words)) if words else 0
    hook = template.slides[0].heading if template.slides else ""
    notes = f"{len(roles)} slides: hook, "
    notes += f"{items} numbered tips" if items else f"{len(roles) - 1} text slides"
    notes += f" (~{avg} words each: short heading + explanation)" if avg else ""
    if template.plug_slide is not None:
        notes += f", product mention on slide {template.plug_slide + 1}"
    if "cta" in roles:
        notes += ", ends with a call to action"
    kind = f"{items} tips" if items else f"{len(roles)} slides"
    if template.plug_slide is not None:
        kind += " + plug"
    author = f"@{template.source_author} · " if template.source_author else ""
    return {
        "name": f"{author}{kind}",
        "hook_pattern": hook,
        "format_notes": notes + ".",
        "photo_vibe": "dark moody aesthetic" if template.visual.bg_luminance < 0.35 else "aesthetic lifestyle",
    }


# --------------------------------------------------------------------------- pipeline


def _download(url: str) -> Optional[Image.Image]:
    try:
        with requests.get(url, proxies=config.proxy, timeout=(15, 60), stream=True) as r:
            r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_content(64 * 1024):
                buf.write(chunk)
                if buf.tell() > MAX_SOURCE_IMAGE_BYTES:
                    raise ValueError("image too large")
        return ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue()))).convert("RGB")
    except Exception as e:
        logger.warning(f"failed to download carousel slide: {type(e).__name__}: {e}")
        return None


def build_template(
    post: CarouselPost,
    images: Optional[List[Image.Image]] = None,
    use_llm: bool = True,
    name: str = "",
) -> tuple[CarouselTemplate, List[Image.Image]]:
    """Analyse a carousel and return an (unsaved) template plus the slides it read."""
    if images is None:
        images = [img for img in (_download(u) for u in post.image_urls) if img is not None]
    if not images:
        raise ValueError("could not download any slides of this carousel")

    slides_lines = [ocr_lines(img) for img in images]
    total = len(images)
    slides = []
    for i, lines in enumerate(slides_lines):
        heading, body = slide_text(lines)
        slides.append(TemplateSlide(role=classify(i, total, heading, body), heading=heading, body=body))
    plug = next((i for i, s in enumerate(slides) if s.role == "plug"), None)

    stamp = datetime.now(timezone.utc)
    template = CarouselTemplate(
        id=f"{stamp.strftime('%Y%m%d%H%M%S')}-{_slugify(post.author or post.id)}",
        name=name,
        created_at=stamp.isoformat(timespec="seconds"),
        source_url=post.url,
        source_author=post.author,
        source_caption=post.desc,
        plays=post.plays,
        likes=post.likes,
        saves=post.saves,
        shares=post.shares,
        comments=post.comments,
        aspect=closest_aspect(*images[0].size),
        slides=slides,
        plug_slide=plug,
        visual=measure_visual(images, slides_lines),
    )

    notes = heuristic_notes(template)
    if use_llm:
        try:
            notes.update({k: v for k, v in describe_format(template).items() if v})
        except Exception as e:
            logger.warning(f"LLM format description failed, using heuristics: {e}")
    template.name = name or notes["name"]
    template.hook_pattern = notes["hook_pattern"]
    template.format_notes = notes["format_notes"]
    template.photo_vibe = notes["photo_vibe"]
    return template, images


def create_template(post: CarouselPost, use_llm: bool = True, name: str = "") -> CarouselTemplate:
    """Analyse, save and keep small reference thumbnails of the source slides."""
    template, images = build_template(post, use_llm=use_llm, name=name)
    thumbs = templates_dir(template.id)
    os.makedirs(thumbs, exist_ok=True)
    for i, image in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((360, 640))
        thumb.save(os.path.join(thumbs, f"slide-{i + 1:02d}.jpg"), "JPEG", quality=80)
    save_template(template)
    logger.success(f"template saved: {template.id} ({template.name})")
    return template
