"""Generate viral-style social-media slideshows (carousels).

Pipeline: LLM writes a hook -> value slides -> call-to-action script, a stock
photo is fetched per slide, each slide is rendered to PNG with Pillow, and the
set is zipped (and optionally stitched into an MP4 with background music).
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import time
import zipfile
from functools import lru_cache
from typing import Callable, List, Optional
from urllib.parse import urlencode

import requests
from loguru import logger
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from app.config import config
from app.models import const
from app.models.slideshow import (
    CarouselTemplate,
    Slide,
    SlideImageSource,
    SlideshowParams,
    SlideshowResult,
    SlideshowScript,
    SlideStyle,
    TemplateVisual,
)
from app.services import llm
from app.services import state as sm
from app.services.material import _get_tls_verify, _redact_request_error, get_api_key
from app.utils import file_security, utils

MAX_IMAGE_DOWNLOAD_BYTES = 25 * 1024 * 1024
# Wikimedia / StockSnap reject anonymous library user agents (HTTP 403).
HTTP_HEADERS = {"User-Agent": "slidesme/1.0 (https://github.com/TayoAki/slidesme; carousel generator)"}
DEFAULT_LATIN_FONT = "BeVietnamPro-Bold.ttf"
# TikTok's own typeface (OFL, variable font) - used by the "native" style.
NATIVE_FONT = "TikTokSans-Variable.ttf"
DEFAULT_CJK_FONT = "MicrosoftYaHeiBold.ttc"
MAX_HEADING_CHARS = 120
MAX_BODY_CHARS = 280

_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
# Bundled fonts have no emoji glyphs, so emoji would render as boxes on slides.
# They are kept in the caption/hashtags, which platforms render natively.
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff"
    "\U0000fe0f\U0000200d\U00002b00-\U00002bff]+"
)

# Backgrounds used when no photo is available (style "minimal" or no API key).
_GRADIENTS = [
    ((15, 32, 39), (44, 83, 100)),
    ((35, 7, 77), (204, 83, 51)),
    ((20, 30, 48), (36, 59, 85)),
    ((66, 39, 90), (115, 75, 109)),
    ((19, 78, 94), (113, 178, 128)),
    ((40, 48, 72), (133, 147, 152)),
]

SLIDESHOW_SYSTEM_PROMPT = """
# Role: Viral Carousel Copywriter

## Goal
Write a {slide_count}-slide photo carousel about the topic below, in the style of
slideshows that go viral on TikTok photo mode and Instagram carousels.

## Rules
1. Slide 1 is the HOOK: a short, curiosity-driven or bold claim that stops the scroll
   (e.g. "7 habits that quietly wreck your sleep", "Nobody tells you this about X").
   Hook "heading" max 12 words; "body" may be empty or a 1-line teaser.
2. Slides 2..{last_value} each deliver ONE concrete, specific idea. Number list items
   in the heading when it is a list ("1. ..."). "heading" max 10 words, "body" max 30
   words, plain conversational language, no filler.
3. Slide {slide_count} is the CTA: tell people to save / share / follow (e.g.
   "Save this for later" + "Follow for part 2"), tied to the topic.
4. "image_query" is a 2-4 word ENGLISH stock-photo search term showing the slide's idea
   visually (concrete objects or scenes, no text, no brand names).
5. Never use emoji inside slides. No hashtags inside slides.
6. "caption" is the post caption: 1-3 punchy sentences ending with a question or call
   to action (emoji allowed). "hashtags" is an array of 5 relevant hashtags with "#".
7. {language_instruction}
8. Respond ONLY with one minified JSON object, no markdown, no commentary:
{{"slides":[{{"heading":"...","body":"...","image_query":"..."}}],"caption":"...","hashtags":["#a"]}}

## Topic
{topic}
""".strip()


# --------------------------------------------------------------------------- script


def build_slideshow_prompt(topic: str, slide_count: int, language: str = "", tone: str = "") -> str:
    if language:
        language_instruction = f'Write "heading", "body" and "caption" in {language}.'
    else:
        language_instruction = 'Write "heading", "body" and "caption" in the language of the topic.'
    prompt = SLIDESHOW_SYSTEM_PROMPT.format(
        slide_count=slide_count,
        last_value=slide_count - 1,
        language_instruction=language_instruction,
        topic=topic.strip(),
    )
    if tone.strip():
        prompt += f"\n\n## Tone / extra requirements\n{tone.strip()}"
    return prompt


TEMPLATE_PROMPT = """
# Role: Viral Carousel Copywriter

## Goal
Write a NEW TikTok / Instagram photo carousel about the topic below that copies the FORMAT of a
proven carousel ({plays:,} views, {saves:,} saves, {shares:,} shares). Copy its structure,
rhythm, length and voice - never its sentences or its topic-specific facts.

## Proven format to copy
- Template: {name}
- Hook formula: {hook_pattern}
- Format notes: {format_notes}
- Exactly {slide_count} slides, roles in order: {roles}
- Words per slide in the original: {word_counts}

## Original slides (for structure only)
{examples}

## Rules
1. Same number of slides and the same role on each slide as listed above.
2. Mirror the hook formula with the new topic; keep its curiosity and tone.
3. Each "item" slide: "heading" is a numbered, specific tip in the same style; "body" is a
   short explanation of similar length. Write like a real person, not a brand.
4. {plug_rule}
5. "image_query": 2-4 word ENGLISH photo search term for a real, candid, phone-style photo that
   fits the slide (concrete objects or scenes; no text, no logos, no people's faces required).
6. No emoji or hashtags inside slides. {language_instruction}
7. "caption": short, casual post caption in the same voice as the original caption; "hashtags":
   5 relevant hashtags.
8. Respond ONLY with one minified JSON object:
{{"slides":[{{"heading":"...","body":"...","image_query":"..."}}],"caption":"...","hashtags":["#a"]}}

## Original caption
{caption}

## New topic
{topic}
""".strip()


def build_template_prompt(
    template, topic: str, language: str = "", tone: str = "", product: str = ""
) -> str:
    examples = "\n".join(
        f"{i + 1}. [{s.role}] heading: {s.heading!r} | body: {s.body!r}"
        for i, s in enumerate(template.slides)
    )
    if template.plug_slide is not None and product.strip():
        plug_rule = (
            f"On slide {template.plug_slide + 1}, mention {product.strip()} casually and "
            "personally, the way the original slide mentions its product (one sentence inside a "
            "genuinely useful tip, never salesy)."
        )
    elif product.strip():
        plug_rule = (
            f"Mention {product.strip()} once, casually and personally, inside the most relevant "
            "tip slide (not the hook, never salesy)."
        )
    else:
        plug_rule = "Do not mention any product, app or website; replace a plug slide with a normal tip."
    if language:
        language_instruction = f"Write all text in {language}."
    else:
        language_instruction = "Write in the language of the topic."
    prompt = TEMPLATE_PROMPT.format(
        plays=template.plays,
        saves=template.saves,
        shares=template.shares,
        name=template.name,
        hook_pattern=template.hook_pattern or template.slides[0].heading,
        format_notes=template.format_notes,
        slide_count=template.slide_count,
        roles=", ".join(s.role for s in template.slides),
        word_counts=", ".join(str(s.words) for s in template.slides),
        examples=examples,
        plug_rule=plug_rule,
        language_instruction=language_instruction,
        caption=template.source_caption[:300] or "(none)",
        topic=topic.strip(),
    )
    if tone.strip():
        prompt += f"\n\n## Tone / extra requirements\n{tone.strip()}"
    return prompt


def _clean_text(value, limit: int) -> str:
    text = _EMOJI_RE.sub("", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip()


def parse_slideshow_script(response: str, slide_count: int) -> SlideshowScript:
    text = llm._strip_code_fence(response or "")
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("slideshow response does not contain a JSON object")
        data = json.loads(match.group())
    if not isinstance(data, dict) or not isinstance(data.get("slides"), list):
        raise ValueError("slideshow response is missing a 'slides' array")

    slides = []
    for raw in data["slides"]:
        if not isinstance(raw, dict):
            continue
        slide = Slide(
            heading=_clean_text(raw.get("heading"), MAX_HEADING_CHARS),
            body=_clean_text(raw.get("body"), MAX_BODY_CHARS),
            image_query=_clean_text(raw.get("image_query"), 80),
        )
        if slide.heading or slide.body:
            slides.append(slide)
    if len(slides) < 2:
        raise ValueError(f"slideshow response has too few usable slides: {len(slides)}")
    if len(slides) != slide_count:
        logger.warning(f"llm returned {len(slides)} slides, expected {slide_count}")

    hashtags = []
    raw_tags = data.get("hashtags") or []
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            tag = re.sub(r"\s+", "", str(tag or ""))
            if tag and not tag.startswith("#"):
                tag = "#" + tag
            if len(tag) > 1 and tag not in hashtags:
                hashtags.append(tag)
    return SlideshowScript(
        slides=slides[:slide_count] if len(slides) > slide_count else slides,
        caption=str(data.get("caption") or "").strip()[:2200],
        hashtags=hashtags[:10],
    )


def generate_slideshow_script(
    topic: str,
    slide_count: int = 7,
    language: str = "",
    tone: str = "",
    template=None,
    product: str = "",
) -> SlideshowScript:
    if template is not None and template.slides:
        slide_count = template.slide_count
        prompt = build_template_prompt(template, topic, language, tone, product)
    else:
        prompt = build_slideshow_prompt(topic, slide_count, language, tone)
        if product.strip():
            prompt += (
                f"\n\n## Product\nMention {product.strip()} once, casually and personally, "
                "inside the most relevant tip slide (not the hook, never salesy)."
            )
    logger.info(f"generating slideshow script: topic={topic!r}, slides={slide_count}")
    last_error: Exception | None = None
    for attempt in range(llm._max_retries):
        response = llm._generate_response(prompt)
        if isinstance(response, str) and response.startswith("Error: "):
            # Free/shared providers fail intermittently (rate limits, 403s); retry with backoff.
            last_error = ValueError(response[7:])
            logger.warning(f"llm error (attempt {attempt + 1}): {response[:300]}")
            time.sleep(min(8, 2**attempt))
            continue
        try:
            script = parse_slideshow_script(response, slide_count)
            logger.success(f"slideshow script generated with {len(script.slides)} slides")
            return script
        except Exception as e:
            last_error = e
            logger.warning(f"invalid slideshow script (attempt {attempt + 1}): {e}")
    raise ValueError(f"failed to generate slideshow script: {last_error}")


# --------------------------------------------------------------------------- images


def _orientation(width: int, height: int) -> str:
    if height > width:
        return "portrait"
    if width > height:
        return "landscape"
    return "square"


def search_images_pexels(query: str, width: int, height: int) -> List[str]:
    api_key = get_api_key("pexels_api_keys")
    params = {"query": query, "per_page": 15, "orientation": _orientation(width, height)}
    url = f"https://api.pexels.com/v1/search?{urlencode(params)}"
    try:
        r = requests.get(
            url,
            headers={"Authorization": api_key},
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(15, 30),
        )
        photos = r.json().get("photos") or []
    except Exception as e:
        logger.error(f"pexels photo search failed: {_redact_request_error(e, api_key)}")
        return []
    urls = []
    for photo in photos:
        src = photo.get("src") if isinstance(photo, dict) else None
        if isinstance(src, dict):
            # "large2x" is ~1880px wide; "original" can be 6000px+ and slow to fetch.
            candidate = src.get("large2x") or src.get("original")
            if isinstance(candidate, str) and candidate.startswith("https://"):
                urls.append(candidate)
    return urls


def search_images_pixabay(query: str, width: int, height: int) -> List[str]:
    api_key = get_api_key("pixabay_api_keys")
    orientation = "vertical" if height > width else "horizontal" if width > height else "all"
    params = {
        "key": api_key,
        "q": query,
        "image_type": "photo",
        "orientation": orientation,
        "per_page": 20,
        "safesearch": "true",
    }
    url = f"https://pixabay.com/api/?{urlencode(params)}"
    try:
        r = requests.get(url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(15, 30))
        hits = r.json().get("hits") or []
    except Exception as e:
        logger.error(f"pixabay photo search failed: {_redact_request_error(e, api_key)}")
        return []
    return [
        h["largeImageURL"]
        for h in hits
        if isinstance(h, dict)
        and isinstance(h.get("largeImageURL"), str)
        and h["largeImageURL"].startswith("https://")
    ]


class Photo:
    __slots__ = ("url", "credit")

    def __init__(self, url: str, credit: str = ""):
        self.url, self.credit = url, credit


def search_images_openverse(query: str, width: int, height: int) -> List[Photo]:
    """Real CC-licensed photographs (Wikimedia, Flickr, ...) - no API key needed.

    Only licenses that allow commercial use and modification are requested;
    credits are returned so they can be shipped alongside the slides.
    """
    base = {
        "q": query,
        "license_type": "commercial,modification",
        "mature": "false",
        "category": "photograph",
        "page_size": 20,
    }
    photos: List[Photo] = []
    # Large images first; widen the net if the query is niche.
    for extra in ({"size": "large"}, {}):
        try:
            r = requests.get(
                f"https://api.openverse.org/v1/images/?{urlencode({**base, **extra})}",
                headers=HTTP_HEADERS,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(15, 30),
            )
            results = r.json().get("results") or []
        except Exception as e:
            logger.error(f"openverse photo search failed: {type(e).__name__}: {e}")
            return photos
        for item in results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            w, h = item.get("width") or 0, item.get("height") or 0
            if not isinstance(url, str) or not url.startswith("https://"):
                continue
            if url.split("?")[0].lower().endswith((".svg", ".gif", ".tif", ".tiff")):
                continue
            if w and h and min(w, h) < 700:
                continue
            title = str(item.get("title") or "photo")[:80]
            creator = str(item.get("creator") or "unknown")[:60]
            license_ = f"CC {str(item.get('license') or '').upper()} {item.get('license_version') or ''}".strip()
            if item.get("license") in ("cc0", "pdm"):
                license_ = "public domain"
            landing = item.get("foreign_landing_url") or url
            photos.append(Photo(url, f'"{title}" by {creator} ({license_}) {landing}'))
        if len(photos) >= 4:
            break
    return photos


def download_image(url: str) -> Optional[Image.Image]:
    try:
        with requests.get(
            url,
            headers=HTTP_HEADERS,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(15, 60),
            stream=True,
        ) as r:
            r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_content(64 * 1024):
                buf.write(chunk)
                if buf.tell() > MAX_IMAGE_DOWNLOAD_BYTES:
                    raise ValueError("image exceeds download size limit")
        image = Image.open(io.BytesIO(buf.getvalue()))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as e:
        logger.warning(f"failed to download image: {type(e).__name__}: {e}")
        return None


def _search_photos(query: str, source: SlideImageSource, size: tuple[int, int]) -> List[Photo]:
    if source == SlideImageSource.openverse:
        return search_images_openverse(query, *size)
    if source == SlideImageSource.pexels:
        return [Photo(u, "Photo from Pexels (pexels.com/license)") for u in search_images_pexels(query, *size)]
    return [Photo(u, "Photo from Pixabay (pixabay.com/service/license-summary)") for u in search_images_pixabay(query, *size)]


def fetch_background(
    query: str,
    source: SlideImageSource,
    size: tuple[int, int],
    used_urls: set[str],
    vibe: str = "",
) -> tuple[Optional[Image.Image], str]:
    """Return (photo, credit). Tries "<query> <vibe>" first, then the bare query."""
    if source == SlideImageSource.none or not query:
        return None, ""
    queries = [f"{query} {vibe}".strip(), query] if vibe.strip() else [query]
    for q in dict.fromkeys(queries):
        try:
            photos = _search_photos(q, source, size)
        except ValueError as e:  # API key not configured
            logger.warning(str(e).strip())
            return None, ""
        # Pick from the top results, but never reuse a photo within one slideshow.
        candidates = [p for p in photos[:8] if p.url not in used_urls]
        random.shuffle(candidates)
        for photo in candidates:
            image = download_image(photo.url)
            if image is not None:
                used_urls.add(photo.url)
                return image, photo.credit
    return None, ""


def load_local_background(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        logger.warning(f"failed to open local image {path}: {e}")
        return None


# --------------------------------------------------------------------------- rendering


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def gradient_background(size: tuple[int, int], index: int) -> Image.Image:
    top, bottom = _GRADIENTS[index % len(_GRADIENTS)]
    w, h = size
    column = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(h - 1, 1)
        column.putpixel((0, y), tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return column.resize((w, h))


def resolve_font_path(font_name: str, sample_text: str, style: SlideStyle = SlideStyle.tiktok) -> str:
    if font_name:
        return file_security.resolve_path_within_directory(utils.font_dir(), font_name)
    if _CJK_RE.search(sample_text or ""):
        default = DEFAULT_CJK_FONT
    elif style == SlideStyle.native and os.path.isfile(os.path.join(utils.font_dir(), NATIVE_FONT)):
        default = NATIVE_FONT
    else:
        default = DEFAULT_LATIN_FONT
    return os.path.join(utils.font_dir(), default)


@lru_cache(maxsize=256)
def load_font(path: str, size: int, weight: str = "SemiBold") -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(path, size)
    try:
        # Variable fonts (TikTok Sans) default to their lightest instance.
        if weight.encode() in font.get_variation_names():
            font.set_variation_by_name(weight)
    except OSError:
        pass  # not a variable font
    return font


def _tokenize(text: str) -> List[str]:
    """Split into wrap units: words for spaced scripts, characters for CJK."""
    tokens: List[str] = []
    for word in text.split(" "):
        if _CJK_RE.search(word):
            tokens.extend(list(word))
        else:
            tokens.append(word)
    return [t for t in tokens if t]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    lines: List[str] = []
    current = ""
    for token in _tokenize(text):
        sep = "" if (not current or _CJK_RE.match(token) or _CJK_RE.search(current[-1:])) else " "
        candidate = f"{current}{sep}{token}"
        if not current or font.getlength(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = token
    if current:
        lines.append(current)
    return lines


def _fit_text(
    text: str, font_path: str, max_width: int, max_height: int, start: int, minimum: int
) -> tuple[ImageFont.FreeTypeFont, List[str], int]:
    """Largest font size (and its wrapped lines) that fits the box."""
    size = start
    while True:
        font = load_font(font_path, size)
        lines = wrap_text(text, font, max_width)
        line_h = int(size * 1.22)
        if (len(lines) * line_h <= max_height and all(font.getlength(ln) <= max_width for ln in lines)) or size <= minimum:
            return font, lines, line_h
        size -= 4


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image, size, method=Image.LANCZOS, centering=(0.5, 0.45))


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    line_h: int,
    x: int,
    y: int,
    width: int,
    align: str,
    fill,
    stroke: int = 0,
    stroke_fill=None,
    box_fill=None,
    box_text_fill=None,
) -> int:
    for line in lines:
        line_w = font.getlength(line)
        lx = x + (width - line_w) / 2 if align == "center" else x
        if box_fill is not None:
            pad_x, pad_y = int(font.size * 0.35), int(font.size * 0.12)
            top = y - pad_y + int(font.size * 0.08)
            draw.rounded_rectangle(
                (lx - pad_x, top, lx + line_w + pad_x, top + font.size + 2 * pad_y),
                radius=int(font.size * 0.25),
                fill=box_fill,
            )
            draw.text((lx, y), line, font=font, fill=box_text_fill)
        else:
            draw.text((lx, y), line, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        y += line_h
    return y


def match_mood(image: Image.Image, visual: TemplateVisual) -> Image.Image:
    """Bring a photo to roughly the source carousel's brightness.

    Bright stock photos get darkened (keeps white text legible and matches the
    moody look); already-dark photos are left alone apart from a small contrast lift.
    """
    current = float(np.asarray(image.convert("L"), dtype=np.float32).mean() / 255.0)
    target = min(0.55, max(0.12, visual.bg_luminance + 0.04))
    factor = min(1.0, max(0.3, target / max(current, 1e-3)))
    image = ImageEnhance.Contrast(image).enhance(1.08)
    return ImageEnhance.Brightness(image).enhance(factor)


def _render_native(
    canvas: Image.Image,
    slide: Slide,
    visual: TemplateVisual,
    font_path: str,
) -> Image.Image:
    """TikTok photo-mode look: one centred block, heading / blank line / body,
    same size, white TikTok Sans with a thin dark outline and soft shadow."""
    w, h = canvas.size
    text_w = int(w * 0.8)
    size = max(24, int(visual.font_ratio * w))
    paragraphs = [p for p in (slide.heading, slide.body) if p]
    while True:
        font = load_font(font_path, size)
        line_h = int(size * 1.2)
        wrapped = [wrap_text(p, font, text_w) for p in paragraphs]
        gap = int(size * 0.9)
        block_h = sum(len(lines) * line_h for lines in wrapped) + gap * (len(wrapped) - 1)
        if block_h <= h * 0.62 or size <= 24:
            break
        size -= 2

    top = int(visual.text_y * h - block_h / 2)
    top = max(int(h * 0.08), min(top, int(h * 0.88) - block_h))
    stroke = max(2, round(size * 0.055))

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    text_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd, td = ImageDraw.Draw(shadow), ImageDraw.Draw(text_layer)
    y = top
    for lines in wrapped:
        for line in lines:
            lw = font.getlength(line)
            x = (w - lw) / 2 if visual.align == "center" else (w - text_w) / 2
            sd.text((x, y + size * 0.05), line, font=font, fill=(0, 0, 0, 170))
            td.text((x, y), line, font=font, fill=(255, 255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0, 200))
            y += line_h
        y += gap
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(2, size // 10)))
    out = Image.alpha_composite(canvas.convert("RGBA"), shadow)
    return Image.alpha_composite(out, text_layer).convert("RGB")


def render_slide(
    slide: Slide,
    index: int,
    total: int,
    params: SlideshowParams,
    background: Optional[Image.Image],
    font_path: str,
    visual: Optional[TemplateVisual] = None,
) -> Image.Image:
    size = params.aspect.to_resolution()
    w, h = size
    style = params.style
    if style == SlideStyle.native:
        visual = visual or TemplateVisual()
        if background is None:
            canvas = gradient_background(size, index)
        else:
            canvas = match_mood(_cover(background, size), visual)
        canvas = _render_native(canvas, slide, visual, font_path)
        if params.handle:
            # Native carousels carry no counters or swipe nudges - only an optional handle.
            draw = ImageDraw.Draw(canvas)
            small = load_font(font_path, int(30 * w / 1080))
            handle = params.handle if params.handle.startswith("@") else f"@{params.handle}"
            draw.text((int(w * 0.08), h - int(110 * w / 1080)), handle, font=small, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        return canvas
    accent = _hex_to_rgb(params.accent_color)
    is_hook = index == 0
    margin = int(w * 0.08)
    text_w = w - 2 * margin

    if style == SlideStyle.minimal or background is None:
        canvas = (
            Image.new("RGB", size, (14, 14, 16))
            if style == SlideStyle.minimal
            else gradient_background(size, index)
        )
    else:
        canvas = _cover(background, size)

    if background is not None and style != SlideStyle.minimal:
        if style == SlideStyle.bold:
            # Bottom-heavy gradient so text sits on near-black.
            shade = Image.new("L", (1, h))
            for y in range(h):
                t = y / h
                shade.putpixel((0, y), int(255 * max(0.0, min(1.0, (t - 0.25) / 0.55)) * 0.92))
            black = Image.new("RGB", size, (0, 0, 0))
            canvas = Image.composite(black, canvas, shade.resize(size))
        elif style == SlideStyle.tiktok:
            dark = Image.new("RGB", size, (0, 0, 0))
            canvas = Image.blend(canvas, dark, 0.38 if not is_hook else 0.3)
        else:  # highlight: keep the photo vivid, just soften it a touch
            canvas = canvas.filter(ImageFilter.GaussianBlur(1))

    draw = ImageDraw.Draw(canvas)
    heading = slide.heading
    body = slide.body
    if style == SlideStyle.bold:
        heading = heading.upper()

    # Font sizes scale with width so 1:1, 4:5 and 9:16 look consistent.
    unit = w / 1080
    head_start = int((118 if is_hook else 92) * unit)
    body_start = int(56 * unit)
    head_box_h = int(h * (0.5 if is_hook else 0.34))
    body_box_h = int(h * 0.3)

    head_font, head_lines, head_lh = _fit_text(heading, font_path, text_w, head_box_h, head_start, int(44 * unit)) if heading else (None, [], 0)
    body_font, body_lines, body_lh = _fit_text(body, font_path, text_w, body_box_h, body_start, int(30 * unit)) if body else (None, [], 0)
    gap = int(36 * unit) if head_lines and body_lines else 0
    block_h = len(head_lines) * head_lh + gap + len(body_lines) * body_lh

    if style == SlideStyle.bold:
        align = "left"
        y = h - margin - block_h - int(40 * unit)
        draw.rectangle((margin, y - int(36 * unit), margin + int(120 * unit), y - int(22 * unit)), fill=accent)
    elif style == SlideStyle.minimal:
        align = "left"
        y = (h - block_h) // 2
    else:
        align = "center"
        y = (h - block_h) // 2

    stroke = max(2, int(6 * unit))
    if head_lines:
        if style == SlideStyle.highlight:
            y = _draw_lines(draw, head_lines, head_font, int(head_lh * 1.12), margin, y, text_w, align, None, box_fill=(255, 255, 255), box_text_fill=(0, 0, 0))
        elif style == SlideStyle.minimal:
            y = _draw_lines(draw, head_lines, head_font, head_lh, margin, y, text_w, align, (255, 255, 255))
        elif style == SlideStyle.bold:
            y = _draw_lines(draw, head_lines, head_font, head_lh, margin, y, text_w, align, accent if is_hook else (255, 255, 255))
        else:
            y = _draw_lines(draw, head_lines, head_font, head_lh, margin, y, text_w, align, (255, 255, 255), stroke, (0, 0, 0))
        y += gap
    if body_lines:
        if style == SlideStyle.highlight:
            _draw_lines(draw, body_lines, body_font, int(body_lh * 1.15), margin, y, text_w, align, None, box_fill=(0, 0, 0), box_text_fill=(255, 255, 255))
        elif style == SlideStyle.minimal:
            _draw_lines(draw, body_lines, body_font, body_lh, margin, y, text_w, align, (190, 190, 196))
        elif style == SlideStyle.bold:
            _draw_lines(draw, body_lines, body_font, body_lh, margin, y, text_w, align, (235, 235, 235))
        else:
            _draw_lines(draw, body_lines, body_font, body_lh, margin, y, text_w, align, (255, 255, 255), max(2, int(4 * unit)), (0, 0, 0))

    # Footer: handle watermark + "2/7" counter. The hook gets a swipe nudge instead.
    small = load_font(font_path, int(30 * unit))
    footer_y = h - int(70 * unit)
    footer_stroke = 0 if style == SlideStyle.minimal else max(1, int(3 * unit))
    if params.handle:
        handle = params.handle if params.handle.startswith("@") else f"@{params.handle}"
        draw.text((margin, footer_y), handle, font=small, fill=(255, 255, 255), stroke_width=footer_stroke, stroke_fill=(0, 0, 0))
    if is_hook and total > 1:
        nudge = "swipe for more"
        draw.text((w - margin - small.getlength(nudge), footer_y), nudge, font=small, fill=(255, 255, 255), stroke_width=footer_stroke, stroke_fill=(0, 0, 0))
    elif params.show_slide_numbers:
        counter = f"{index + 1}/{total}"
        draw.text((w - margin - small.getlength(counter), footer_y), counter, font=small, fill=(255, 255, 255), stroke_width=footer_stroke, stroke_fill=(0, 0, 0))
    return canvas


# --------------------------------------------------------------------------- export


def write_zip(image_paths: List[str], caption_text: str, zip_path: str, credits: Optional[List[str]] = None) -> str:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in image_paths:
            zf.write(path, os.path.basename(path))
        zf.writestr("caption.txt", caption_text)
        if credits:
            zf.writestr("credits.txt", "Photo credits:\n" + "\n".join(credits))
    return zip_path


def write_video(image_paths: List[str], params: SlideshowParams, output_path: str) -> str:
    from moviepy import AudioFileClip, ImageClip, afx, concatenate_videoclips, vfx

    fade = min(0.3, params.seconds_per_slide / 4)
    clips = [
        ImageClip(path).with_duration(params.seconds_per_slide).with_effects([vfx.FadeIn(fade)])
        for path in image_paths
    ]
    video = concatenate_videoclips(clips, method="compose")
    audio = None
    if params.bgm_enabled and params.bgm_volume > 0:
        from app.services import bgm as bgm_service

        songs = bgm_service.list_builtin_bgm_files()
        if songs:
            song = random.choice(songs)
            audio = AudioFileClip(song).with_effects(
                [afx.AudioLoop(duration=video.duration), afx.MultiplyVolume(params.bgm_volume), afx.AudioFadeOut(1.0)]
            )
            video = video.with_audio(audio)
    try:
        video.write_videofile(
            output_path,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=2,
            logger=None,
        )
    finally:
        video.close()
        if audio is not None:
            audio.close()
    return output_path


def caption_text(script: SlideshowScript) -> str:
    return "\n\n".join(p for p in (script.caption, " ".join(script.hashtags)) if p)


# --------------------------------------------------------------------------- pipeline


def generate(
    task_id: str,
    params: SlideshowParams,
    local_images: Optional[List[str]] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> SlideshowResult:
    """Run the whole slideshow pipeline and return the produced files.

    ``local_images`` (trusted paths, e.g. WebUI uploads) are used as slide
    backgrounds in order before any stock search happens.
    """

    def report(pct: int, message: str):
        logger.info(f"[slideshow {task_id}] {pct}% {message}")
        if progress:
            progress(pct, message)

    out_dir = utils.task_dir(task_id)
    template: Optional[CarouselTemplate] = None
    if params.template_id:
        from app.services import carousel_template

        template = carousel_template.load_template(params.template_id)
        # Unless the caller chose explicitly, inherit the proven format's look.
        explicit = params.model_fields_set
        params = params.model_copy(
            update={
                "aspect": params.aspect if "aspect" in explicit else template.aspect,
                "style": params.style if "style" in explicit else template.visual.style,
                "photo_vibe": params.photo_vibe or template.photo_vibe,
            }
        )
    visual = template.visual if template else None

    report(5, "writing slides")
    script = params.script or generate_slideshow_script(
        params.topic, params.slide_count, params.language, params.tone, template, params.product
    )
    if not script.slides:
        raise ValueError("slideshow script has no slides")

    sample = " ".join(s.heading + s.body for s in script.slides)
    font_path = resolve_font_path(params.font_name, sample, params.style)
    size = params.aspect.to_resolution()
    credits: List[str] = []
    used_urls: set[str] = set()
    local_images = list(local_images or [])
    image_paths: List[str] = []
    total = len(script.slides)

    for i, slide in enumerate(script.slides):
        report(10 + int(70 * i / total), f"rendering slide {i + 1}/{total}")
        background = None
        if params.style != SlideStyle.minimal:
            if i < len(local_images):
                background = load_local_background(local_images[i])
            if background is None:
                background, credit = fetch_background(
                    slide.image_query or params.topic, params.image_source, size, used_urls, params.photo_vibe
                )
                if credit:
                    credits.append(f"Slide {i + 1}: {credit}")
        image = render_slide(slide, i, total, params, background, font_path, visual)
        path = os.path.join(out_dir, f"slide-{i + 1:02d}.png")
        image.save(path, "PNG", optimize=True)
        image_paths.append(path)

    text = caption_text(script)
    if credits:
        text_credits = "Photo credits:\n" + "\n".join(credits)
        with open(os.path.join(out_dir, "credits.txt"), "w", encoding="utf-8") as f:
            f.write(text_credits)
    with open(os.path.join(out_dir, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out_dir, "slideshow.json"), "w", encoding="utf-8") as f:
        f.write(script.model_dump_json(indent=2))

    report(82, "zipping slides")
    zip_path = write_zip(image_paths, text, os.path.join(out_dir, "slides.zip"), credits)

    video_path = ""
    if params.export_video:
        report(88, "encoding video")
        video_path = write_video(image_paths, params, os.path.join(out_dir, "slideshow.mp4"))

    report(100, "done")
    return SlideshowResult(
        task_id=task_id,
        images=image_paths,
        photo_credits=credits,
        zip_file=zip_path,
        video_file=video_path,
        script=script,
    )


def start(task_id: str, params: SlideshowParams) -> Optional[SlideshowResult]:
    """Background-task entry point that mirrors progress into the task state store."""
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=1, kind="slideshow")
    try:
        result = generate(
            task_id,
            params,
            progress=lambda pct, msg: sm.state.update_task(
                task_id, state=const.TASK_STATE_PROCESSING, progress=pct, kind="slideshow", stage=msg
            ),
        )
    except Exception as e:
        logger.exception(f"slideshow task {task_id} failed")
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED, progress=0, kind="slideshow", error=str(e))
        return None
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_COMPLETE,
        progress=100,
        kind="slideshow",
        images=result.images,
        zip_file=result.zip_file,
        video_file=result.video_file,
        photo_credits=result.photo_credits,
        script=result.script.model_dump(),
    )
    return result
