"""Find TikTok photo carousels that already perform, via the Scrape Creators API.

https://docs.scrapecreators.com - authenticated with an ``x-api-key`` header.
The key is read from ``config.app["scrapecreators_api_key"]`` or the
``SCRAPECREATORS_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, List, Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config

API_BASE = "https://api.scrapecreators.com"
PUBLISH_TIMES = ("all-time", "yesterday", "this-week", "this-month", "last-3-months", "last-6-months")
SORT_OPTIONS = ("relevance", "most-liked", "date-posted")
_TIKTOK_URL_RE = re.compile(r"^https://(www\.|vm\.|m\.)?tiktok\.com/", re.IGNORECASE)


class ScrapeCreatorsError(RuntimeError):
    pass


class CarouselPost(BaseModel):
    id: str
    url: str = ""
    author: str = ""
    desc: str = ""
    image_urls: List[str] = Field(default_factory=list)
    width: int = 1080
    height: int = 1920
    plays: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    create_time: str = ""

    @property
    def slide_count(self) -> int:
        return len(self.image_urls)

    @property
    def save_rate(self) -> float:
        return self.saves / self.plays if self.plays else 0.0

    @property
    def share_rate(self) -> float:
        return self.shares / self.plays if self.plays else 0.0

    @property
    def engagement_rate(self) -> float:
        if not self.plays:
            return 0.0
        return (self.likes + self.comments + self.shares + self.saves) / self.plays

    @property
    def score(self) -> float:
        """Rank by proven reach weighted by how much people *kept* it.

        Saves and shares are the strongest "this format works" signal for
        carousels, so they count more than likes. log-scaled plays stop one
        mega-viral outlier from drowning everything else.
        """
        import math

        if not self.plays:
            return 0.0
        quality = (self.likes + 3 * self.comments + 4 * self.shares + 5 * self.saves) / self.plays
        return round(math.log10(self.plays + 1) * quality * 100, 2)


def get_api_key() -> str:
    key = os.environ.get("SCRAPECREATORS_API_KEY") or config.app.get("scrapecreators_api_key", "")
    if not key:
        raise ScrapeCreatorsError(
            "Scrape Creators API key is not set. Add scrapecreators_api_key to config.toml "
            "or set the SCRAPECREATORS_API_KEY environment variable."
        )
    return str(key).strip()


def _get(path: str, params: dict) -> dict:
    api_key = get_api_key()
    try:
        r = requests.get(
            f"{API_BASE}{path}",
            params={k: v for k, v in params.items() if v not in (None, "")},
            headers={"x-api-key": api_key},
            proxies=config.proxy,
            timeout=(15, 90),
        )
    except requests.RequestException as e:
        raise ScrapeCreatorsError(f"Scrape Creators request failed: {type(e).__name__}") from e
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code != 200 or not isinstance(data, dict) or data.get("success") is False:
        message = data.get("message") or data.get("error") if isinstance(data, dict) else ""
        raise ScrapeCreatorsError(f"Scrape Creators error {r.status_code}: {message or r.text[:200]}")
    logger.info(
        f"scrapecreators {path}: credits charged={data.get('credits_charged')}, "
        f"remaining={data.get('credits_remaining')}"
    )
    return data


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _pick_image_url(urls: Any) -> str:
    """TikTok lists HEIC first; Pillow reads JPEG/WebP out of the box."""
    if not isinstance(urls, list):
        return ""
    urls = [u for u in urls if isinstance(u, str) and u.startswith("https://")]
    for ext in (".jpeg", ".jpg", ".webp", ".png"):
        for u in urls:
            if ext in u.split("?")[0]:
                return u
    return ""


def _format_time(value: Any) -> str:
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
    if isinstance(value, str):
        return value[:10]
    return ""


def _stats_kwargs(stats: Any) -> dict:
    stats = stats if isinstance(stats, dict) else {}
    return {
        "plays": _int(stats.get("play_count")),
        "likes": _int(stats.get("digg_count")),
        "comments": _int(stats.get("comment_count")),
        "shares": _int(stats.get("share_count")),
        "saves": _int(stats.get("collect_count")),
    }


def parse_aweme(aweme: dict) -> Optional[CarouselPost]:
    """Parse a raw TikTok item (aweme) - returns None when it is not a photo carousel."""
    if not isinstance(aweme, dict):
        return None
    info = aweme.get("image_post_info")
    if not isinstance(info, dict) or not isinstance(info.get("images"), list):
        return None
    urls, width, height = [], 1080, 1920
    for image in info["images"]:
        if not isinstance(image, dict):
            continue
        display = image.get("display_image") or {}
        url = _pick_image_url(display.get("url_list")) or _pick_image_url(
            (image.get("thumbnail") or {}).get("url_list")
        )
        if url:
            urls.append(url)
            if len(urls) == 1:
                width = _int(display.get("width")) or width
                height = _int(display.get("height")) or height
    if not urls:
        return None
    author = (aweme.get("author") or {}).get("unique_id") or ""
    post_id = str(aweme.get("aweme_id") or "")
    return CarouselPost(
        id=post_id,
        url=f"https://www.tiktok.com/@{author}/photo/{post_id}" if author else "",
        author=author,
        desc=str(aweme.get("desc") or ""),
        image_urls=urls,
        width=width,
        height=height,
        create_time=_format_time(aweme.get("create_time")),
        **_stats_kwargs(aweme.get("statistics")),
    )


def parse_top_item(item: dict) -> Optional[CarouselPost]:
    """Parse an item from /v1/tiktok/search/top (already-flattened shape)."""
    if not isinstance(item, dict) or item.get("content_type") != "multi_photo":
        return None
    urls = [u for u in (item.get("images") or []) if isinstance(u, str) and u.startswith("https://")]
    if not urls:
        return None
    author = (item.get("author") or {}).get("unique_id") or ""
    return CarouselPost(
        id=str(item.get("id") or ""),
        url=str(item.get("url") or ""),
        author=author,
        desc=str(item.get("desc") or ""),
        image_urls=urls,
        create_time=_format_time(item.get("create_time")),
        **_stats_kwargs(item.get("statistics")),
    )


def _dedupe_rank(posts: List[CarouselPost], min_slides: int) -> List[CarouselPost]:
    seen, out = set(), []
    for post in posts:
        if post.id in seen or post.slide_count < min_slides:
            continue
        seen.add(post.id)
        out.append(post)
    return sorted(out, key=lambda p: p.score, reverse=True)


def search_carousels(
    query: str,
    publish_time: str = "all-time",
    sort_by: str = "relevance",
    pages: int = 1,
    region: str = "",
    min_slides: int = 3,
) -> List[CarouselPost]:
    """TikTok 'Top' search, keeping only photo carousels, best performers first.

    Each page costs one Scrape Creators credit.
    """
    if not query.strip():
        raise ValueError("query is required")
    posts: List[CarouselPost] = []
    cursor = None
    for _ in range(max(1, min(pages, 5))):
        data = _get(
            "/v1/tiktok/search/top",
            {
                "query": query.strip(),
                "publish_time": publish_time if publish_time in PUBLISH_TIMES else None,
                "sort_by": sort_by if sort_by in SORT_OPTIONS else None,
                "region": region,
                "cursor": cursor,
            },
        )
        posts.extend(p for p in map(parse_top_item, data.get("items") or []) if p)
        cursor = data.get("cursor")
        if not cursor:
            break
    return _dedupe_rank(posts, min_slides)


def profile_carousels(handle: str, sort_by: str = "popular", min_slides: int = 3) -> List[CarouselPost]:
    """A creator's photo carousels (one credit)."""
    handle = handle.strip().lstrip("@")
    if not handle:
        raise ValueError("handle is required")
    data = _get("/v3/tiktok/profile/videos", {"handle": handle, "sort_by": sort_by})
    posts = [p for p in map(parse_aweme, data.get("aweme_list") or []) if p]
    return _dedupe_rank(posts, min_slides)


def get_carousel(url: str) -> CarouselPost:
    """A single carousel by its TikTok URL (one credit)."""
    url = url.strip()
    if not _TIKTOK_URL_RE.match(url):
        raise ValueError("expected a https://www.tiktok.com/... URL")
    data = _get("/v2/tiktok/video", {"url": url})
    post = parse_aweme(data.get("aweme_detail") or {})
    if post is None:
        raise ScrapeCreatorsError("that TikTok is a video, not a photo carousel")
    return post
