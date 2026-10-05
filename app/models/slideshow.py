"""Request / result models for social-media slideshow (carousel) generation."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

MIN_SLIDES = 3
MAX_SLIDES = 12


class SlideAspect(str, Enum):
    """Canvas sizes used by the big carousel / photo-mode platforms."""

    portrait_4_5 = "4:5"  # Instagram / LinkedIn carousel
    portrait_9_16 = "9:16"  # TikTok photo mode, Reels, Shorts
    square = "1:1"

    def to_resolution(self) -> tuple[int, int]:
        if self == SlideAspect.portrait_4_5:
            return 1080, 1350
        if self == SlideAspect.portrait_9_16:
            return 1080, 1920
        return 1080, 1080


class SlideStyle(str, Enum):
    # White outlined text centred over a darkened photo: the classic TikTok slideshow look.
    tiktok = "tiktok"
    # Black text on white rounded boxes over a photo, like TikTok's "highlight" text style.
    highlight = "highlight"
    # Photo with a heavy bottom gradient, big uppercase headline and an accent colour.
    bold = "bold"
    # No photo: plain dark canvas with large left-aligned text (tweet / thread style).
    minimal = "minimal"


class SlideImageSource(str, Enum):
    pexels = "pexels"
    pixabay = "pixabay"
    # Gradient backgrounds only; needs no API keys.
    none = "none"


class Slide(BaseModel):
    heading: str = ""
    body: str = ""
    image_query: str = ""


class SlideshowScript(BaseModel):
    slides: List[Slide] = Field(default_factory=list)
    caption: str = ""
    hashtags: List[str] = Field(default_factory=list)


class SlideshowParams(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    language: str = Field("", max_length=50)
    slide_count: int = Field(7, ge=MIN_SLIDES, le=MAX_SLIDES)
    # Free-form steer for the copywriter, e.g. "funny", "for beginners", "gen-z voice".
    tone: str = Field("", max_length=500)
    # When provided, these slides are rendered as-is and the LLM is skipped.
    script: Optional[SlideshowScript] = None

    aspect: SlideAspect = SlideAspect.portrait_4_5
    style: SlideStyle = SlideStyle.tiktok
    image_source: SlideImageSource = SlideImageSource.pexels
    font_name: str = Field("", max_length=200)
    accent_color: str = Field("#FFD400", pattern=r"^#[0-9A-Fa-f]{6}$")
    # Small "@handle" watermark shown on every slide; empty disables it.
    handle: str = Field("", max_length=50)
    show_slide_numbers: bool = True

    export_video: bool = False
    seconds_per_slide: float = Field(3.0, ge=1.0, le=15.0)
    bgm_enabled: bool = True
    bgm_volume: float = Field(0.4, ge=0.0, le=1.0)


class SlideshowResult(BaseModel):
    task_id: str
    images: List[str]
    zip_file: str
    video_file: str = ""
    script: SlideshowScript
