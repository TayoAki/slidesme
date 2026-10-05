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
    # Mimics native TikTok photo-mode text: TikTok Sans, white, thin outline,
    # heading + explanation in one centred block. Usually what templates pick.
    native = "native"
    # White outlined text centred over a darkened photo: the classic TikTok slideshow look.
    tiktok = "tiktok"
    # Black text on white rounded boxes over a photo, like TikTok's "highlight" text style.
    highlight = "highlight"
    # Photo with a heavy bottom gradient, big uppercase headline and an accent colour.
    bold = "bold"
    # No photo: plain dark canvas with large left-aligned text (tweet / thread style).
    minimal = "minimal"


class SlideImageSource(str, Enum):
    # Real CC-licensed photographs from Openverse; needs no API key.
    openverse = "openverse"
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


class TemplateSlide(BaseModel):
    # hook | item | cta | plug | photo | text
    role: str = "text"
    heading: str = ""
    body: str = ""

    @property
    def words(self) -> int:
        return len(f"{self.heading} {self.body}".split())


class TemplateVisual(BaseModel):
    style: SlideStyle = SlideStyle.native
    # Vertical centre of the text block, as a fraction of slide height.
    text_y: float = Field(0.42, ge=0.1, le=0.9)
    # Font size as a fraction of slide width.
    font_ratio: float = Field(0.046, ge=0.02, le=0.12)
    align: str = "center"
    # 0 = photo untouched, 1 = black. Matches how moody the source photos are.
    overlay: float = Field(0.25, ge=0.0, le=0.8)
    # Mean background brightness of the source slides (0-1), informational.
    bg_luminance: float = 0.4


class CarouselTemplate(BaseModel):
    """A reusable format reverse-engineered from a carousel that performed."""

    id: str
    name: str
    created_at: str = ""
    source_url: str = ""
    source_author: str = ""
    source_caption: str = ""
    plays: int = 0
    likes: int = 0
    saves: int = 0
    shares: int = 0
    comments: int = 0
    aspect: SlideAspect = SlideAspect.portrait_9_16
    slides: List[TemplateSlide] = Field(default_factory=list)
    # Index of the slide that quietly mentions a product/app/site, if any.
    plug_slide: Optional[int] = None
    hook_pattern: str = ""
    format_notes: str = ""
    # Extra words appended to photo searches, e.g. "dark moody aesthetic".
    photo_vibe: str = ""
    visual: TemplateVisual = Field(default_factory=TemplateVisual)

    @property
    def slide_count(self) -> int:
        return len(self.slides)


class SlideshowParams(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    language: str = Field("", max_length=50)
    slide_count: int = Field(7, ge=MIN_SLIDES, le=MAX_SLIDES)
    # Free-form steer for the copywriter, e.g. "funny", "for beginners", "gen-z voice".
    tone: str = Field("", max_length=500)
    # Reusable format (see app/services/carousel_template.py) to copy.
    template_id: str = Field("", max_length=100, pattern=r"^[A-Za-z0-9_-]*$")
    # Product / app / site to weave in the way the template's plug slide did.
    product: str = Field("", max_length=300)
    # When provided, these slides are rendered as-is and the LLM is skipped.
    script: Optional[SlideshowScript] = None

    aspect: SlideAspect = SlideAspect.portrait_9_16
    style: SlideStyle = SlideStyle.native
    image_source: SlideImageSource = SlideImageSource.openverse
    # Extra photo-search words; defaults to the template's photo_vibe.
    photo_vibe: str = Field("", max_length=100)
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
    photo_credits: List[str] = Field(default_factory=list)
    zip_file: str
    video_file: str = ""
    script: SlideshowScript
