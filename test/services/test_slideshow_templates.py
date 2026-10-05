import json
import os
import shutil
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.models.slideshow import (
    CarouselTemplate,
    Slide,
    SlideshowParams,
    SlideshowScript,
    SlideStyle,
    TemplateSlide,
    TemplateVisual,
)
from app.services import carousel_template as ct
from app.services import scrapecreators as sc
from app.services import slideshow
from app.utils import utils


def _aweme(**overrides):
    image = {
        "display_image": {
            "width": 1080,
            "height": 1920,
            "url_list": ["https://cdn.example/a.heic", "https://cdn.example/a.jpeg"],
        }
    }
    data = {
        "aweme_id": "123",
        "desc": "tips #sleep",
        "create_time": 1700000000,
        "author": {"unique_id": "creator"},
        "statistics": {"play_count": 1000, "digg_count": 100, "comment_count": 5, "share_count": 10, "collect_count": 50},
        "image_post_info": {"images": [image, image, image]},
    }
    data.update(overrides)
    return data


class TestScrapeCreatorsParsing(unittest.TestCase):
    def test_parse_aweme_prefers_jpeg_over_heic(self):
        post = sc.parse_aweme(_aweme())
        self.assertEqual(post.image_urls, ["https://cdn.example/a.jpeg"] * 3)
        self.assertEqual(post.url, "https://www.tiktok.com/@creator/photo/123")
        self.assertEqual((post.plays, post.saves, post.shares), (1000, 50, 10))
        self.assertEqual(post.create_time, "2023-11-14")
        self.assertAlmostEqual(post.save_rate, 0.05)

    def test_videos_are_not_carousels(self):
        self.assertIsNone(sc.parse_aweme(_aweme(image_post_info=None)))
        self.assertIsNone(sc.parse_top_item({"content_type": "video", "images": []}))

    def test_parse_top_item(self):
        post = sc.parse_top_item(
            {
                "id": "9",
                "content_type": "multi_photo",
                "url": "https://www.tiktok.com/@a/photo/9",
                "images": ["https://cdn/1.webp", "https://cdn/2.webp", "https://cdn/3.webp"],
                "author": {"unique_id": "a"},
                "statistics": {"play_count": 10},
                "create_time": "2026-08-18T22:32:51.000Z",
            }
        )
        self.assertEqual(post.slide_count, 3)
        self.assertEqual(post.create_time, "2026-08-18")

    def test_search_ranks_by_score_and_dedupes(self):
        weak = {"id": "1", "content_type": "multi_photo", "images": ["https://x/1"] * 4,
                "statistics": {"play_count": 100000, "digg_count": 100}}
        strong = {"id": "2", "content_type": "multi_photo", "images": ["https://x/1"] * 4,
                  "statistics": {"play_count": 100000, "digg_count": 5000, "collect_count": 4000}}
        short = {"id": "3", "content_type": "multi_photo", "images": ["https://x/1"],
                 "statistics": {"play_count": 9}}
        page = {"success": True, "items": [weak, strong, strong, short], "cursor": None}
        with patch.object(sc, "_get", return_value=page) as get:
            posts = sc.search_carousels("sleep", pages=3)
        get.assert_called_once()
        self.assertEqual([p.id for p in posts], ["2", "1"])

    def test_missing_key_is_a_clear_error(self):
        with patch.dict(os.environ, {"SCRAPECREATORS_API_KEY": ""}), patch.dict(
            sc.config.app, {"scrapecreators_api_key": ""}
        ):
            with self.assertRaises(sc.ScrapeCreatorsError):
                sc.get_api_key()

    def test_get_carousel_rejects_non_tiktok_urls(self):
        with self.assertRaises(ValueError):
            sc.get_carousel("https://evil.example/photo/1")


def _line(text, y, x0=200, x1=880, h=80):
    return ct.OcrLine(text, x0, y, x1, y + h)


class TestTemplateAnalysis(unittest.TestCase):
    def test_paragraphs_split_on_large_gaps(self):
        lines = [_line("1. Sleep with your feet", 540), _line("outside the blanket", 620),
                 _line("It helps your body", 760), _line("cool down faster.", 840)]
        heading, body = ct.slide_text(lines)
        self.assertEqual(heading, "1. Sleep with your feet outside the blanket")
        self.assertEqual(body, "It helps your body cool down faster.")

    def test_roles(self):
        self.assertEqual(ct.classify(0, 5, "Weird hacks", ""), "hook")
        self.assertEqual(ct.classify(2, 5, "2. Do this", "because"), "item")
        self.assertEqual(ct.classify(2, 5, "rule 1: coffee first", ""), "item")
        self.assertEqual(ct.classify(2, 5, "Step 3) breathe", ""), "item")
        self.assertEqual(ct.classify(3, 5, "3. Journal", "Using ventytherapy.com helped"), "plug")
        self.assertEqual(ct.classify(4, 5, "Save this for later", "follow for more"), "cta")
        self.assertEqual(ct.classify(2, 5, "", ""), "photo")

    def test_measure_visual_on_dark_photo_with_centred_text(self):
        img = Image.new("RGB", (1080, 1920), (30, 30, 35))
        lines = [_line("a", 900, 240, 840), _line("b", 990, 240, 840)]
        visual = ct.measure_visual([img], [lines])
        self.assertEqual(visual.style, SlideStyle.native)
        self.assertEqual(visual.align, "center")
        self.assertAlmostEqual(visual.text_y, (900 + 1070) / 2 / 1920, places=2)
        self.assertLess(visual.bg_luminance, 0.2)
        self.assertGreaterEqual(visual.overlay, 0.4)

    def test_measure_visual_detects_highlight_boxes(self):
        img = Image.new("RGB", (1080, 1920), (60, 70, 80))
        ImageDraw.Draw(img).rectangle((200, 900, 880, 980), fill=(255, 255, 255))
        visual = ct.measure_visual([img], [[_line("boxed", 900)]])
        self.assertEqual(visual.style, SlideStyle.highlight)

    def test_build_and_store_template(self):
        post = sc.parse_aweme(_aweme())
        images = [Image.new("RGB", (1080, 1920), (20, 20, 20)) for _ in range(3)]
        fake_ocr = [
            [_line("Weird hacks from my doctor", 800)],
            [_line("1. Drink water", 700), _line("It helps.", 900)],
            [_line("2. Try myapp.com", 700), _line("I use it daily.", 900)],
        ]
        with patch.object(ct, "ocr_lines", side_effect=fake_ocr):
            template, _ = ct.build_template(post, images=images, use_llm=False)
        self.assertEqual([s.role for s in template.slides], ["hook", "item", "plug"])
        self.assertEqual(template.plug_slide, 2)
        self.assertEqual(template.aspect.value, "9:16")
        self.assertIn("@creator", template.name)
        self.assertEqual(template.photo_vibe, "dark moody aesthetic")

        try:
            ct.save_template(template)
            self.assertEqual(ct.load_template(template.id).slides[1].body, "It helps.")
            self.assertIn(template.id, [t.id for t in ct.list_templates()])
        finally:
            ct.delete_template(template.id)
        with self.assertRaises(FileNotFoundError):
            ct.load_template(template.id)

    def test_template_id_cannot_traverse(self):
        for bad in ("../config", "a/b", ""):
            with self.assertRaises(ValueError):
                ct.load_template(bad)

    def test_llm_notes_fall_back_to_heuristics(self):
        post = sc.parse_aweme(_aweme())
        images = [Image.new("RGB", (1080, 1920)) for _ in range(3)]
        ocr = [[_line("Hook", 800)], [_line("1. a", 800)], [_line("2. b", 800)]]
        with patch.object(ct, "ocr_lines", side_effect=ocr), patch.object(
            ct, "describe_format", side_effect=ValueError("no llm")
        ):
            template, _ = ct.build_template(post, images=images, use_llm=True)
        self.assertIn("2 numbered tips", template.format_notes)


TEMPLATE = CarouselTemplate(
    id="t1",
    name="authority hacks",
    plays=1_000_000,
    saves=50_000,
    hook_pattern="Some weird [topic] hacks from my [authority]",
    source_caption="tips #x",
    slides=[
        TemplateSlide(role="hook", heading="Weird sleep hacks from my doctor", body="(that work)"),
        TemplateSlide(role="item", heading="1. Feet out", body="Cools you down."),
        TemplateSlide(role="plug", heading="2. Journal", body="Using venty.com helped."),
        TemplateSlide(role="item", heading="3. Melt", body="Relax each part."),
    ],
    plug_slide=2,
    photo_vibe="dark moody",
    visual=TemplateVisual(text_y=0.5, font_ratio=0.055, bg_luminance=0.25),
)


class TestTemplateGeneration(unittest.TestCase):
    def setUp(self):
        self.task_id = "test-tpl-" + utils.get_uuid(True)

    def tearDown(self):
        shutil.rmtree(utils.task_dir(self.task_id), ignore_errors=True)

    def test_prompt_copies_structure_and_places_product(self):
        prompt = slideshow.build_template_prompt(TEMPLATE, "procrastination", product="Focusly")
        self.assertIn("Exactly 4 slides", prompt)
        self.assertIn("hook, item, plug, item", prompt)
        self.assertIn("On slide 3, mention Focusly", prompt)
        self.assertIn("1,000,000 views", prompt)
        self.assertIn("procrastination", prompt)

    def test_prompt_without_product_drops_plug(self):
        prompt = slideshow.build_template_prompt(TEMPLATE, "x")
        self.assertIn("Do not mention any product", prompt)

    def test_template_forces_slide_count(self):
        reply = json.dumps({"slides": [{"heading": f"s{i}"} for i in range(4)], "caption": "c"})
        with patch.object(slideshow.llm, "_generate_response", return_value=reply) as gen:
            script = slideshow.generate_slideshow_script("x", slide_count=9, template=TEMPLATE)
        self.assertEqual(len(script.slides), 4)
        self.assertIn("Exactly 4 slides", gen.call_args.args[0])

    def test_generate_inherits_template_look(self):
        script = SlideshowScript(slides=[Slide(heading=s.heading, body=s.body) for s in TEMPLATE.slides])
        with patch.object(ct, "load_template", return_value=TEMPLATE), patch.object(
            slideshow, "fetch_background", return_value=(Image.new("RGB", (900, 1600), "white"), "credit X")
        ) as fetch:
            result = slideshow.generate(self.task_id, SlideshowParams(topic="x", template_id="t1", script=script))
        self.assertEqual(fetch.call_args.args[4], "dark moody")  # template photo vibe
        self.assertEqual(len(result.photo_credits), 4)
        with Image.open(result.images[0]) as img:
            self.assertEqual(img.size, (1080, 1920))
            # A white stock photo was darkened toward the template's moody brightness.
            self.assertLess(img.convert("L").getpixel((20, 20)), 170)

    def test_native_style_uses_tiktok_sans(self):
        path = slideshow.resolve_font_path("", "hello", SlideStyle.native)
        self.assertTrue(path.endswith(slideshow.NATIVE_FONT))
        font = slideshow.load_font(path, 50)
        self.assertGreater(font.getlength("hello"), 0)


class TestOpenverse(unittest.TestCase):
    def test_filters_small_and_vector_results_and_builds_credits(self):
        payload = {
            "results": [
                {"url": "https://x/map.svg", "width": 2000, "height": 2000},
                {"url": "https://x/tiny.jpg", "width": 300, "height": 400},
                {"url": "https://x/good.jpg", "width": 1200, "height": 1800, "title": "Desk",
                 "creator": "Ann", "license": "by", "license_version": "4.0",
                 "foreign_landing_url": "https://src/desk"},
            ]
        }

        class Resp:
            def json(self):
                return payload

        with patch.object(slideshow.requests, "get", return_value=Resp()):
            photos = slideshow.search_images_openverse("desk", 1080, 1920)
        urls = [p.url for p in photos]
        self.assertNotIn("https://x/map.svg", urls)
        self.assertNotIn("https://x/tiny.jpg", urls)
        self.assertIn("https://x/good.jpg", urls)
        self.assertIn('"Desk" by Ann (CC BY 4.0) https://src/desk', photos[urls.index("https://x/good.jpg")].credit)


class TestEnvOverrides(unittest.TestCase):
    def test_env_vars_configure_keys(self):
        from app.config import config

        env = {
            "LLM_PROVIDER": "gemini",
            "LLM_API_KEY": "g-key",
            "PEXELS_API_KEY": "p1, p2",
            "SCRAPECREATORS_API_KEY": "sc-key",
        }
        saved = dict(config.app)
        try:
            with patch.dict(os.environ, env):
                config._apply_env_overrides()
            self.assertEqual(config.app["llm_provider"], "gemini")
            self.assertEqual(config.app["gemini_api_key"], "g-key")
            self.assertEqual(config.app["pexels_api_keys"], ["p1", "p2"])
            self.assertEqual(config.app["scrapecreators_api_key"], "sc-key")
        finally:
            config.app.clear()
            config.app.update(saved)


if __name__ == "__main__":
    unittest.main()
