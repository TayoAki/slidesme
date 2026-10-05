import json
import os
import shutil
import unittest
import zipfile
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageFont
from pydantic import ValidationError

from app.controllers.v1 import slideshow as slideshow_controller
from app.models.exception import HttpException
from app.models.slideshow import (
    Slide,
    SlideAspect,
    SlideshowParams,
    SlideshowScript,
    SlideStyle,
)
from app.services import slideshow
from app.utils import utils

SCRIPT = SlideshowScript(
    slides=[
        Slide(heading="5 habits that quietly wreck your sleep", body="Number 3 hurts"),
        Slide(heading="1. Scrolling in bed", body="Blue light keeps your brain wired."),
        Slide(heading="Save this for tonight", body="Follow for part 2"),
    ],
    caption="Which one are you guilty of?",
    hashtags=["#sleep", "#habits"],
)


class TestSlideshowScriptParsing(unittest.TestCase):
    def test_parses_fenced_json_and_strips_emoji(self):
        response = "```json\n" + json.dumps(
            {
                "slides": [
                    {"heading": "Hook 😱", "body": "", "image_query": "dark bedroom"},
                    {"heading": "1. Tip", "body": "Do  this 🔥", "image_query": "phone"},
                    {"heading": "Follow", "body": "", "image_query": "smile"},
                ],
                "caption": "Try it 💤",
                "hashtags": ["sleep", "#sleep", "#tips"],
            }
        ) + "\n```"
        script = slideshow.parse_slideshow_script(response, 3)
        self.assertEqual([s.heading for s in script.slides], ["Hook", "1. Tip", "Follow"])
        self.assertEqual(script.slides[1].body, "Do this")
        # Emoji is fine in the caption: platforms render it natively.
        self.assertEqual(script.caption, "Try it 💤")
        self.assertEqual(script.hashtags, ["#sleep", "#tips"])

    def test_truncates_extra_slides(self):
        slides = [{"heading": f"S{i}"} for i in range(6)]
        script = slideshow.parse_slideshow_script(json.dumps({"slides": slides}), 4)
        self.assertEqual(len(script.slides), 4)

    def test_rejects_unusable_response(self):
        for response in ("no json here", '{"slides": []}', '{"slides": [{"heading": "only"}]}'):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    slideshow.parse_slideshow_script(response, 5)

    def test_generate_retries_invalid_output(self):
        good = json.dumps({"slides": [{"heading": "a"}, {"heading": "b"}, {"heading": "c"}]})
        with patch.object(slideshow.llm, "_generate_response", side_effect=["garbage", good]) as gen:
            script = slideshow.generate_slideshow_script("topic", 3)
        self.assertEqual(gen.call_count, 2)
        self.assertEqual(len(script.slides), 3)

    def test_prompt_contains_structure_and_language(self):
        prompt = slideshow.build_slideshow_prompt("budget travel", 6, "Spanish", "funny")
        self.assertIn("6-slide", prompt)
        self.assertIn("Slide 6 is the CTA", prompt)
        self.assertIn("in Spanish", prompt)
        self.assertIn("funny", prompt)


class TestSlideshowRendering(unittest.TestCase):
    def setUp(self):
        self.task_id = "test-slideshow-" + utils.get_uuid(True)

    def tearDown(self):
        shutil.rmtree(utils.task_dir(self.task_id), ignore_errors=True)

    def test_wrap_text_respects_width(self):
        font = ImageFont.truetype(slideshow.resolve_font_path("", "abc"), 60)
        lines = slideshow.wrap_text("the quick brown fox jumps over the lazy dog " * 3, font, 500)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(font.getlength(line) <= 500 for line in lines))

    def test_wrap_text_breaks_cjk_without_spaces(self):
        font = ImageFont.truetype(slideshow.resolve_font_path("", "睡眠"), 60)
        lines = slideshow.wrap_text("改善睡眠质量的七个简单习惯让你每天精力充沛", font, 400)
        self.assertGreater(len(lines), 1)
        self.assertNotIn(" ", "".join(lines))

    def test_font_name_cannot_escape_font_dir(self):
        with self.assertRaises(ValueError):
            slideshow.resolve_font_path("../../config.toml", "")

    def test_generate_renders_every_style_and_size(self):
        for style in SlideStyle:
            for aspect in SlideAspect:
                with self.subTest(style=style, aspect=aspect):
                    params = SlideshowParams(
                        topic="sleep",
                        script=SCRIPT,
                        style=style,
                        aspect=aspect,
                        image_source="none",
                        handle="coach",
                    )
                    image = slideshow.render_slide(
                        SCRIPT.slides[1], 1, 3, params, Image.new("RGB", (800, 600), "teal"),
                        slideshow.resolve_font_path("", ""),
                    )
                    self.assertEqual(image.size, aspect.to_resolution())

    def test_generate_writes_pngs_zip_and_caption(self):
        params = SlideshowParams(topic="sleep", script=SCRIPT, image_source="none")
        with patch.object(slideshow, "generate_slideshow_script") as gen:
            result = slideshow.generate(self.task_id, params)
        gen.assert_not_called()
        self.assertEqual(len(result.images), 3)
        for path in result.images:
            with Image.open(path) as img:
                self.assertEqual(img.size, (1080, 1350))
        with zipfile.ZipFile(result.zip_file) as zf:
            names = zf.namelist()
            self.assertIn("caption.txt", names)
            self.assertIn("slide-01.png", names)
            self.assertIn("#sleep", zf.read("caption.txt").decode())
        self.assertEqual(result.video_file, "")

    def test_stock_photo_failure_falls_back_to_gradient(self):
        params = SlideshowParams(topic="sleep", script=SCRIPT, image_source="pexels")
        with patch.object(slideshow, "search_images_pexels", side_effect=ValueError("no key")):
            result = slideshow.generate(self.task_id, params)
        self.assertEqual(len(result.images), 3)

    def test_local_images_are_used_before_search(self):
        local = os.path.join(utils.task_dir(self.task_id), "bg.jpg")
        Image.new("RGB", (900, 1600), (255, 0, 0)).save(local)
        params = SlideshowParams(topic="t", script=SCRIPT, style="highlight", image_source="pexels")
        with patch.object(slideshow, "fetch_background", return_value=None) as fetch:
            result = slideshow.generate(self.task_id, params, local_images=[local])
        # Slide 1 used the local image, the remaining two went to stock search.
        self.assertEqual(fetch.call_count, 2)
        with Image.open(result.images[0]) as img:
            r, g, b = img.convert("RGB").getpixel((5, 5))
        self.assertGreater(r, 200)
        self.assertLess(g, 60)


class TestSlideshowParams(unittest.TestCase):
    def test_bounds(self):
        with self.assertRaises(ValidationError):
            SlideshowParams(topic="x", slide_count=2)
        with self.assertRaises(ValidationError):
            SlideshowParams(topic="x", accent_color="red")
        with self.assertRaises(ValidationError):
            SlideshowParams(topic="")


class TestSlideshowController(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(slideshow_controller.router)
        self.client = TestClient(app)

    def test_create_schedules_background_task(self):
        with patch.object(slideshow_controller.slideshow_service, "start") as start:
            response = self.client.post("/api/v1/slideshows", json={"topic": "sleep tips"})
        self.assertEqual(response.status_code, 200)
        task_id = response.json()["data"]["task_id"]
        start.assert_called_once()
        self.assertEqual(start.call_args.args[0], task_id)
        self.assertEqual(start.call_args.args[1].topic, "sleep tips")

    def test_rejects_font_outside_font_dir(self):
        client = TestClient(self.client.app, raise_server_exceptions=True)
        with patch.object(slideshow_controller.slideshow_service, "start") as start:
            with self.assertRaises(HttpException) as raised:
                client.post(
                    "/api/v1/slideshows", json={"topic": "x", "font_name": "../../config.toml"}
                )
        self.assertEqual(raised.exception.status_code, 400)
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
