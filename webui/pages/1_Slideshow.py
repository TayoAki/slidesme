"""WebUI page: generate viral-style photo carousels / slideshows."""

import os
import sys

import streamlit as st

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.models.slideshow import (  # noqa: E402
    MAX_SLIDES,
    MIN_SLIDES,
    Slide,
    SlideAspect,
    SlideImageSource,
    SlideshowParams,
    SlideshowScript,
    SlideStyle,
)
from app.services import slideshow  # noqa: E402
from app.utils import utils  # noqa: E402

st.set_page_config(page_title="Slideshow Maker", page_icon="🖼️", layout="wide")
st.title("🖼️ Viral Slideshow Maker")
st.caption(
    "Topic in, carousel out: hook slide → value slides → CTA, rendered for "
    "TikTok photo mode, Instagram and LinkedIn carousels. LLM and Pexels/Pixabay "
    "keys are shared with the main video page settings."
)

STYLE_HELP = {
    "tiktok": "White outlined text over a darkened photo",
    "highlight": "Text on white/black boxes, like TikTok's highlight text",
    "bold": "Uppercase headline over a dark bottom gradient with accent colour",
    "minimal": "No photo, dark canvas, tweet/thread look",
}

left, right = st.columns([1, 1])
with left:
    topic = st.text_area("Topic", placeholder="e.g. 7 habits that quietly wreck your sleep", height=80)
    c1, c2 = st.columns(2)
    slide_count = c1.slider("Slides", MIN_SLIDES, MAX_SLIDES, 7)
    language = c2.text_input("Language", placeholder="auto (topic language)")
    tone = st.text_input("Tone / extra instructions", placeholder="e.g. funny, gen-z voice, for beginners")
with right:
    c1, c2 = st.columns(2)
    style = c1.selectbox(
        "Style", [s.value for s in SlideStyle], format_func=lambda s: f"{s} — {STYLE_HELP[s]}"
    )
    aspect = c2.selectbox(
        "Size",
        [a.value for a in SlideAspect],
        format_func=lambda a: {"4:5": "4:5 · Instagram/LinkedIn", "9:16": "9:16 · TikTok/Reels", "1:1": "1:1 · Square"}[a],
    )
    image_source = c1.selectbox("Background photos", [s.value for s in SlideImageSource])
    handle = c2.text_input("@handle watermark", placeholder="optional")
    accent = c1.color_picker("Accent colour (bold style)", "#FFD400")
    show_numbers = c2.checkbox("Show slide counter", True)
    uploads = st.file_uploader(
        "Your own background images (optional, used in slide order)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    c1, c2, c3 = st.columns(3)
    export_video = c1.checkbox("Also export MP4", False)
    seconds = c2.number_input("Sec / slide", 1.0, 15.0, 3.0, 0.5, disabled=not export_video)
    bgm = c3.checkbox("Music", True, disabled=not export_video)

st.divider()

if st.button("✍️ Write slides", disabled=not topic.strip()):
    with st.spinner("Writing a scroll-stopping carousel…"):
        try:
            st.session_state["slideshow_script"] = slideshow.generate_slideshow_script(
                topic, slide_count, language, tone
            )
        except Exception as e:
            st.error(f"Script generation failed: {e}")

script: SlideshowScript | None = st.session_state.get("slideshow_script")
if script:
    st.subheader("Edit slides")
    rows = st.data_editor(
        [s.model_dump() for s in script.slides],
        num_rows="dynamic",
        width="stretch",
        column_config={
            "heading": st.column_config.TextColumn("Heading", width="large"),
            "body": st.column_config.TextColumn("Body", width="large"),
            "image_query": st.column_config.TextColumn("Photo search"),
        },
        key="slideshow_editor",
    )
    caption = st.text_area("Post caption", script.caption)
    hashtags = st.text_input("Hashtags", " ".join(script.hashtags))

    if st.button("🎨 Render slideshow", type="primary"):
        edited = SlideshowScript(
            slides=[Slide(**{k: (r.get(k) or "") for k in ("heading", "body", "image_query")}) for r in rows if (r.get("heading") or r.get("body"))],
            caption=caption,
            hashtags=[t for t in hashtags.split() if t],
        )
        task_id = utils.get_uuid()
        local_images = []
        if uploads:
            upload_dir = utils.task_dir(os.path.join(task_id, "uploads"))
            os.makedirs(upload_dir, exist_ok=True)
            for i, f in enumerate(uploads):
                path = os.path.join(upload_dir, f"{i:02d}{os.path.splitext(f.name)[1].lower()}")
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                local_images.append(path)
        params = SlideshowParams(
            topic=topic or "slideshow",
            language=language,
            slide_count=max(MIN_SLIDES, min(MAX_SLIDES, len(edited.slides))),
            script=edited,
            aspect=aspect,
            style=style,
            image_source=image_source,
            accent_color=accent.upper(),
            handle=handle,
            show_slide_numbers=show_numbers,
            export_video=export_video,
            seconds_per_slide=seconds,
            bgm_enabled=bgm,
        )
        bar = st.progress(0, "Starting…")
        try:
            st.session_state["slideshow_result"] = slideshow.generate(
                task_id, params, local_images=local_images, progress=lambda p, m: bar.progress(p, m)
            )
        except Exception as e:
            st.error(f"Rendering failed: {e}")

result = st.session_state.get("slideshow_result")
if result:
    st.subheader("Your slideshow")
    cols = st.columns(4)
    for i, path in enumerate(result.images):
        cols[i % 4].image(path, width="stretch")
    c1, c2 = st.columns(2)
    with open(result.zip_file, "rb") as f:
        c1.download_button("⬇️ Download slides (.zip)", f.read(), "slides.zip", "application/zip")
    if result.video_file:
        with open(result.video_file, "rb") as f:
            c2.download_button("⬇️ Download MP4", f.read(), "slideshow.mp4", "video/mp4")
        st.video(result.video_file)
    st.text_area("Caption to paste", slideshow.caption_text(result.script), height=120)
    st.caption(f"Files saved in {os.path.dirname(result.zip_file)}")
