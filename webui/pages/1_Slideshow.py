"""WebUI page: find carousels that already work, turn them into templates, make new ones."""

import os
import sys

import streamlit as st

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.config import config  # noqa: E402
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
from app.services import carousel_template, scrapecreators, slideshow  # noqa: E402
from app.utils import utils  # noqa: E402
from app.utils.webui_auth import require_password  # noqa: E402

st.set_page_config(page_title="Slideshow Maker", page_icon="🖼️", layout="wide")
require_password()

st.title("🖼️ Viral Slideshow Maker")
st.caption(
    "**1. Find winners** – search TikTok for photo carousels that already got views and saves. "
    "**2. Templates** – turn one into a reusable format (slide text is read by on-device OCR). "
    "**3. Create** – AI writes a new carousel in that format on your topic, with real photos."
)

STYLE_HELP = {
    "native": "TikTok-native text (TikTok Sans, centred)",
    "tiktok": "Big white outlined text",
    "highlight": "Text on white/black boxes",
    "bold": "Uppercase headline, dark gradient",
    "minimal": "No photo, dark tweet look",
}
PHOTO_HELP = {
    "openverse": "openverse – real CC photos, no key",
    "pexels": "pexels – needs API key",
    "pixabay": "pixabay – needs API key",
    "none": "none – gradients",
}


def _fmt(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


def _llm_ready() -> bool:
    provider = str(config.app.get("llm_provider", "")).lower()
    return bool(provider) and bool(config.app.get(f"{provider}_api_key") or provider in ("ollama", "claude_code"))


tab_find, tab_templates, tab_create = st.tabs(["① Find winners", "② Templates", "③ Create"])

# --------------------------------------------------------------------------- find
with tab_find:
    has_sc_key = bool(os.getenv("SCRAPECREATORS_API_KEY") or config.app.get("scrapecreators_api_key"))
    if not has_sc_key:
        key = st.text_input("Scrape Creators API key", type="password", help="https://scrapecreators.com")
        if key and st.button("Save key"):
            config.app["scrapecreators_api_key"] = key.strip()
            config.save_config()
            st.rerun()

    mode = st.radio("Find by", ["Keyword", "Creator", "Post URL"], horizontal=True)
    c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1])
    if mode == "Keyword":
        query = c1.text_input("Niche / keyword", placeholder="e.g. sleep tips, budgeting, skincare routine")
        when = c2.selectbox("Posted", scrapecreators.PUBLISH_TIMES, index=0)
        sort = c3.selectbox("Sort", scrapecreators.SORT_OPTIONS, index=0)
        pages = c4.number_input("Pages", 1, 5, 2, help="1 credit per page, ~5-10 carousels each")
    elif mode == "Creator":
        query = c1.text_input("TikTok handle", placeholder="@creator")
    else:
        query = c1.text_input("Carousel URL", placeholder="https://www.tiktok.com/@user/photo/123...")

    if st.button("🔎 Search", type="primary", disabled=not (query or "").strip() or not has_sc_key):
        with st.spinner("Asking TikTok…"):
            try:
                if mode == "Keyword":
                    posts = scrapecreators.search_carousels(query, when, sort, int(pages))
                elif mode == "Creator":
                    posts = scrapecreators.profile_carousels(query)
                else:
                    posts = [scrapecreators.get_carousel(query)]
                st.session_state["found_posts"] = posts
                if not posts:
                    st.warning("No photo carousels found – try a broader keyword or more pages.")
            except Exception as e:
                st.error(str(e))

    posts = st.session_state.get("found_posts") or []
    if posts:
        st.caption(
            f"{len(posts)} carousels, ranked by score = reach × (likes + 3·comments + 4·shares + 5·saves) / views. "
            "High **save rate** = people want to keep it = a format worth copying."
        )
    for post in posts:
        with st.container(border=True):
            thumbs, info = st.columns([3, 2])
            cols = thumbs.columns(5)
            for i, url in enumerate(post.image_urls[:5]):
                cols[i].image(url, width="stretch")
            info.markdown(
                f"**@{post.author}** · {post.slide_count} slides · {post.create_time}  \n"
                f"👁 {_fmt(post.plays)} · ❤️ {_fmt(post.likes)} · 🔖 {_fmt(post.saves)} · ↗️ {_fmt(post.shares)}  \n"
                f"save rate **{post.save_rate:.1%}** · share rate {post.share_rate:.1%} · score **{post.score}**"
            )
            info.caption(post.desc[:200])
            if post.url:
                info.markdown(f"[Open on TikTok]({post.url})")
            if info.button("➕ Make template", key=f"tpl-{post.id}"):
                with st.spinner("Reading slides with OCR and measuring the format…"):
                    try:
                        template = carousel_template.create_template(post, use_llm=_llm_ready())
                        st.session_state["selected_template"] = template.id
                        st.success(f"Saved template “{template.name}”. Open ② Templates or ③ Create.")
                    except Exception as e:
                        st.error(f"Could not build template: {e}")

# --------------------------------------------------------------------------- templates
with tab_templates:
    templates = carousel_template.list_templates()
    if not templates:
        st.info("No templates yet – find a winning carousel in ① and click “Make template”.")
    for t in templates:
        with st.container(border=True):
            left, right = st.columns([2, 3])
            thumbs = carousel_template.template_thumbnails(t.id)
            if thumbs:
                tcols = left.columns(min(4, len(thumbs)))
                for i, path in enumerate(thumbs[:4]):
                    tcols[i].image(path, width="stretch")
            new_name = right.text_input("Name", t.name, key=f"name-{t.id}")
            right.markdown(
                f"{t.slide_count} slides · {t.aspect.value} · 👁 {_fmt(t.plays)} · 🔖 {_fmt(t.saves)}"
                + (f" · [source]({t.source_url})" if t.source_url else "")
            )
            hook = right.text_input("Hook formula", t.hook_pattern, key=f"hook-{t.id}")
            notes = right.text_area("Format notes", t.format_notes, key=f"notes-{t.id}", height=80)
            vibe = right.text_input("Photo vibe (added to photo searches)", t.photo_vibe, key=f"vibe-{t.id}")
            with right.expander("Slide-by-slide (read by OCR)"):
                for i, s in enumerate(t.slides):
                    st.markdown(f"**{i + 1}. `{s.role}`** {s.heading}  \n{s.body}")
                st.caption(
                    f"Visual: {t.visual.style.value}, text at {t.visual.text_y:.0%} height, "
                    f"font {t.visual.font_ratio:.1%} of width, photo brightness {t.visual.bg_luminance:.0%}"
                )
            b1, b2, b3 = right.columns(3)
            if b1.button("💾 Save", key=f"save-{t.id}"):
                t.name, t.hook_pattern, t.format_notes, t.photo_vibe = new_name, hook, notes, vibe
                carousel_template.save_template(t)
                st.toast("Saved")
            if b2.button("Use →", key=f"use-{t.id}", type="primary"):
                st.session_state["selected_template"] = t.id
                st.toast("Selected – open ③ Create")
            if b3.button("🗑 Delete", key=f"del-{t.id}"):
                carousel_template.delete_template(t.id)
                st.rerun()

# --------------------------------------------------------------------------- create
with tab_create:
    templates = carousel_template.list_templates()
    options = ["(no template – generic viral format)"] + [t.id for t in templates]
    names = {t.id: f"{t.name}  ·  {t.slide_count} slides · {_fmt(t.plays)} views" for t in templates}
    selected = st.session_state.get("selected_template")
    template_id = st.selectbox(
        "Template",
        options,
        index=options.index(selected) if selected in options else 0,
        format_func=lambda o: names.get(o, o),
    )
    template = next((t for t in templates if t.id == template_id), None)
    if not _llm_ready():
        st.warning("No AI model configured yet – set the LLM provider and API key on the main page (sidebar → Main).")

    left, right = st.columns(2)
    with left:
        topic = st.text_area("Your topic", placeholder="e.g. how to stop procrastinating", height=80)
        product = st.text_input(
            "Product to plug (optional)",
            placeholder="e.g. Focusly – a focus timer app (focusly.app)",
            help="Woven in casually"
            + (f", on slide {template.plug_slide + 1} like the original" if template and template.plug_slide is not None else "")
            + ". Leave empty for pure value content.",
        )
        c1, c2 = st.columns(2)
        slide_count = c1.slider(
            "Slides", MIN_SLIDES, MAX_SLIDES, template.slide_count if template else 7, disabled=template is not None,
            help="Fixed by the template" if template else None,
        )
        language = c2.text_input("Language", placeholder="auto")
        tone = st.text_input("Extra instructions", placeholder="e.g. gen-z voice, for beginners")
    with right:
        c1, c2 = st.columns(2)
        styles = [s.value for s in SlideStyle]
        style = c1.selectbox(
            "Style", styles, index=styles.index(template.visual.style.value) if template else 0,
            format_func=lambda s: f"{s} – {STYLE_HELP[s]}",
        )
        aspects = [a.value for a in SlideAspect]
        aspect = c2.selectbox(
            "Size", aspects, index=aspects.index(template.aspect.value) if template else aspects.index("9:16"),
            format_func=lambda a: {"4:5": "4:5 · Instagram", "9:16": "9:16 · TikTok", "1:1": "1:1"}[a],
        )
        sources = [s.value for s in SlideImageSource]
        image_source = c1.selectbox("Photos", sources, format_func=lambda s: PHOTO_HELP[s])
        vibe = c2.text_input("Photo vibe", template.photo_vibe if template else "")
        handle = c1.text_input("@handle (optional)")
        accent = c2.color_picker("Accent (bold style)", "#FFD400")
        uploads = st.file_uploader(
            "Or use your own photos (slide order)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
        )
        c1, c2 = st.columns(2)
        export_video = c1.checkbox("Also export MP4 (with music)")
        seconds = c2.number_input("Sec / slide", 1.0, 15.0, 3.0, 0.5, disabled=not export_video)

    if st.button("✍️ Write slides", type="primary", disabled=not topic.strip()):
        with st.spinner("Writing in the proven format…"):
            try:
                st.session_state["slideshow_script"] = slideshow.generate_slideshow_script(
                    topic, slide_count, language, tone, template, product
                )
                st.session_state.pop("slideshow_result", None)
            except Exception as e:
                st.error(f"Writing failed: {e}")

    script: SlideshowScript | None = st.session_state.get("slideshow_script")
    if script:
        st.subheader("Edit, then render")
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
                slides=[
                    Slide(**{k: (r.get(k) or "") for k in ("heading", "body", "image_query")})
                    for r in rows
                    if (r.get("heading") or r.get("body"))
                ],
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
                template_id=template.id if template else "",
                product=product,
                script=edited,
                aspect=aspect,
                style=style,
                image_source=image_source,
                photo_vibe=vibe,
                accent_color=accent.upper(),
                handle=handle,
                export_video=export_video,
                seconds_per_slide=seconds,
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
        cols = st.columns(5)
        for i, path in enumerate(result.images):
            cols[i % 5].image(path, width="stretch")
        c1, c2 = st.columns(2)
        with open(result.zip_file, "rb") as f:
            c1.download_button("⬇️ Download slides (.zip)", f.read(), "slides.zip", "application/zip")
        if result.video_file:
            with open(result.video_file, "rb") as f:
                c2.download_button("⬇️ Download MP4", f.read(), "slideshow.mp4", "video/mp4")
            st.video(result.video_file)
        st.text_area("Caption to paste", slideshow.caption_text(result.script), height=120)
        if result.photo_credits:
            with st.expander("Photo credits (also in the zip)"):
                st.text("\n".join(result.photo_credits))
