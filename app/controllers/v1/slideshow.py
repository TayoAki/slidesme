"""Slideshow (carousel) generation endpoints."""

from fastapi import BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models import const
from app.models.exception import HttpException
from app.models.slideshow import SlideshowParams
from app.services import carousel_template, scrapecreators
from app.services import slideshow as slideshow_service
from app.services import state as sm
from app.utils import file_security, utils

router = new_router(dependencies=[Depends(base.verify_token)])


@router.post("/slideshows", summary="Generate a social-media slideshow (carousel)")
def create_slideshow(
    background_tasks: BackgroundTasks, request: Request, body: SlideshowParams
):
    task_id = utils.get_uuid()
    request_id = base.get_task_id(request)
    if body.font_name:
        try:
            file_security.resolve_path_within_directory(utils.font_dir(), body.font_name)
        except ValueError as e:
            raise HttpException(
                task_id=task_id, status_code=400, message=f"{request_id}: invalid font_name: {e}"
            )
    if body.template_id:
        try:
            carousel_template.load_template(body.template_id)
        except (ValueError, FileNotFoundError) as e:
            raise HttpException(task_id=task_id, status_code=400, message=f"{request_id}: {e}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=0, kind="slideshow")
    background_tasks.add_task(slideshow_service.start, task_id, body)
    # Poll GET /api/v1/tasks/{task_id}; files are served under /tasks/{task_id}/.
    return utils.get_response(200, {"task_id": task_id})


@router.get("/carousels/search", summary="Find TikTok photo carousels that perform (Scrape Creators)")
def search_carousels(
    request: Request,
    query: str = "",
    handle: str = "",
    publish_time: str = "all-time",
    sort_by: str = "relevance",
    pages: int = 1,
):
    request_id = base.get_task_id(request)
    try:
        if handle:
            posts = scrapecreators.profile_carousels(handle)
        else:
            posts = scrapecreators.search_carousels(query, publish_time, sort_by, pages)
    except (ValueError, scrapecreators.ScrapeCreatorsError) as e:
        raise HttpException(task_id=request_id, status_code=400, message=str(e))
    return utils.get_response(
        200,
        {"carousels": [p.model_dump() | {"score": p.score, "save_rate": p.save_rate} for p in posts]},
    )


class TemplateFromUrlRequest(BaseModel):
    url: str = Field(..., max_length=500)
    name: str = Field("", max_length=100)


@router.post("/templates", summary="Create a reusable template from a TikTok carousel URL")
def create_template(request: Request, body: TemplateFromUrlRequest):
    request_id = base.get_task_id(request)
    try:
        post = scrapecreators.get_carousel(body.url)
        template = carousel_template.create_template(post, name=body.name)
    except (ValueError, scrapecreators.ScrapeCreatorsError) as e:
        raise HttpException(task_id=request_id, status_code=400, message=str(e))
    return utils.get_response(200, template.model_dump())


@router.get("/templates", summary="List saved carousel templates")
def list_templates():
    return utils.get_response(200, {"templates": [t.model_dump() for t in carousel_template.list_templates()]})


@router.delete("/templates/{template_id}", summary="Delete a carousel template")
def delete_template(request: Request, template_id: str):
    try:
        carousel_template.delete_template(template_id)
    except (ValueError, FileNotFoundError) as e:
        raise HttpException(task_id=base.get_task_id(request), status_code=404, message=str(e))
    return utils.get_response(200, {"deleted": template_id})
