"""Slideshow (carousel) generation endpoints."""

from fastapi import BackgroundTasks, Depends, Request

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models import const
from app.models.exception import HttpException
from app.models.slideshow import SlideshowParams
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
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=0, kind="slideshow")
    background_tasks.add_task(slideshow_service.start, task_id, body)
    # Poll GET /api/v1/tasks/{task_id}; files are served under /tasks/{task_id}/.
    return utils.get_response(200, {"task_id": task_id})
