"""GET /health —— 健康检查。

用途：确认服务活着。前端启动时先调这个，失败就提示"后端没启动"。
这是 W1 的最小可交付物，也是最简单的接口。
"""

from datetime import datetime

from fastapi import APIRouter

from app.config import APP_VERSION
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="返回服务状态、版本号和当前时间。用来确认后端已启动。",
)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        time=datetime.now(),
    )
