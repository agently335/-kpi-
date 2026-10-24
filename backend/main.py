"""FastAPI 应用入口。

启动：
    cd backend
    python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

打开 http://127.0.0.1:8000/docs 看接口文档。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_VERSION, ensure_dirs
from app.routers import health

ensure_dirs()

app = FastAPI(
    title="KPI 故障诊断 API",
    description=(
        "教学项目：上传 KPI CSV → 训练基础分类器 → 批量预测故障类别。\n\n"
        "**故障类别**：0 正常 / 1 CPU 饱和 / 2 内存泄漏 / 3 磁盘 IO 瓶颈 / "
        "4 网络异常 / 5 服务异常"
    ),
    version=APP_VERSION,
)

# 开发阶段允许前端跨域访问（本地 5173 是 Vite 默认端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    """根路径给人看，指向文档。"""
    return {
        "service": "KPI 故障诊断 API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
