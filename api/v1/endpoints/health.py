# -*- coding: utf-8 -*-
"""
===================================
健康检查接口
===================================

职责：
1. 提供 /api/v1/health 健康检查接口
2. 用于负载均衡器、Docker 健康检查和监控系统

数据库不可用时返回 503，避免"进程活着但分析必然失败"却被判为健康。
"""

from datetime import datetime

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from api.v1.schemas.common import HealthResponse
from src.storage import check_database_connection

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    健康检查接口

    同时探测进程与数据库：数据库失败时返回 503 且 status=degraded，
    便于 Docker / 监控及时发现"服务起来了但库连不上"的状态。

    Returns:
        HealthResponse: 包含服务状态、时间戳和数据库状态
    """
    db_ok, db_error = await run_in_threadpool(check_database_connection)
    payload = HealthResponse(
        status="ok" if db_ok else "degraded",
        timestamp=datetime.now().isoformat(),
        database="ok" if db_ok else db_error,
    )
    if db_ok:
        return payload
    return JSONResponse(status_code=503, content=payload.model_dump())
