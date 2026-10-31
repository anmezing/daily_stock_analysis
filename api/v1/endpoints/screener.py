# -*- coding: utf-8 -*-
"""Read-only screener endpoints for the Web UI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data"
_REPORTS_DIR = _PROJECT_ROOT / "reports"


@router.get("/latest")
def get_latest_screener() -> Dict[str, Any]:
    """Return the most recent screener result and its markdown report."""
    path = _DATA_DIR / "screener_latest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="还没有选股结果，请先运行 --screen")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"选股结果读取失败: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="选股结果格式异常")
    return payload


@router.get("/history")
def get_screener_history() -> Dict[str, Any]:
    """List available daily screener reports (newest first)."""
    items: List[Dict[str, str]] = []
    if _REPORTS_DIR.exists():
        for path in _REPORTS_DIR.glob("screener_*.md"):
            match = re.fullmatch(r"screener_(\d{8})\.md", path.name)
            if not match:
                continue
            raw = match.group(1)
            items.append({
                "date": f"{raw[:4]}-{raw[4:6]}-{raw[6:]}",
                "filename": path.name,
            })
    items.sort(key=lambda item: item["date"], reverse=True)
    return {"items": items}


@router.get("/report/{report_date}")
def get_screener_report(report_date: str) -> Dict[str, Any]:
    """Return one daily screener markdown report by YYYY-MM-DD."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", report_date):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    filename = f"screener_{report_date.replace('-', '')}.md"
    path = _REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="该日期没有选股报告")
    return {
        "date": report_date,
        "filename": filename,
        "markdown": path.read_text(encoding="utf-8"),
    }


@router.get("/learning")
def get_screener_learning() -> Dict[str, Any]:
    """Return the current self-learning state (samples / adjustments)."""
    path = _DATA_DIR / "screener_learning.json"
    if not path.exists():
        return {"status": "empty", "message": "还没有积累到可用的选股样本"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"学习状态读取失败: {exc}") from exc


@router.get("/alerts")
def get_screener_alerts() -> Dict[str, Any]:
    """Return post-selection alerts (stop/target/earnings/filings)."""
    path = _DATA_DIR / "screener_alerts.json"
    if not path.exists():
        return {"status": "empty", "alerts": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"选股告警读取失败: {exc}") from exc
