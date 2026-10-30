# -*- coding: utf-8 -*-
"""
===================================
定时调度模块
===================================

职责：
1. 支持每日定时执行股票分析
2. 支持定时执行大盘复盘
3. 优雅处理信号，确保可靠退出

依赖：
- schedule: 轻量级定时任务库
"""

import json
import logging
import os
import re
import signal
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# 补跑窗口：容器在计划时间之后重启时，只要还在窗口内就补跑一次，
# 避免"重启一次就整天没分析"。
CATCH_UP_WINDOW_HOURS = 6
# 记录最近一次成功执行的日期（本地时区），用于判断是否需要补跑
SCHEDULE_STATE_PATH = os.getenv("SCHEDULE_STATE_PATH") or os.path.join("data", ".schedule_state.json")


def _state_file_path() -> Path:
    return Path(SCHEDULE_STATE_PATH)


def load_last_run_date() -> Optional[str]:
    """读取上次成功执行的日期（ISO 格式）；文件损坏/不存在时返回 None。"""
    try:
        raw = _state_file_path().read_text(encoding="utf-8")
        payload = json.loads(raw)
        value = str(payload.get("last_run_date") or "").strip()
        return value or None
    except Exception:
        return None


def mark_run_completed(run_date: Optional[str] = None) -> None:
    """记录一次成功执行；失败不影响主流程。"""
    target = run_date or datetime.now().date().isoformat()
    try:
        path = _state_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"last_run_date": target, "updated_at": datetime.now().isoformat()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - 磁盘异常不阻断分析
        logger.warning("[Scheduler] 记录执行状态失败: %s", exc)


def should_catch_up(
    now: datetime,
    schedule_time: str,
    last_run_date: Optional[str],
    window_hours: int = CATCH_UP_WINDOW_HOURS,
) -> bool:
    """判断启动时是否需要补跑当天错过的那一轮。"""
    if last_run_date == now.date().isoformat():
        return False
    match = re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", (schedule_time or "").strip())
    if not match:
        return False
    hour, minute = (int(part) for part in schedule_time.strip().split(":"))
    scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now < scheduled:
        return False
    return now - scheduled <= timedelta(hours=window_hours)


class GracefulShutdown:
    """
    优雅退出处理器
    
    捕获 SIGTERM/SIGINT 信号，确保任务完成后再退出
    """
    
    def __init__(self):
        self.shutdown_requested = False
        self._lock = threading.Lock()
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        with self._lock:
            if not self.shutdown_requested:
                logger.info(f"收到退出信号 ({signum})，等待当前任务完成...")
                self.shutdown_requested = True
    
    @property
    def should_shutdown(self) -> bool:
        """检查是否应该退出"""
        with self._lock:
            return self.shutdown_requested


class Scheduler:
    """
    定时任务调度器
    
    基于 schedule 库实现，支持：
    - 每日定时执行
    - 启动时立即执行
    - 优雅退出
    """
    
    def __init__(
        self,
        schedule_time: str = "18:00",
        schedule_time_provider: Optional[Callable[[], str]] = None,
    ):
        """
        初始化调度器

        Args:
            schedule_time: 每日执行时间，格式 "HH:MM"
            schedule_time_provider: 可选回调，运行期间定期调用以检测时间变更
        """
        try:
            import schedule
            self.schedule = schedule
        except ImportError:
            logger.error("schedule 库未安装，请执行: pip install schedule")
            raise ImportError("请安装 schedule 库: pip install schedule")

        self.schedule_time = schedule_time
        self._schedule_time_provider = schedule_time_provider
        self.shutdown_handler = GracefulShutdown()
        self._task_callback: Optional[Callable] = None
        self._daily_job: Optional[Any] = None
        self._running = False
        
    def set_daily_task(self, task: Callable, run_immediately: bool = True):
        """
        设置每日定时任务

        Args:
            task: 要执行的任务函数（无参数）
            run_immediately: 是否在设置后立即执行一次
        """
        self._task_callback = task
        if not self._configure_daily_task(self.schedule_time):
            raise ValueError(f"无效的定时执行时间: {self.schedule_time!r}")

        if run_immediately:
            logger.info("立即执行一次任务...")
            self._safe_run_task()

    @staticmethod
    def _is_valid_schedule_time(schedule_time: str) -> bool:
        """Validate time string in HH:MM 24-hour format."""
        candidate = (schedule_time or "").strip()
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", candidate):
            return False
        return True

    def _cancel_daily_job(self) -> None:
        """Remove the currently registered daily job if one exists."""
        if self._daily_job is None:
            return

        if hasattr(self.schedule, "cancel_job"):
            self.schedule.cancel_job(self._daily_job)
        else:  # pragma: no cover - compatibility fallback
            jobs = getattr(self.schedule, "jobs", None)
            if isinstance(jobs, list) and self._daily_job in jobs:
                jobs.remove(self._daily_job)

        self._daily_job = None

    def _configure_daily_task(self, schedule_time: str) -> bool:
        """(Re)register the daily job at the requested time."""
        candidate = (schedule_time or "").strip()
        if not self._is_valid_schedule_time(candidate):
            logger.warning(
                "检测到无效的定时执行时间 %r，继续沿用当前时间 %s",
                schedule_time,
                self.schedule_time,
            )
            return False

        previous_time = self.schedule_time
        self._cancel_daily_job()
        self._daily_job = self.schedule.every().day.at(candidate).do(self._safe_run_task)
        self.schedule_time = candidate

        if previous_time == candidate:
            logger.info("已设置每日定时任务，执行时间: %s", self.schedule_time)
        else:
            logger.info(
                "检测到 SCHEDULE_TIME 变更，已将每日定时任务从 %s 更新为 %s",
                previous_time,
                self.schedule_time,
            )
        return True

    def _refresh_daily_schedule_if_needed(self) -> None:
        """Reload daily schedule time from the latest runtime config if needed."""
        if self._task_callback is None or self._schedule_time_provider is None:
            return

        try:
            latest_schedule_time = (self._schedule_time_provider() or "").strip()
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.warning("读取最新 SCHEDULE_TIME 失败，继续沿用 %s: %s", self.schedule_time, exc)
            return

        if not latest_schedule_time or latest_schedule_time == self.schedule_time:
            return

        if self._configure_daily_task(latest_schedule_time):
            logger.info("更新后的下次执行时间: %s", self._get_next_run_time())
    
    def _safe_run_task(self):
        """安全执行任务（带异常捕获）"""
        if self._task_callback is None:
            return
        
        try:
            logger.info("=" * 50)
            logger.info(f"定时任务开始执行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 50)
            
            self._task_callback()
            
            logger.info(f"定时任务执行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.exception(f"定时任务执行失败: {e}")
    
    def run(self):
        """
        运行调度器主循环

        阻塞运行，直到收到退出信号
        """
        self._running = True
        logger.info("调度器开始运行...")
        logger.info(f"下次执行时间: {self._get_next_run_time()}")

        while self._running and not self.shutdown_handler.should_shutdown:
            self._refresh_daily_schedule_if_needed()
            self.schedule.run_pending()
            time.sleep(30)  # 每30秒检查一次

            # 每小时打印一次心跳
            if datetime.now().minute == 0 and datetime.now().second < 30:
                logger.info(f"调度器运行中... 下次执行: {self._get_next_run_time()}")

        logger.info("调度器已停止")
    
    def _get_next_run_time(self) -> str:
        """获取下次执行时间"""
        jobs = self.schedule.get_jobs()
        if jobs:
            next_run = min(job.next_run for job in jobs)
            return next_run.strftime('%Y-%m-%d %H:%M:%S')
        return "未设置"
    
    def stop(self):
        """停止调度器"""
        self._running = False


def run_with_schedule(
    task: Callable,
    schedule_time: str = "18:00",
    run_immediately: bool = True,
    background_tasks: Optional[List[Dict[str, Any]]] = None,
    schedule_time_provider: Optional[Callable[[], str]] = None,
):
    """
    便捷函数：使用定时调度运行任务

    Args:
        task: 要执行的任务函数
        schedule_time: 每日执行时间
        run_immediately: 是否立即执行一次
        background_tasks: 后台周期任务列表，每项包含:
            - task: Callable  周期执行的函数
            - interval_seconds: int  执行间隔（秒）
            - run_immediately: bool  是否立即执行一次
            - name: str  任务名称（用于日志）
        schedule_time_provider: 可选回调，运行期间定期调用以检测定时时间变更
    """
    scheduler = Scheduler(
        schedule_time=schedule_time,
        schedule_time_provider=schedule_time_provider,
    )

    def _task_with_state() -> None:
        """执行分析并在成功后记录日期，供补跑判断使用。"""
        task()
        mark_run_completed()

    catch_up = False
    if not run_immediately:
        last_run = load_last_run_date()
        catch_up = should_catch_up(datetime.now(), schedule_time, last_run)
        if catch_up:
            logger.warning(
                "[Scheduler] 检测到今天 %s 的计划任务尚未执行（上次成功: %s），立即补跑一次",
                schedule_time,
                last_run or "无记录",
            )

    scheduler.set_daily_task(_task_with_state, run_immediately=run_immediately or catch_up)

    # 注册后台周期任务
    for bg in (background_tasks or []):
        bg_task = bg["task"]
        interval = bg.get("interval_seconds", 300)
        name = bg.get("name", "background")
        should_run_now = bg.get("run_immediately", False)

        def _make_safe_bg(fn: Callable, label: str) -> Callable:
            """Wrap background task with exception guard."""
            def _safe():
                try:
                    fn()
                except Exception as exc:
                    logger.error("[Scheduler] 后台任务 %s 执行失败: %s", label, exc)
            return _safe

        safe_fn = _make_safe_bg(bg_task, name)
        scheduler.schedule.every(interval).seconds.do(safe_fn)
        logger.info("[Scheduler] 已注册后台任务: %s (间隔 %ds)", name, interval)

        if should_run_now:
            logger.info("[Scheduler] 立即执行后台任务: %s", name)
            safe_fn()

    scheduler.run()


if __name__ == "__main__":
    # 测试定时调度
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    def test_task():
        print(f"任务执行中... {datetime.now()}")
        time.sleep(2)
        print("任务完成!")
    
    print("启动测试调度器（按 Ctrl+C 退出）")
    run_with_schedule(test_task, schedule_time="23:59", run_immediately=True)
