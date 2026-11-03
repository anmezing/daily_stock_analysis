# -*- coding: utf-8 -*-
"""Intraday volume helpers.

The realtime quote often has the current cumulative volume but no volume
ratio.  This module turns those two numbers into a time-normalised ratio so
the prompt does not show "量能未知" during the whole session.
"""

from __future__ import annotations

from typing import Optional


def normalize_intraday_volume_ratio(
    current_volume: Optional[float],
    avg_daily_volume: Optional[float],
    elapsed_minutes: Optional[int],
    total_minutes: int = 390,
) -> Optional[float]:
    try:
        volume = float(current_volume or 0)
        average = float(avg_daily_volume or 0)
    except (TypeError, ValueError):
        return None
    if volume <= 0 or average <= 0 or total_minutes <= 0:
        return None

    if elapsed_minutes and 0 < int(elapsed_minutes) < total_minutes:
        expected = average * (float(elapsed_minutes) / float(total_minutes))
    else:
        expected = average
    if expected <= 0:
        return None

    ratio = volume / expected
    if ratio <= 0.02 or ratio >= 20.0:
        return None
    return round(ratio, 2)
