# -*- coding: utf-8 -*-
"""One-sentence fallback must not cut mid-sentence."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.orchestrator import _first_complete_sentence


class FirstCompleteSentenceTest(unittest.TestCase):
    def test_uses_first_complete_sentence(self) -> None:
        text = "空头排列，先观望。等站回 MA20 再考虑。"
        self.assertEqual(_first_complete_sentence(text), "空头排列，先观望。")

    def test_keeps_long_unpunctuated_text(self) -> None:
        text = "空仓者耐心观望等待价格重新站稳MA20并缩量回踩不破再考虑介入"
        self.assertEqual(_first_complete_sentence(text), text)

    def test_empty_text(self) -> None:
        self.assertEqual(_first_complete_sentence(""), "")


if __name__ == "__main__":
    unittest.main()
