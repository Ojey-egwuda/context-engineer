"""
test_boundaries.py

Boundary tests at 64K, 100K, and 128K+ token thresholds.

WHY BOUNDARY TESTS MATTER
--------------------------
Most AI projects are tested at comfortable mid-range values.
Systems break at the edges:
  - Just below the threshold (should NOT offload)
  - At exactly the threshold (should offload)
  - Just above (should have already offloaded)
  - Extreme overload (all messages are CRITICAL — can't offload)

These tests prove the system degrades gracefully, not silently.
A CV bullet that says "validated at 64K, 100K and 128K token
boundaries" is a concrete signal of production-grade thinking.

All tests are deterministic — no API calls.
"""

import pytest
import time
import uuid

from src.context.token_counter import is_approaching_threshold, get_token_percentage
from src.context.layer_manager import ContextLayer, get_offload_candidates
from src.context.context_manager import build_context_window


def make_msg(layer, tokens, content="content", role="user"):
    return {
        "message_id":  str(uuid.uuid4()),
        "role":        role,
        "content":     content,
        "layer":       layer,
        "token_count": tokens,
        "timestamp":   time.time(),
    }


# ── 64K Boundary ─────────────────────────────────────────────────────────────

class TestBoundary64K:
    BUDGET    = 100_000
    THRESHOLD = 0.70  # 70K threshold

    def test_at_63k_does_not_trigger_offload(self):
        """Just below 64K — should not trigger at 70% threshold."""
        assert is_approaching_threshold(63_000, self.BUDGET, self.THRESHOLD) is False

    def test_at_64k_does_not_trigger_offload(self):
        """64K is 64% — below 70% threshold."""
        assert is_approaching_threshold(64_000, self.BUDGET, self.THRESHOLD) is False

    def test_at_70k_triggers_offload(self):
        """Exactly at threshold."""
        assert is_approaching_threshold(70_000, self.BUDGET, self.THRESHOLD) is True

    def test_offload_candidate_selection_at_64k_load(self):
        """With 64K token load, selecting candidates works correctly."""
        messages = [
            make_msg(ContextLayer.BACKGROUND.value, 10_000),
            make_msg(ContextLayer.WORKING.value,    20_000),
            make_msg(ContextLayer.CRITICAL.value,   34_000),
        ]
        # Need to free 30% of 100K = 30K tokens
        candidates = get_offload_candidates(messages, tokens_to_free=30_000)
        # Should offload background (10K) + working (20K) = 30K. Critical stays.
        assert len(candidates) == 2

    def test_build_context_window_at_64k_excludes_background(self):
        """Context window builder excludes background messages."""
        messages = [
            make_msg(ContextLayer.CRITICAL.value,   5_000, content="system prompt"),
            make_msg(ContextLayer.WORKING.value,    20_000, content="current task"),
            make_msg(ContextLayer.BACKGROUND.value, 39_000, content="old chat"),
        ]
        window = build_context_window(messages)
        # Background should be excluded
        for msg in window:
            assert msg.get("layer") != ContextLayer.BACKGROUND.value


# ── 100K Boundary ─────────────────────────────────────────────────────────────

class TestBoundary100K:
    BUDGET    = 100_000
    THRESHOLD = 0.70

    def test_at_99k_triggers_offload(self):
        """Well above threshold — definitely should trigger."""
        assert is_approaching_threshold(99_000, self.BUDGET, self.THRESHOLD) is True

    def test_at_100k_triggers_offload(self):
        """At full budget."""
        assert is_approaching_threshold(100_000, self.BUDGET, self.THRESHOLD) is True

    def test_pct_at_full_budget(self):
        pct = get_token_percentage(100_000, 100_000)
        assert pct == pytest.approx(1.0)

    def test_offload_with_only_critical_messages_returns_empty(self):
        """
        If ALL messages are critical, offloading returns no candidates.
        This is the worst case — we cannot free space but we also
        cannot lose critical information. The system must handle this
        gracefully (not crash, just proceed with full context).
        """
        messages = [
            make_msg(ContextLayer.CRITICAL.value, 30_000),
            make_msg(ContextLayer.CRITICAL.value, 30_000),
            make_msg(ContextLayer.CRITICAL.value, 40_000),
        ]
        candidates = get_offload_candidates(messages, tokens_to_free=30_000)
        assert candidates == []

    def test_context_window_built_with_only_critical_messages(self):
        """Even at full budget, context window builds without crashing."""
        messages = [
            make_msg(ContextLayer.CRITICAL.value, 50_000, content="critical system info"),
            make_msg(ContextLayer.CRITICAL.value, 50_000, content="more critical info"),
        ]
        # Should not raise
        window = build_context_window(messages)
        assert len(window) == 2


# ── 128K Boundary (Beyond Typical Limits) ────────────────────────────────────

class TestBoundary128K:
    BUDGET    = 100_000
    THRESHOLD = 0.70

    def test_at_128k_triggers_offload(self):
        """128K tokens in a 100K budget — severely over limit."""
        assert is_approaching_threshold(128_000, self.BUDGET, self.THRESHOLD) is True

    def test_percentage_above_100_percent(self):
        """System handles over-budget gracefully (no crash)."""
        pct = get_token_percentage(128_000, 100_000)
        assert pct == pytest.approx(1.28)
        assert pct > 1.0

    def test_offload_candidates_at_extreme_overload(self):
        """
        At 128K+ effective load, offloading selects all non-critical messages.
        """
        messages = [
            make_msg(ContextLayer.CRITICAL.value,   20_000),
            make_msg(ContextLayer.WORKING.value,    40_000),
            make_msg(ContextLayer.WORKING.value,    40_000),
            make_msg(ContextLayer.BACKGROUND.value, 28_000),
        ]
        # Need to free 50K (50% of budget)
        candidates = get_offload_candidates(messages, tokens_to_free=50_000)
        # Background (28K) + first Working (40K) = 68K >= 50K needed
        assert len(candidates) >= 2

    def test_critical_still_not_selected_at_extreme_overload(self):
        """Critical messages are NEVER offloaded, even at extreme overload."""
        messages = [
            make_msg(ContextLayer.CRITICAL.value,   60_000),
            make_msg(ContextLayer.BACKGROUND.value, 68_000),
        ]
        candidates = get_offload_candidates(messages, tokens_to_free=128_000)
        # Only the background message should be a candidate
        background_id = messages[1]["message_id"]
        assert background_id in candidates
        # Critical message should NOT be a candidate
        critical_id = messages[0]["message_id"]
        assert critical_id not in candidates

    def test_context_window_built_at_128k_does_not_crash(self):
        """
        Even with an impossibly large context load, build_context_window
        should not raise an exception. It builds with what it has.
        """
        messages = [
            make_msg(ContextLayer.CRITICAL.value, 128_000, content="massive critical block"),
        ]
        # This should not raise
        try:
            window = build_context_window(messages)
            assert len(window) >= 0
        except Exception as e:
            pytest.fail(f"build_context_window raised unexpectedly: {e}")
