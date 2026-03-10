"""
test_token_counter.py

Tests for the token counting module.
These are deterministic — no API calls, no external dependencies.
"""

import pytest
from src.context.token_counter import (
    count_tokens,
    count_messages_tokens,
    get_token_percentage,
    is_approaching_threshold,
    tokens_remaining,
)


class TestCountTokens:

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_none_like_empty_returns_zero(self):
        # Edge case: falsy input
        assert count_tokens("") == 0

    def test_short_string_returns_positive(self):
        result = count_tokens("Hello")
        assert result > 0

    def test_longer_text_returns_more_tokens(self):
        short = count_tokens("Hi")
        long  = count_tokens("This is a much longer sentence with many more words in it.")
        assert long > short

    def test_includes_safety_buffer(self):
        # Count should be at least 10% higher than raw character estimate
        text = "The quick brown fox jumps over the lazy dog."
        result = count_tokens(text)
        raw_estimate = len(text) // 4
        # With 10% buffer, result should exceed raw estimate
        assert result > raw_estimate


class TestCountMessagesTokens:

    def test_empty_list_returns_zero(self):
        assert count_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello there"}]
        result = count_messages_tokens(msgs)
        assert result > 0

    def test_multiple_messages_more_than_single(self):
        single = count_messages_tokens([{"role": "user", "content": "Hello"}])
        multi  = count_messages_tokens([
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
            {"role": "user",      "content": "Tell me about tokens."},
        ])
        assert multi > single

    def test_includes_per_message_overhead(self):
        # Each message adds 4 tokens overhead; 3 messages = 12 extra tokens minimum
        msgs = [
            {"role": "user", "content": ""},
            {"role": "user", "content": ""},
            {"role": "user", "content": ""},
        ]
        result = count_messages_tokens(msgs)
        assert result >= 12  # 3 messages * 4 overhead each


class TestGetTokenPercentage:

    def test_zero_current_returns_zero(self):
        assert get_token_percentage(0, 100_000) == 0.0

    def test_full_budget_returns_one(self):
        assert get_token_percentage(100_000, 100_000) == 1.0

    def test_half_budget_returns_half(self):
        assert get_token_percentage(50_000, 100_000) == pytest.approx(0.5)

    def test_over_budget_returns_above_one(self):
        result = get_token_percentage(110_000, 100_000)
        assert result > 1.0

    def test_zero_budget_returns_one(self):
        # Edge case: zero budget means we're at capacity
        assert get_token_percentage(1000, 0) == 1.0


class TestIsApproachingThreshold:

    def test_below_threshold_returns_false(self):
        assert is_approaching_threshold(60_000, 100_000, 0.70) is False

    def test_at_threshold_returns_true(self):
        assert is_approaching_threshold(70_000, 100_000, 0.70) is True

    def test_above_threshold_returns_true(self):
        assert is_approaching_threshold(85_000, 100_000, 0.70) is True

    def test_zero_tokens_returns_false(self):
        assert is_approaching_threshold(0, 100_000, 0.70) is False

    def test_high_threshold_not_triggered(self):
        # Threshold of 95% — at 70% should not trigger
        assert is_approaching_threshold(70_000, 100_000, 0.95) is False


class TestTokensRemaining:

    def test_below_threshold_returns_headroom(self):
        result = tokens_remaining(60_000, 100_000, 0.70)
        assert result == 10_000  # 70_000 - 60_000

    def test_at_threshold_returns_zero(self):
        result = tokens_remaining(70_000, 100_000, 0.70)
        assert result == 0

    def test_above_threshold_returns_zero_not_negative(self):
        result = tokens_remaining(80_000, 100_000, 0.70)
        assert result == 0  # Clamped to 0

    def test_empty_context_returns_full_headroom(self):
        result = tokens_remaining(0, 100_000, 0.70)
        assert result == 70_000
