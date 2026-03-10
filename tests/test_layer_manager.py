"""
test_layer_manager.py

Tests for the Layered Action Space classifier and offload candidate selector.
All deterministic — no API calls.
"""

import pytest
import time
import uuid
from src.context.layer_manager import (
    ContextLayer,
    classify_layer,
    get_offload_candidates,
    layer_summary,
)


class TestClassifyLayer:

    def test_system_messages_are_always_critical(self):
        result = classify_layer("You are a helpful assistant.", role="system")
        assert result == ContextLayer.CRITICAL

    def test_remember_keyword_triggers_critical(self):
        result = classify_layer("Please remember my name is Ojey.", role="user")
        assert result == ContextLayer.CRITICAL

    def test_important_keyword_triggers_critical(self):
        result = classify_layer("This is important: never share personal data.", role="user")
        assert result == ContextLayer.CRITICAL

    def test_must_keyword_triggers_critical(self):
        result = classify_layer("You must always respond in English.", role="user")
        assert result == ContextLayer.CRITICAL

    def test_very_short_message_is_background(self):
        result = classify_layer("Okay.", role="user")
        assert result == ContextLayer.BACKGROUND

    def test_short_message_under_15_words_is_background(self):
        result = classify_layer("Got it, thanks.", role="user")
        assert result == ContextLayer.BACKGROUND

    def test_by_the_way_triggers_background(self):
        result = classify_layer(
            "By the way, I was also thinking about something else entirely unrelated.",
            role="user"
        )
        assert result == ContextLayer.BACKGROUND

    def test_normal_message_is_working(self):
        result = classify_layer(
            "Can you explain how transformer attention mechanisms work in neural networks?",
            role="user"
        )
        assert result == ContextLayer.WORKING

    def test_long_assistant_response_is_working(self):
        result = classify_layer(
            "Transformer attention mechanisms work by computing query, key, and value "
            "matrices from the input embeddings. The attention score between positions "
            "is calculated using the dot product of queries and keys.",
            role="assistant"
        )
        assert result == ContextLayer.WORKING


class TestGetOffloadCandidates:

    def _make_msg(self, layer, token_count=100, timestamp_offset=0):
        return {
            "message_id":  str(uuid.uuid4()),
            "role":        "user",
            "content":     "test content",
            "layer":       layer,
            "token_count": token_count,
            "timestamp":   time.time() + timestamp_offset,
        }

    def test_critical_messages_never_selected(self):
        messages = [
            self._make_msg(ContextLayer.CRITICAL.value, 500),
            self._make_msg(ContextLayer.CRITICAL.value, 500),
        ]
        candidates = get_offload_candidates(messages, tokens_to_free=800)
        assert len(candidates) == 0

    def test_background_selected_before_working(self):
        bg_msg = self._make_msg(ContextLayer.BACKGROUND.value, 200, timestamp_offset=0)
        wk_msg = self._make_msg(ContextLayer.WORKING.value,    200, timestamp_offset=1)

        candidates = get_offload_candidates([bg_msg, wk_msg], tokens_to_free=150)
        # Should select background message, not working
        assert bg_msg["message_id"] in candidates
        assert wk_msg["message_id"] not in candidates

    def test_stops_when_enough_tokens_freed(self):
        messages = [
            self._make_msg(ContextLayer.BACKGROUND.value, 300, timestamp_offset=0),
            self._make_msg(ContextLayer.BACKGROUND.value, 300, timestamp_offset=1),
            self._make_msg(ContextLayer.BACKGROUND.value, 300, timestamp_offset=2),
        ]
        # Only need to free 350 tokens — should select at most 2 messages
        candidates = get_offload_candidates(messages, tokens_to_free=350)
        assert len(candidates) <= 2

    def test_returns_empty_when_nothing_eligible(self):
        messages = [
            self._make_msg(ContextLayer.CRITICAL.value, 100),
        ]
        candidates = get_offload_candidates(messages, tokens_to_free=50)
        assert candidates == []

    def test_oldest_messages_selected_first(self):
        old_msg = self._make_msg(ContextLayer.BACKGROUND.value, 100, timestamp_offset=0)
        new_msg = self._make_msg(ContextLayer.BACKGROUND.value, 100, timestamp_offset=10)

        candidates = get_offload_candidates([old_msg, new_msg], tokens_to_free=80)
        # Should prefer the older message
        assert old_msg["message_id"] in candidates


class TestLayerSummary:

    def test_empty_messages_returns_zero_counts(self):
        result = layer_summary([])
        assert result["critical"]["count"] == 0
        assert result["working"]["count"] == 0
        assert result["background"]["count"] == 0

    def test_counts_by_layer(self):
        messages = [
            {"layer": "critical",   "token_count": 100},
            {"layer": "working",    "token_count": 200},
            {"layer": "working",    "token_count": 150},
            {"layer": "background", "token_count": 50},
        ]
        result = layer_summary(messages)

        assert result["critical"]["count"]   == 1
        assert result["working"]["count"]    == 2
        assert result["background"]["count"] == 1

    def test_token_totals_per_layer(self):
        messages = [
            {"layer": "critical",  "token_count": 300},
            {"layer": "working",   "token_count": 200},
            {"layer": "working",   "token_count": 100},
        ]
        result = layer_summary(messages)

        assert result["critical"]["tokens"]  == 300
        assert result["working"]["tokens"]   == 300
        assert result["background"]["tokens"] == 0
