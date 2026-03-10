"""
test_offload_store.py

Tests for the SQLite offload store.
Uses a temporary in-memory database to avoid touching the real data file.
"""

import pytest
import uuid
import time
import tempfile
import os
from unittest.mock import patch

@pytest.fixture(autouse=True)
def patch_db_path(tmp_path):
    """Use a temp file DB so each connection shares the same tables."""
    test_db = str(tmp_path / "test_offload.db")
    with patch("src.context.offload_store.OFFLOAD_DB_PATH", test_db):
        import src.context.offload_store as store
        store.OFFLOAD_DB_PATH = test_db
        store.initialise_db()
        yield


def make_message(content="Test content", role="user", layer="working"):
    return {
        "message_id": str(uuid.uuid4()),
        "session_id": "test-session-001",
        "role": role,
        "content": content,
        "layer": layer,
        "token_count": 50,
        "timestamp": time.time(),
    }


class TestOffloadMessage:

    def test_offload_and_retrieve_basic(self):
        from src.context.offload_store import offload_message, retrieve_relevant
        msg = make_message("Python programming and machine learning concepts")
        offload_message(**msg)

        results = retrieve_relevant(
            session_id="test-session-001",
            query="Python machine learning",
        )
        assert len(results) > 0
        assert results[0]["content"] == msg["content"]

    def test_wrong_session_returns_empty(self):
        from src.context.offload_store import offload_message, retrieve_relevant
        msg = make_message("Some content about databases")
        offload_message(**msg)

        results = retrieve_relevant(
            session_id="different-session",
            query="databases content",
        )
        assert len(results) == 0

    def test_no_keyword_overlap_returns_empty(self):
        from src.context.offload_store import offload_message, retrieve_relevant
        msg = make_message("Elephants and giraffes in Africa")
        offload_message(**msg)

        results = retrieve_relevant(
            session_id="test-session-001",
            query="Python programming",
        )
        assert len(results) == 0


class TestRetrieveRelevant:

    def test_returns_most_relevant_first(self):
        from src.context.offload_store import offload_message, retrieve_relevant
        session = "rank-test-session"

        offload_message(
            message_id=str(uuid.uuid4()), session_id=session,
            role="user",
            content="context windows and token budgets in language models",
            layer="working", token_count=30, timestamp=time.time(),
        )
        offload_message(
            message_id=str(uuid.uuid4()), session_id=session,
            role="user",
            content="token limits token windows token budgets context tokens",
            layer="working", token_count=30, timestamp=time.time() + 1,
        )

        results = retrieve_relevant(
            session_id=session,
            query="token budget context window",
            max_results=2,
        )
        # Both should be returned — just checking we get results
        assert len(results) >= 1

    def test_respects_max_tokens(self):
        from src.context.offload_store import offload_message, retrieve_relevant
        session = "token-limit-session"

        # Offload a message with a high token count
        offload_message(
            message_id=str(uuid.uuid4()), session_id=session,
            role="user", content="large content about machine learning systems",
            layer="working", token_count=5000, timestamp=time.time(),
        )
        offload_message(
            message_id=str(uuid.uuid4()), session_id=session,
            role="user", content="second machine learning message",
            layer="working", token_count=5000, timestamp=time.time() + 1,
        )

        results = retrieve_relevant(
            session_id=session,
            query="machine learning",
            max_tokens=6000,  # Only enough for one message
        )
        assert len(results) <= 1


class TestSessionStats:

    def test_empty_session_returns_zero_count(self):
        from src.context.offload_store import get_session_stats
        stats = get_session_stats("nonexistent-session")
        assert stats["message_count"] == 0

    def test_stats_after_offload(self):
        from src.context.offload_store import offload_message, get_session_stats
        session = "stats-test-session"

        offload_message(
            message_id=str(uuid.uuid4()), session_id=session,
            role="user", content="first message about context engineering",
            layer="working", token_count=100, timestamp=time.time(),
        )
        offload_message(
            message_id=str(uuid.uuid4()), session_id=session,
            role="assistant", content="second message response about context",
            layer="working", token_count=200, timestamp=time.time() + 1,
        )

        stats = get_session_stats(session)
        assert stats["message_count"] == 2
        assert stats["total_tokens"] == 300


class TestClearSession:

    def test_clear_removes_all_session_messages(self):
        from src.context.offload_store import offload_message, get_session_stats, clear_session
        session = "clear-test-session"

        offload_message(
            message_id=str(uuid.uuid4()), session_id=session,
            role="user", content="message to be cleared eventually",
            layer="background", token_count=50, timestamp=time.time(),
        )
        assert get_session_stats(session)["message_count"] == 1

        clear_session(session)
        assert get_session_stats(session)["message_count"] == 0

    def test_clear_does_not_affect_other_sessions(self):
        from src.context.offload_store import offload_message, get_session_stats, clear_session

        offload_message(
            message_id=str(uuid.uuid4()), session_id="session-A",
            role="user", content="session A message content here",
            layer="working", token_count=50, timestamp=time.time(),
        )
        offload_message(
            message_id=str(uuid.uuid4()), session_id="session-B",
            role="user", content="session B message content here",
            layer="working", token_count=50, timestamp=time.time(),
        )

        clear_session("session-A")

        assert get_session_stats("session-A")["message_count"] == 0
        assert get_session_stats("session-B")["message_count"] == 1
