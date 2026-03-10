"""
token_counter.py — Know exactly how many tokens any text consumes.

WHY THIS EXISTS
---------------
LLMs have a fixed context window. To manage context intelligently,
you must count tokens BEFORE sending content to the API — not after.
Counting after the fact is too late. Once you've sent 130K tokens to
a 128K model, you'll get an error, not a graceful degradation.

HOW IT WORKS
------------
tiktoken is OpenAI's tokeniser library. Claude uses a different
tokeniser internally, but cl100k_base (GPT-4's encoding) gives a
close approximation — within ~5% for typical English text.

We add a 10% safety buffer (TOKEN_SAFETY_BUFFER) to account for
the difference. This means we'll offload slightly earlier than
strictly necessary, which is the safe direction to err.

PREREQUISITE CONCEPT: What is a token?
  A token is a chunk of text — roughly 4 characters or 0.75 words
  in English. "hello" = 1 token. "tokenisation" = 3 tokens.
  "   " (spaces) = 1 token. The model reads tokens, not characters.
"""

import sys
from src.config import TIKTOKEN_ENCODING, TOKEN_SAFETY_BUFFER, CHARS_PER_TOKEN_FALLBACK

# Load the encoding once at module level — this is an expensive disk read,
# we don't want to do it on every function call.
try:
    import tiktoken
    _encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    _USE_TIKTOKEN = True
except (ImportError, Exception):
    _USE_TIKTOKEN = False


def count_tokens(text: str) -> int:
    """
    Count tokens in a string with a safety buffer.

    Args:
        text: Any string — message content, system prompt, etc.

    Returns:
        Token count with safety buffer applied.

    Example:
        >>> count_tokens("Hello, how are you?")
        # Returns ~7 (5 real tokens + 10% buffer)
    """
    if not text:
        return 0

    if _USE_TIKTOKEN:
        base_count = len(_encoder.encode(text))
    else:
        # Fallback: ~4 characters per token (rough but better than nothing)
        base_count = max(1, len(text) // CHARS_PER_TOKEN_FALLBACK)

    # Apply safety buffer — round up, never down
    return int(base_count * TOKEN_SAFETY_BUFFER) + 1


def count_messages_tokens(messages: list[dict]) -> int:
    """
    Count total tokens across a list of message dicts.

    Each message has overhead beyond its content (~4 tokens for
    role formatting and separators in the API call).

    Args:
        messages: List of dicts with at least a "content" key.

    Returns:
        Total token estimate including per-message overhead.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        total += 4  # Per-message API overhead (role, separators)
    return total


def get_token_percentage(current: int, budget: int) -> float:
    """
    Return token usage as a fraction of budget (0.0 to 1.0+).

    Can exceed 1.0 if somehow over budget (edge case protection).

    Args:
        current: Current token count in active context.
        budget:  Total token budget for the session.

    Returns:
        Float between 0.0 and ~1.0+
    """
    if budget <= 0:
        return 1.0
    return current / budget


def is_approaching_threshold(current: int, budget: int, threshold: float) -> bool:
    """
    Check if we've hit the Pre-Rot Threshold.

    This is the key decision function for Technique 1.
    Returns True when it's time to proactively clean up context.

    Args:
        current:   Current active token count.
        budget:    Total token budget.
        threshold: Fraction at which to trigger cleanup (e.g. 0.70).

    Returns:
        True if cleanup should be triggered.

    Example:
        >>> is_approaching_threshold(72000, 100000, 0.70)
        True  # 72% >= 70% threshold
        >>> is_approaching_threshold(65000, 100000, 0.70)
        False  # 65% < 70% threshold
    """
    return get_token_percentage(current, budget) >= threshold


def tokens_remaining(current: int, budget: int, threshold: float) -> int:
    """
    How many tokens until the Pre-Rot Threshold is hit?

    Useful for the Streamlit dashboard "headroom" indicator.

    Returns:
        Tokens available before threshold triggers. 0 if already over.
    """
    threshold_tokens = int(budget * threshold)
    return max(0, threshold_tokens - current)
