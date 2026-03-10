"""
context_manager.py — Ties all context engineering techniques together.

WHY THIS EXISTS
---------------
Individual techniques (token counting, layering, offloading, retrieval)
are powerful on their own. This module provides the functions that
COMBINE them — specifically for building the optimal context window
to send to Claude at each step.

Think of it as the scheduler in an operating system:
  - It knows what's in RAM (active context)
  - It knows what's on disk (offload store)
  - It decides what to load, what to evict, and what to compress

KEY FUNCTION: build_context_window()
  This is called by reason_node before every Claude API call.
  It assembles the best possible context window from:
    1. CRITICAL messages (always included)
    2. Retrieved memory (compressed, injected as a system note)
    3. WORKING messages (current task context)
  BACKGROUND messages are excluded — they've been offloaded.

TECHNIQUE 7: Retrieval-Augmented Compression
  When we retrieve messages from the offload store, we don't just
  paste them in verbatim. We compress them to their most information-
  dense form first. This maximises how much past context we can
  re-inject within a fixed token budget.
"""

import time
from src.context.token_counter import count_tokens
from src.context.layer_manager import ContextLayer
from src.context.offload_store import initialise_db

# Initialise DB on module load — safe, idempotent
initialise_db()


def build_context_window(
    messages: list[dict],
    retrieved_context: str = "",
) -> list[dict]:
    """
    Assemble the optimised context window for a Claude API call.

    Structure (in order):
      1. CRITICAL messages  — Always present, never removed
      2. Retrieved context  — Injected as a system note (if any)
      3. WORKING messages   — Current task conversation

    BACKGROUND messages are excluded — they've been offloaded.

    Args:
        messages:          The full active message list from agent state.
        retrieved_context: Compressed text from the offload store.

    Returns:
        Ordered list of message dicts ready for the Anthropic API.
    """
    critical_msgs = [
        m for m in messages
        if m.get("layer") == ContextLayer.CRITICAL.value
        and m.get("message_id") not in ("retrieved", "scratchpad")
    ]
    working_msgs = [
        m for m in messages
        if m.get("layer") == ContextLayer.WORKING.value
    ]

    context: list[dict] = []

    # Layer 1: Critical (always first — the model sees these before anything)
    context.extend(critical_msgs)

    # Layer 2: Retrieved memory (injected as a system note with clear markers)
    if retrieved_context:
        context.append({
            "role": "system",
            "content": (
                "[RETRIEVED FROM LONG-TERM MEMORY]\n"
                "The following was retrieved from past conversation that "
                "was offloaded to storage. Use it if relevant:\n\n"
                f"{retrieved_context}\n"
                "[END RETRIEVED MEMORY]"
            ),
            "layer": ContextLayer.CRITICAL.value,
            "token_count": count_tokens(retrieved_context),
            "message_id": "retrieved",
            "timestamp": time.time(),
        })

    # Layer 3: Working context (most recent task-relevant messages)
    context.extend(working_msgs)

    return context


def compress_retrieved(
    retrieved_messages: list[dict],
    max_tokens: int = 1500,
) -> str:
    """
    Technique 7: Retrieval-Augmented Compression.

    Takes retrieved messages and compresses them into a concise
    summary that fits within max_tokens.

    WHY COMPRESS?
    Because retrieved messages come with their full original token
    count. If we injected them verbatim, we might spend 3000 tokens
    on retrieved context alone. Compressing to 1500 tokens halves
    the retrieval cost while keeping the essential information.

    Current approach: extract first 300 chars per message (simple).
    Production approach: use LLM to summarise the retrieved set.

    Args:
        retrieved_messages: Messages from retrieve_relevant().
        max_tokens:         Token budget for the compressed output.

    Returns:
        Compressed string ready to inject into the context window.
    """
    if not retrieved_messages:
        return ""

    lines = []
    used = 0

    for msg in retrieved_messages:
        # Truncate long messages to their most informative opening
        content_snippet = msg["content"]
        if len(content_snippet) > 350:
            content_snippet = content_snippet[:350] + "..."

        line = f"[{msg['role'].upper()}]: {content_snippet}"
        line_tokens = count_tokens(line)

        if used + line_tokens > max_tokens:
            break

        lines.append(line)
        used += line_tokens

    return "\n\n".join(lines)


def calculate_window_stats(messages: list[dict]) -> dict:
    """
    Analyse the current context window composition.

    Returns stats used by the Streamlit token dashboard.
    """
    total_tokens = sum(m.get("token_count", 0) for m in messages)
    by_layer = {layer.value: 0 for layer in ContextLayer}
    by_role = {"user": 0, "assistant": 0, "system": 0}

    for msg in messages:
        layer = msg.get("layer", ContextLayer.WORKING.value)
        role = msg.get("role", "user")
        tokens = msg.get("token_count", 0)

        if layer in by_layer:
            by_layer[layer] += tokens
        if role in by_role:
            by_role[role] += tokens

    return {
        "total_tokens": total_tokens,
        "by_layer": by_layer,
        "by_role": by_role,
        "message_count": len(messages),
    }
