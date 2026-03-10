"""
main_agent.py — The public interface to the context engineering system.

WHY THIS EXISTS
---------------
The graph is the engine. This module is the steering wheel.
It provides three clean functions that the Streamlit app (and tests)
use to interact with the system:

  create_session() — Initialise a new agent with empty state
  chat()           — Send a message, get a response + updated state
  get_context_health() — Snapshot of all context engineering metrics

AGENT-AS-TOOL (Technique 4)
---------------------------
In this build, the main agent is a single LangGraph graph.
The Agent-as-Tool pattern would extend this by having reason_node
call sub-agents as tool calls — each sub-agent managing its own
focused context window.

The extension would look like:
  main_agent.chat()
    → reason_node calls retrieval_agent as a tool
    → reason_node calls summariser_agent as a tool
    → Each sub-agent has a fresh, focused context window
    → Results are returned to main_agent as tool outputs

This architecture prevents any single agent from being overloaded
with context while still giving the system access to specialised logic.
"""

import uuid
import time
import os

from langsmith import traceable
from src.graph.graph import agent_graph
from src.graph.state import AgentState
from src.context.token_counter import count_tokens, get_token_percentage
from src.context.layer_manager import ContextLayer, layer_summary
from src.context.offload_store import (
    get_session_stats, clear_session,
    register_session, update_session_activity,
    save_critical_memory, load_critical_memory,
    load_prior_session_messages, get_user_session_count,
    get_user_last_active, offload_message, flush_session_messages,
)
from src.config import TOKEN_BUDGET, PRE_ROT_THRESHOLD


def create_session(
    system_prompt: str = None,
    token_budget: int = TOKEN_BUDGET,
    pre_rot_threshold: float = PRE_ROT_THRESHOLD,
    user_id: str = None,
) -> AgentState:
    """
    Create a new agent session, restoring prior context for returning users.

    PERSISTENCE BEHAVIOUR
    ---------------------
    If user_id is provided:
      1. Load CRITICAL messages from all prior sessions → restore to active context
         These are identity facts the user established in previous sessions.
         A returning user never has to re-introduce themselves.
      2. Load prior offloaded messages → re-offload under new session_id
         These become searchable via retrieve_from_memory during this session.
         The agent has access to full conversation history on demand.
      3. Register the new session_id against the user_id for future sessions.

    If user_id is None (anonymous):
      Behaves exactly as before — clean session, no persistence.

    Args:
        system_prompt:      Optional instructions for the agent.
        token_budget:       Max tokens for this session.
        pre_rot_threshold:  Fraction of budget that triggers offload.
        user_id:            Optional user identifier for cross-session memory.
                            Use the user's name or any stable identifier.

    Returns:
        A fully initialised AgentState dict, with prior context restored
        if user_id matched a returning user.
    """
    session_id = str(uuid.uuid4())
    messages   = []

    if system_prompt:
        messages.append({
            "role":        "system",
            "content":     system_prompt,
            "layer":       ContextLayer.CRITICAL.value,
            "token_count": count_tokens(system_prompt),
            "message_id":  str(uuid.uuid4()),
            "timestamp":   time.time(),
        })

    # Restore prior context for returning users
    is_returning_user = False
    if user_id:
        prior_sessions = get_user_session_count(user_id)
        is_returning_user = prior_sessions > 0

        if is_returning_user:
            # Step 1: Restore CRITICAL messages into active context
            # These are identity facts from prior sessions — inject them
            # with slightly older timestamps so they don't displace new messages
            critical_memories = load_critical_memory(user_id)
            for mem in critical_memories:
                messages.append({
                    "role":        mem["role"],
                    "content":     mem["content"],
                    "layer":       ContextLayer.CRITICAL.value,
                    "token_count": mem["token_count"],
                    "message_id":  str(uuid.uuid4()),
                    "timestamp":   mem["created_at"],
                })

            # Step 2: Re-offload prior session messages under new session_id
            # This makes them searchable via retrieve_from_memory without
            # putting them all in the active context window
            prior_messages = load_prior_session_messages(user_id, max_sessions=3)
            for msg in prior_messages:
                offload_message(
                    message_id=  str(uuid.uuid4()),
                    session_id=  session_id,
                    role=        msg["role"],
                    content=     msg["content"],
                    layer=       msg.get("layer", "working"),
                    token_count= msg["token_count"],
                    timestamp=   msg["timestamp"],
                )

        # Register this new session against the user
        try:
            register_session(user_id, session_id)
        except Exception as e:
            import warnings
            warnings.warn(f"[Persistence] register_session failed for {user_id}: {e}")

    initial_tokens = sum(m["token_count"] for m in messages)

    state = AgentState(
        messages           = messages,
        session_id         = session_id,
        token_budget       = token_budget,
        current_tokens     = initial_tokens,
        pre_rot_threshold  = pre_rot_threshold,
        needs_offload      = False,
        offloaded_count    = len([m for m in messages if False]),  # 0 — prior msgs go to offload store directly
        offloaded_tokens   = 0,
        latest_query       = "",
        retrieved_context  = "",
        scratchpad         = "",
        agent_mode         = "idle",
        final_response     = "",
    )

    # Store user_id and returning status in state for app.py to read
    state["user_id"]           = user_id
    state["is_returning_user"] = is_returning_user

    return state


@traceable(
    name="context-engineer-chat",
    tags=["production"],
    metadata={"project": "context-engineer", "version": "1.0"},
)
def chat(state: AgentState, user_message: str) -> tuple[AgentState, str]:
    """
    Send a user message through the agent and return the response.

    The state is passed in and a new updated state is returned.
    This keeps the session alive across multiple turns — the caller
    (Streamlit app or test) holds the state between calls.

    The @traceable decorator instruments this function with LangSmith.
    Every call logs:
      - Input: user_message + current token counts
      - Output: response + updated health metrics
      - Metadata: session_id, offload counts, technique activations
    This gives you a complete audit trail of every conversation turn
    including which context engineering techniques fired.

    Args:
        state:        The current AgentState (from create_session or previous chat).
        user_message: The user's input text.

    Returns:
        Tuple of (updated_state, response_text).

    Example:
        state, response = chat(state, "What is the UK basic tax rate?")
        print(response)
        state, response = chat(state, "And what about National Insurance?")
        print(response)
    """
    # Add the user message to state (unclassified — classify_input_node tags it)
    user_msg = {
        "role":        "user",
        "content":     user_message,
        "layer":       None,   # classify_input_node will set this
        "token_count": 0,      # classify_input_node will set this
        "message_id":  str(uuid.uuid4()),
        "timestamp":   time.time(),
    }

    updated_state = {
        **state,
        "messages":       state["messages"] + [user_msg],
        "latest_query":   user_message,
        "final_response": "",  # Clear previous response
    }

    # Run the full graph: classify → monitor → [offload?] → retrieve → reason → respond
    result   = agent_graph.invoke(updated_state)
    response = result.get("final_response", "No response generated.")

    # Attach context health metadata to the trace for LangSmith visibility.
    # This makes every run searchable by token usage, offload events,
    # and technique activations — essential for production monitoring.
    health = get_context_health(result)
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        try:
            from langsmith import get_current_run_tree
            run = get_current_run_tree()
            if run:
                run.add_metadata({
                    "session_id":        health["session_id"],
                    "tokens_used":       health["current_tokens"],
                    "token_budget":      health["token_budget"],
                    "usage_pct":         round(health["usage_pct"] * 100, 1),
                    "offloaded_count":   health["offloaded_count"],
                    "offloaded_tokens":  health["offloaded_tokens"],
                    "needs_offload":     health["needs_offload"],
                    "active_messages":   health["active_messages"],
                    "agent_mode":        health["agent_mode"],
                    "retrieved_context": health["retrieved_context_len"] > 0,
                })
        except Exception:
            pass  # Never let tracing break the app

    # Persist CRITICAL messages for returning sessions
    # After each turn, if the newly classified user message is CRITICAL,
    # save it to critical_memory so future sessions can restore it.
    user_id = state.get("user_id")
    if user_id:
        messages_after = result.get("messages", [])
        for msg in reversed(messages_after):
            if (msg.get("role") == "user"
                    and msg.get("layer") == ContextLayer.CRITICAL.value
                    and msg.get("content") == user_message):
                try:
                    save_critical_memory(
                        user_id=    user_id,
                        session_id= result["session_id"],
                        message_id= msg.get("message_id", str(uuid.uuid4())),
                        role=       "user",
                        content=    user_message,
                        token_count=msg.get("token_count", count_tokens(user_message)),
                    )
                except Exception as e:
                    import warnings
                    warnings.warn(f"[Persistence] save_critical_memory failed: {e}")
                break

        # Update session activity counter in DB
        try:
            update_session_activity(
                session_id=    result["session_id"],
                message_count= len(messages_after),
            )
        except Exception as e:
            import warnings
            warnings.warn(f"[Persistence] update_session_activity failed: {e}")

    # Carry user_id forward so state never loses it between turns
    result["user_id"] = user_id

    return result, response


def get_context_health(state: AgentState) -> dict:
    """
    Return a complete snapshot of context health metrics.

    This is what the Streamlit token dashboard visualises.
    Every number shown in the UI comes from this function.

    Returns a dict with:
      - Token usage and percentage
      - Threshold position
      - Layer breakdown (count and tokens per layer)
      - Offloading statistics
      - Agent mode
      - Scratchpad entry count
    """
    budget    = state.get("token_budget", TOKEN_BUDGET)
    current   = state.get("current_tokens", 0)
    threshold = state.get("pre_rot_threshold", PRE_ROT_THRESHOLD)
    messages  = state.get("messages", [])

    # Layer breakdown from active messages
    layers = layer_summary(messages)

    # Offload store stats for this session
    store_stats = get_session_stats(state["session_id"])

    return {
        # Token window
        "token_budget":     budget,
        "current_tokens":   current,
        "usage_pct":        get_token_percentage(current, budget),
        "threshold_pct":    threshold,
        "threshold_tokens": int(budget * threshold),
        "headroom_tokens":  max(0, int(budget * threshold) - current),
        "pct_label":        f"{get_token_percentage(current, budget)*100:.1f}%",

        # Active context layers
        "layer_breakdown":  layers,   # {"critical": {"count":x,"tokens":y}, ...}
        "active_messages":  len(messages),

        # Offloading
        "offloaded_count":  state.get("offloaded_count", 0),
        "offloaded_tokens": state.get("offloaded_tokens", 0),
        "store_total":      store_stats.get("message_count", 0),

        # Retrieval
        "retrieved_context_len": len(state.get("retrieved_context", "")),

        # Agent state
        "agent_mode":          state.get("agent_mode", "idle"),
        "session_id":          state["session_id"],
        "scratchpad_lines":    len([
            l for l in state.get("scratchpad", "").split("\n") if l.strip()
        ]),
        "needs_offload":       state.get("needs_offload", False),
    }


def reset_session(state: AgentState) -> AgentState:
    """
    Clear the offload store for a session and return a fresh state.
    Used by the Streamlit 'New Session' button.
    """
    clear_session(state["session_id"])
    return create_session(
        token_budget=state.get("token_budget", TOKEN_BUDGET),
        pre_rot_threshold=state.get("pre_rot_threshold", PRE_ROT_THRESHOLD),
    )