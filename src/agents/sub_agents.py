"""
sub_agents.py — Sub-agents that Claude calls as tools (Technique 4).

WHAT AGENT-AS-TOOL MEANS
-------------------------
In Techniques 1-3 and 5-7, WE decide when to retrieve, when to offload,
when to compress. The logic is hardcoded in the graph.

Agent-as-Tool flips this. We give Claude access to sub-agents as tools.
Claude autonomously decides WHEN to call them and WHAT to ask for.

This matters because Claude has information we don't:
  - It knows when its own reasoning is uncertain
  - It knows when it's missing context to answer well
  - It knows when the conversation is too dense to reason over clearly

By letting Claude call retrieval when IT feels it needs it (rather than
when our token counter says to), we get smarter, more targeted retrieval.

TWO SUB-AGENTS
--------------
1. RetrievalAgent  — Searches offloaded long-term memory by query.
   Claude calls this when it recognises it may be missing context.
   It formulates its own search query — often better than keyword matching.

2. SummariserAgent — Compresses a block of dense context.
   Claude calls this when the working context is too large to reason
   over in one pass. Returns a compressed summary it can reason with.

HOW TOOLS WORK IN THE ANTHROPIC API
-------------------------------------
You define tools as JSON schemas. Claude can return a tool_use block
instead of (or alongside) a text response. Your code executes the tool
and returns the result. Claude then continues reasoning with the result.

The loop:
  1. Send messages + tool definitions to Claude
  2. Claude returns tool_use block: {name, input}
  3. Execute the tool with the given input
  4. Send tool_result back to Claude
  5. Claude continues — may call more tools or give final response
  6. Loop ends when Claude returns a text response with no tool calls
"""

import anthropic
from src.context.offload_store import retrieve_relevant
from src.context.context_manager import compress_retrieved
from src.config import ANTHROPIC_API_KEY, MODEL_NAME


_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# Tool Definitions (JSON Schema for Claude)

TOOL_DEFINITIONS = [
    {
        "name": "retrieve_from_memory",
        "description": (
            "Search long-term memory for context relevant to a specific query. "
            "Call this when you think important context may have been offloaded "
            "from the active conversation window. Formulate a specific search "
            "query — more specific queries return better results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query to find relevant past context. "
                        "Be specific — e.g. 'user name and location' rather than 'user info'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of past messages to retrieve. Default 3.",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "summarise_context",
        "description": (
            "Compress a long block of text into a concise summary. "
            "Call this when the working context contains long technical explanations "
            "that you need to reason about but don't need verbatim. "
            "Returns a compressed version that preserves key facts and decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarise.",
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "What aspect to focus the summary on. "
                        "E.g. 'key technical decisions', 'user requirements', 'facts mentioned'."
                    ),
                },
            },
            "required": ["text"],
        },
    },
]


# Sub-Agent Implementations

class RetrievalAgent:
    """
    Searches the offload store for past context.

    This is the same retrieval logic from retrieve_context_node, but
    now Claude controls WHEN it runs and WHAT it searches for.
    Claude's self-formulated query is often more precise than the
    keyword overlap from the user's raw message.
    """

    def run(self, query: str, session_id: str, max_results: int = 3) -> str:
        """
        Execute a retrieval search and return compressed results as text.

        Args:
            query:       Claude's search query (formulated by the LLM itself)
            session_id:  Session to search within
            max_results: Max messages to return

        Returns:
            Formatted string of retrieved context, or a "nothing found" message.
        """
        relevant = retrieve_relevant(
            session_id=session_id,
            query=query,
            max_results=max_results,
            max_tokens=2000,
        )

        if not relevant:
            return f"No relevant past context found for query: '{query}'"

        compressed = compress_retrieved(relevant, max_tokens=1500)
        return (
            f"Retrieved {len(relevant)} message(s) from long-term memory "
            f"for query '{query}':\n\n{compressed}"
        )


class SummariserAgent:
    """
    Compresses dense context into a focused summary.

    Uses Claude itself to do the summarisation — a sub-agent calling
    the same LLM with a focused summarisation prompt. The key insight
    is that a summarisation call with 500 tokens of focused input is
    much cheaper than reasoning over 3000 tokens of dense context.
    """

    def run(self, text: str, focus: str = "key facts and decisions") -> str:
        """
        Summarise a block of text with a specific focus.

        Args:
            text:  The text to compress.
            focus: What aspect to emphasise in the summary.

        Returns:
            A concise summary string.
        """
        if len(text) < 200:
            # No point summarising short text
            return text

        try:
            response = _client.messages.create(
                model=MODEL_NAME,
                max_tokens=400,
                system=(
                    "You are a precise summarisation assistant. "
                    "Produce concise summaries that preserve all key facts, "
                    "decisions, and specific details. Never add information "
                    "not present in the original text."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Summarise the following text, focusing on: {focus}\n\n"
                        f"Text to summarise:\n{text}\n\n"
                        f"Produce a concise summary (3-5 sentences maximum)."
                    ),
                }],
            )
            return response.content[0].text if response.content else text

        except Exception:
            # If summarisation fails, return original (graceful degradation)
            return text[:500] + "..." if len(text) > 500 else text


# Tool Executor

def execute_tool(
    tool_name: str,
    tool_input: dict,
    session_id: str,
) -> str:
    """
    Route a tool call to the correct sub-agent and return the result.

    Called by reason_node when Claude returns a tool_use block.
    The result is sent back to Claude as a tool_result message.

    Args:
        tool_name:  Name of the tool Claude called.
        tool_input: Arguments Claude provided.
        session_id: Current session (needed for retrieval).

    Returns:
        String result to send back to Claude as tool_result content.
    """
    if tool_name == "retrieve_from_memory":
        agent = RetrievalAgent()
        return agent.run(
            query=tool_input.get("query", ""),
            session_id=session_id,
            max_results=tool_input.get("max_results", 3),
        )

    elif tool_name == "summarise_context":
        agent = SummariserAgent()
        return agent.run(
            text=tool_input.get("text", ""),
            focus=tool_input.get("focus", "key facts and decisions"),
        )

    else:
        return f"Unknown tool: {tool_name}"
