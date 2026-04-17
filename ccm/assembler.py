# ccm/assembler.py
#
# Builds the final compressed context packet.
# This is what actually gets injected into every LLM prompt.
#
# WHAT IT DOES:
#   Takes all memory tiers and assembles them into one
#   clean, structured context block that fits within
#   the token budget.
#
# OUTPUT STRUCTURE:
#   [CRITICAL CONSTRAINTS AND PREFERENCES]
#   Always present. From Working Memory (Tier 1).
#
#   [RELEVANT CONVERSATION HISTORY]
#   RAG-retrieved episodic summaries. From Tier 2.
#
#   [RELEVANT RESEARCH AND DETAILS]
#   RAG-retrieved tool results. From Tier 3.
#
#   [RECENT CONVERSATION]
#   Last N turns verbatim. Always present.
#
# TOKEN BUDGET:
#   Total target: 2000 tokens
#   Working memory:  ~400 tokens (always)
#   Recent turns:    ~600 tokens (always)
#   Retrieved:      ~1000 tokens (RAG fills this)

import tiktoken
from travel_agent.prompts import (
    SECTION_WORKING_MEMORY,
    SECTION_EPISODIC,
    SECTION_ARCHIVED,
    SECTION_RECENT,
    SECTION_DIVIDER
)

# Total token budget for the assembled context
# This is what we inject in addition to the system prompt
TOTAL_CONTEXT_BUDGET = 2000
RECENT_TURNS_BUDGET = 600    # Always reserve this for recent turns
WORKING_MEM_BUDGET = 400     # Always reserve for working memory
RETRIEVAL_BUDGET = TOTAL_CONTEXT_BUDGET - RECENT_TURNS_BUDGET - WORKING_MEM_BUDGET


def count_tokens(text: str) -> int:
    """Accurate token count using tiktoken."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


class ContextAssembler:
    """
    Assembles the compressed context packet for each LLM call.

    This is the final step in the CCM pipeline.
    It takes outputs from all memory tiers and retrieval,
    and builds one clean structured string to prepend
    to the user message before sending to the LLM.

    The assembled context is what makes the difference
    between the baseline and CCM agents.

    Baseline agent sends:    Full raw history (~11,000 tokens)
    CCM agent sends:         Assembled context (~1,400 tokens)
    """

    def __init__(self):
        self.last_token_count = 0
        self.last_assembly_breakdown = {}

    def assemble(
        self,
        working_memory,
        retrieved: dict,
        recent_turns: list,
        max_recent_turns: int = 3
    ) -> str:
        """
        Build the full context packet.

        Parameters:
          working_memory:   WorkingMemory instance (Tier 1)
          retrieved:        Output from Retriever.retrieve()
                           {episodic: [...], semantic: [...]}
          recent_turns:     List of recent conversation dicts
                           [{"role": "user/assistant", "content": "..."}]
          max_recent_turns: How many recent turns to include in full

        Returns:
          Assembled context string ready to prepend to user message
        """
        sections = []
        token_breakdown = {}

        # ── Section 1: Working Memory (always present) ──────────
        working_mem_text = working_memory.format_for_prompt()
        working_tokens = count_tokens(working_mem_text)

        sections.append(SECTION_DIVIDER)
        sections.append(SECTION_WORKING_MEMORY.strip())
        sections.append(working_mem_text)
        token_breakdown["working_memory"] = working_tokens

        # ── Section 2: Retrieved Episodic Memories ───────────────
        episodic_results = retrieved.get("episodic", [])
        if episodic_results:
            episodic_lines = []
            episodic_tokens = 0

            for result in episodic_results:
                line = f"  • {result['text']}"
                line_tokens = count_tokens(line)
                if episodic_tokens + line_tokens > RETRIEVAL_BUDGET // 2:
                    break
                episodic_lines.append(line)
                episodic_tokens += line_tokens

            if episodic_lines:
                sections.append(SECTION_DIVIDER)
                sections.append(SECTION_EPISODIC.strip())
                sections.extend(episodic_lines)
                token_breakdown["episodic"] = episodic_tokens

        # ── Section 3: Retrieved Semantic/Archived Memories ──────
        semantic_results = retrieved.get("semantic", [])
        if semantic_results:
            semantic_lines = []
            semantic_tokens = 0

            for result in semantic_results:
                tool = result.get("tool_name", "tool")
                line = f"  [{tool}] {result['text']}"
                line_tokens = count_tokens(line)
                if semantic_tokens + line_tokens > RETRIEVAL_BUDGET // 2:
                    break
                semantic_lines.append(line)
                semantic_tokens += line_tokens

            if semantic_lines:
                sections.append(SECTION_DIVIDER)
                sections.append(SECTION_ARCHIVED.strip())
                sections.extend(semantic_lines)
                token_breakdown["semantic"] = semantic_tokens

        # ── Section 4: Recent Conversation Turns ─────────────────
        # Take the last N turns (not counting current message)
        recent = recent_turns[-max_recent_turns * 2:] \
            if recent_turns else []

        if recent:
            recent_lines = []
            recent_tokens = 0

            for turn in recent:
                role = turn.get("role", "unknown").upper()
                content = turn.get("content", "")

                # Skip tool call messages — too verbose
                if role == "TOOL":
                    continue

                # Truncate very long messages
                if len(content) > 500:
                    content = content[:500] + "...[truncated]"

                line = f"  {role}: {content}"
                line_tokens = count_tokens(line)

                if recent_tokens + line_tokens > RECENT_TURNS_BUDGET:
                    break

                recent_lines.append(line)
                recent_tokens += line_tokens

            if recent_lines:
                sections.append(SECTION_DIVIDER)
                sections.append(SECTION_RECENT.strip())
                sections.extend(recent_lines)
                token_breakdown["recent_turns"] = recent_tokens

        sections.append(SECTION_DIVIDER)

        # Join everything into one string
        assembled = "\n".join(sections)

        # Track metrics
        self.last_token_count = count_tokens(assembled)
        self.last_assembly_breakdown = token_breakdown

        print(
            f"[Assembler] Context assembled: "
            f"{self.last_token_count} tokens"
        )
        print(f"  Breakdown: {token_breakdown}")

        return assembled

    def get_last_token_count(self) -> int:
        """Returns token count of last assembled context."""
        return self.last_token_count

    def get_breakdown(self) -> dict:
        """Returns section-by-section token breakdown."""
        return self.last_assembly_breakdown

    def format_for_display(
        self,
        working_memory,
        retrieved: dict,
        recent_turns: list
    ) -> dict:
        """
        Format context info for UI display.
        Used by Gradio to show memory state panel.

        Returns dict with sections separated for display.
        """
        return {
            "working_memory": working_memory.format_for_prompt(),
            "episodic": [r["text"] for r in retrieved.get("episodic", [])],
            "semantic": [
                f"[{r.get('tool_name', 'tool')}] {r['text']}"
                for r in retrieved.get("semantic", [])
            ],
            "recent_turns": recent_turns[-6:],
            "total_tokens": self.last_token_count,
            "breakdown": self.last_assembly_breakdown
        }