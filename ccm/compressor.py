# ccm/compressor.py
#
# Compresses tool results before storing in semantic memory.
# This is the single biggest source of token reduction.
#
# WITHOUT compression:
#   places_search returns ~600 tokens of hotel data
#   web_search returns ~800 tokens of flight data
#   15 tool calls = ~10,500 tokens just from tools
#
# WITH compression:
#   Each tool result → 60-100 token summary
#   15 tool calls = ~1,200 tokens
#   Compression ratio: ~8x on tool results alone
#
# AGENT-CENTRIC:
#   Compressor does not know about travel.
#   It receives: tool_type, tool_result, user_constraints
#   It returns: compressed summary
#   Works for any tool in any agent domain.

import os
import json
from groq import Groq
from dotenv import load_dotenv
from travel_agent.prompts import COMPRESSION_PROMPT

load_dotenv()


class ToolCompressor:
    """
    Compresses raw tool results into compact summaries.

    Called immediately after every tool execution,
    before the result is stored in semantic memory
    or injected into the agent context.

    The compressed version serves two purposes:
    1. Stored in SemanticMemory for future RAG retrieval
    2. Returned to the agent as a cleaner version of the result

    Token savings example:
      web_search("flights to Tokyo") → 820 tokens raw
      After compression               →  75 tokens
      Saving: 745 tokens per tool call
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Use fast 8B model — compression is straightforward
        self.model = "llama-3.1-8b-instant"

        # Track compression stats for metrics reporting
        self.stats = {
            "total_calls": 0,
            "total_tokens_before": 0,
            "total_tokens_after": 0
        }

    def compress(
        self,
        tool_result: dict,
        tool_name: str,
        user_constraints: list = None
    ) -> str:
        """
        Compress a tool result dictionary into a brief summary.

        Parameters:
          tool_result:       Raw dict returned by the tool function
          tool_name:         Name of the tool (for context-aware compression)
          user_constraints:  List of constraint strings from working memory
                            Used to flag conflicts in compressed output

        Returns:
          Compressed summary string (60-100 words typically)

        Example:
          raw = places_search("Tokyo", "hotels")  # 600 token dict
          compressed = compressor.compress(raw, "places_search",
                         ["allergic to shellfish", "budget $3000"])
          # Returns: "Tokyo hotels: Shinjuku Park $120★4.2 ✅,
          #           Hilton $220★4.6 ❌over budget,
          #           Capsule Inn $55★3.8"
          # ~35 tokens
        """
        if not tool_result:
            return "Tool returned no results."

        # Convert dict to string for the prompt
        # JSON format preserves structure better than str()
        tool_result_str = json.dumps(tool_result, indent=2)

        # Estimate token count before compression
        # Rough estimate: 4 chars ≈ 1 token
        tokens_before = len(tool_result_str) // 4
        self.stats["total_tokens_before"] += tokens_before

        # Format constraints as readable list
        constraints_text = "None specified"
        if user_constraints:
            constraints_text = "\n".join(
                f"  - {c}" for c in user_constraints
            )

        # Build compression prompt
        prompt = COMPRESSION_PROMPT.format(
            tool_type=tool_name,
            user_constraints=constraints_text,
            tool_result=tool_result_str
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You compress tool results into brief plain text summaries. "
                            "No markdown. No bold text. No asterisks. No headers. "
                            "Use plain sentences or simple dashes for lists. "
                            "Preserve all numbers exactly. Be concise."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.0
            )

            compressed = response.choices[0].message.content.strip()

            # Track stats
            tokens_after = len(compressed) // 4
            self.stats["total_tokens_after"] += tokens_after
            self.stats["total_calls"] += 1

            ratio = tokens_before / max(tokens_after, 1)
            print(
                f"[Compressor] {tool_name}: "
                f"{tokens_before} → {tokens_after} tokens "
                f"({ratio:.1f}x compression)"
            )

            return compressed

        except Exception as e:
            print(f"[Compressor] Error: {e}")
            # Fallback: return truncated raw result
            return self._fallback_compress(tool_result, tool_name)

    def _fallback_compress(
        self,
        tool_result: dict,
        tool_name: str
    ) -> str:
        """
        Fallback compression without LLM.
        Used if the API call fails.
        Extracts key fields based on tool type.
        Simple but reliable.
        """
        try:
            if tool_name == "places_search":
                return self._compress_places_fallback(tool_result)
            elif tool_name == "web_search":
                return self._compress_web_fallback(tool_result)
            elif tool_name == "weather_fetch":
                return self._compress_weather_fallback(tool_result)
            elif tool_name == "budget_tracker":
                return self._compress_budget_fallback(tool_result)
            else:
                # Generic: just get first 200 chars
                raw = json.dumps(tool_result)
                return raw[:200] + "..." if len(raw) > 200 else raw
        except Exception:
            return f"{tool_name} returned results (compression failed)"

    def _compress_places_fallback(self, result: dict) -> str:
        """Extract key info from places_search result."""
        lines = []
        search_type = result.get("search_type", "places")
        location = result.get("location", "")
        lines.append(f"{search_type.title()} in {location}:")

        # Hotels
        all_results = result.get("all_results", [])
        if not all_results:
            all_results = result.get("results", [])

        for item in all_results[:3]:
            name = item.get("name", "Unknown")
            price = item.get("price_per_night", item.get("price_range", "?"))
            rating = item.get("rating", "")
            price_str = f"${price}/night" if isinstance(price, (int, float)) else price
            lines.append(f"  {name}: {price_str} ★{rating}")

        return "\n".join(lines)

    def _compress_web_fallback(self, result: dict) -> str:
        """Extract key info from web_search result."""
        search_type = result.get("search_type", "search")
        if search_type == "flights":
            route = result.get("route", "")
            cheapest = result.get("cheapest_price", "?")
            count = result.get("results_count", 0)
            return (
                f"Flights {route}: {count} options found. "
                f"Cheapest: ${cheapest}"
            )
        return f"Search returned {result.get('results_count', 'some')} results"

    def _compress_weather_fallback(self, result: dict) -> str:
        """Extract key info from weather_fetch result."""
        city = result.get("city", "")
        conditions = result.get("current_conditions", {})
        temp = conditions.get("temperature_f", "?")
        desc = conditions.get("description", "")
        return f"Weather in {city}: {temp}°F, {desc}"

    def _compress_budget_fallback(self, result: dict) -> str:
        """Extract key info from budget_tracker result."""
        remaining = result.get("remaining", "?")
        total = result.get("total_budget", "?")
        spent = result.get("total_spent", "?")
        return (
            f"Budget: ${spent} spent of ${total} total. "
            f"${remaining} remaining."
        )

    def get_compression_stats(self) -> dict:
        """
        Return compression statistics for metrics reporting.
        Used in evaluation to demonstrate token reduction.
        """
        total_before = self.stats["total_tokens_before"]
        total_after = self.stats["total_tokens_after"]
        calls = self.stats["total_calls"]

        ratio = total_before / max(total_after, 1)
        saved = total_before - total_after

        return {
            "total_tool_calls_compressed": calls,
            "total_tokens_before": total_before,
            "total_tokens_after": total_after,
            "tokens_saved": saved,
            "overall_compression_ratio": round(ratio, 2),
            "average_tokens_before": round(total_before / max(calls, 1)),
            "average_tokens_after": round(total_after / max(calls, 1))
        }

    def reset_stats(self):
        """Reset compression statistics for new conversation."""
        self.stats = {
            "total_calls": 0,
            "total_tokens_before": 0,
            "total_tokens_after": 0
        }