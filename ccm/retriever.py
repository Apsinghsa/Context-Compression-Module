# ccm/retriever.py
#
# Unified retrieval across Tier 2 (episodic) and Tier 3 (semantic).
# This is the RAG engine of the CCM.
#
# WHAT IT DOES:
#   1. Takes the current user query
#   2. Searches episodic memory (conversation summaries)
#   3. Searches semantic memory (tool results)
#   4. Optionally re-ranks by relevance using LLM
#   5. Returns a unified list of relevant memories
#
# WHY THIS IS NOT SIMPLE RAG:
#   Simple RAG: search documents → paste top N
#   Our system:
#     - Searches TWO separate memory tiers
#     - Each tier has different retention/staleness rules
#     - Optional LLM re-ranking pass (RETRIEVAL_RELEVANCE_PROMPT)
#     - Token-budget aware — knows how many tokens it can return
#     - Stale-aware — never returns cancelled context
#
# AGENT-CENTRIC:
#   This class only knows about episodic and semantic memory.
#   It does not know what the memories contain (travel or otherwise).

import os
import json
from typing import Optional
from groq import Groq
from dotenv import load_dotenv
from ccm.episodic_memory import EpisodicMemory
from ccm.semantic_memory import SemanticMemory
from travel_agent.prompts import RETRIEVAL_RELEVANCE_PROMPT

load_dotenv()

# Token budget for retrieved memories
# Working memory uses ~400 tokens (always present)
# Recent turns use ~600 tokens (always present)
# That leaves ~1000 tokens for retrieved memories
# in a 2000-token total context budget
MAX_RETRIEVAL_TOKENS = 1000
TOKENS_PER_CHAR = 0.25  # Rough estimate: 1 token ≈ 4 chars


def estimate_tokens(text: str) -> int:
    """Rough token count from character count."""
    return int(len(text) * TOKENS_PER_CHAR)


class Retriever:
    """
    Unified RAG retrieval engine.

    Searches both episodic and semantic memory tiers
    and returns the most relevant results within a token budget.

    The two-stage retrieval process:
      Stage 1 — Vector similarity search (fast, ChromaDB)
        Finds semantically similar memories using embeddings.
        May return loosely related results.

      Stage 2 — LLM re-ranking (optional, slower but precise)
        Scores each result for actual relevance to the query.
        Filters out results that are similar but not useful.
        Uses RETRIEVAL_RELEVANCE_PROMPT.
        Skip this stage for speed if needed.

    In production you would always use Stage 2.
    For demo purposes, Stage 1 alone works well enough.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        use_reranking: bool = True
    ):
        """
        Parameters:
          episodic_memory: Tier 2 memory instance
          semantic_memory: Tier 3 memory instance
          use_reranking:   If True, use LLM to re-rank results
                          Set to False for faster but less precise retrieval
        """
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.use_reranking = use_reranking
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def retrieve(
        self,
        query: str,
        n_episodic: int = 4,
        n_semantic: int = 3,
        token_budget: int = MAX_RETRIEVAL_TOKENS
    ) -> dict:
        """
        Main retrieval method. Searches both memory tiers.

        Parameters:
          query:        Current user message
          n_episodic:   Max episodic results to fetch initially
          n_semantic:   Max semantic results to fetch initially
          token_budget: Max tokens the returned results can use

        Returns:
          {
            "episodic": [...],   List of relevant episodic memories
            "semantic": [...],   List of relevant semantic memories
            "total_tokens": int, Estimated tokens used
            "query": str         The query used for retrieval
          }
        """
        print(f"\n[Retriever] Searching for: '{query[:60]}'")

        # Stage 1: Vector similarity search
        raw_episodic = self.episodic.retrieve(
            query=query,
            top_k=n_episodic
        )
        raw_semantic = self.semantic.retrieve(
            query=query,
            top_k=n_semantic
        )

        print(
            f"[Retriever] Stage 1 found: "
            f"{len(raw_episodic)} episodic, "
            f"{len(raw_semantic)} semantic"
        )

        # Stage 2: LLM re-ranking (if enabled and results exist)
        if self.use_reranking and (raw_episodic or raw_semantic):
            all_results = raw_episodic + raw_semantic
            reranked = self._rerank(query, all_results)

            # Split back into episodic and semantic
            episodic_ids = {r["id"] for r in raw_episodic}
            final_episodic = [
                r for r in reranked
                if r["id"] in episodic_ids
            ]
            final_semantic = [
                r for r in reranked
                if r["id"] not in episodic_ids
            ]
        else:
            final_episodic = raw_episodic
            final_semantic = raw_semantic

        # Apply token budget
        final_episodic, final_semantic, total_tokens = (
            self._apply_token_budget(
                final_episodic,
                final_semantic,
                token_budget
            )
        )

        print(
            f"[Retriever] Final: "
            f"{len(final_episodic)} episodic, "
            f"{len(final_semantic)} semantic, "
            f"~{total_tokens} tokens"
        )

        return {
            "episodic": final_episodic,
            "semantic": final_semantic,
            "total_tokens": total_tokens,
            "query": query
        }

    def _rerank(self, query: str, results: list) -> list:
        """
        LLM-based re-ranking of retrieved results.

        Uses RETRIEVAL_RELEVANCE_PROMPT to score each result
        0-3 based on actual relevance to the current query.
        Filters out results scoring 0 or 1.

        This is what separates our RAG from simple RAG.
        Simple RAG returns top-K by vector similarity.
        We additionally score by actual usefulness.
        """
        if not results:
            return []

        # Format results for the prompt
        items_text = ""
        for i, r in enumerate(results):
            items_text += (
                f"ID: {r['id']}\n"
                f"Text: {r['text']}\n"
                f"Vector similarity: {r.get('similarity', '?')}\n\n"
            )

        prompt = RETRIEVAL_RELEVANCE_PROMPT.format(
            query=query,
            retrieved_items=items_text
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You score retrieved memory items for relevance. "
                            "Return only valid JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.0
            )

            raw = response.choices[0].message.content.strip()

            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            scores_data = json.loads(raw)
            scores = {
                s["id"]: s["score"]
                for s in scores_data.get("scores", [])
            }

            # Filter results by score threshold
            # Score 2 or 3 = keep, Score 0 or 1 = drop
            reranked = []
            for result in results:
                score = scores.get(result["id"], 1)
                if score >= 2:
                    result["relevance_score"] = score
                    reranked.append(result)
                else:
                    print(
                        f"[Retriever] Dropped (score={score}): "
                        f"{result['text'][:50]}..."
                    )

            # Sort by relevance score descending
            reranked.sort(
                key=lambda x: x.get("relevance_score", 0),
                reverse=True
            )

            print(
                f"[Retriever] Re-ranking: "
                f"{len(results)} → {len(reranked)} results"
            )
            return reranked

        except Exception as e:
            print(f"[Retriever] Re-ranking failed: {e}. Using raw results.")
            return results

    def _apply_token_budget(
        self,
        episodic: list,
        semantic: list,
        budget: int
    ) -> tuple:
        """
        Trim results to fit within token budget.

        Priority order:
          1. Episodic memories (conversation context)
          2. Semantic memories (tool results)

        Returns:
          (trimmed_episodic, trimmed_semantic, total_tokens_used)
        """
        total_tokens = 0
        final_episodic = []
        final_semantic = []

        # Add episodic first (higher priority)
        for result in episodic:
            tokens = estimate_tokens(result["text"])
            if total_tokens + tokens <= budget:
                final_episodic.append(result)
                total_tokens += tokens
            else:
                print(
                    f"[Retriever] Budget limit: dropped episodic "
                    f"({tokens} tokens would exceed budget)"
                )
                break

        # Add semantic with remaining budget
        for result in semantic:
            tokens = estimate_tokens(result["text"])
            if total_tokens + tokens <= budget:
                final_semantic.append(result)
                total_tokens += tokens
            else:
                print(
                    f"[Retriever] Budget limit: dropped semantic "
                    f"({tokens} tokens would exceed budget)"
                )
                break

        return final_episodic, final_semantic, total_tokens