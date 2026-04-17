# ccm/semantic_memory.py
#
# Tier 3: Semantic / Archived Memory
#
# WHAT IT STORES:
#   Compressed tool results and detailed research.
#   Things like: compressed hotel search results,
#   compressed flight data, weather summaries.
#
# DIFFERENCE FROM EPISODIC:
#   Episodic = conversation summaries (what happened)
#   Semantic  = factual content from tools (what was found)
#
# RETRIEVAL BEHAVIOR:
#   Only retrieved when query directly needs factual details.
#   Example: "What hotels did we look at in Tokyo?"
#   → Retrieves the archived hotel search result
#   → Does NOT retrieve the episodic "we searched hotels" summary
#
# Both episodic and semantic use the same ChromaDB instance
# but different collections (like different tables in a database).

import os
import uuid
from datetime import datetime
from typing import Optional
import chromadb

# Reuse the same embedding model from episodic_memory
# No need to load it twice
from ccm.episodic_memory import embed, get_embedding_model

# ── Configuration ──────────────────────────────────────────────
CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "semantic_memory"
DEFAULT_TOP_K = 3           # Fewer results than episodic
                            # Archived items are larger/more specific
SIMILARITY_THRESHOLD = 0.35  # Slightly higher threshold
                              # We only want truly relevant archives


class SemanticMemory:
    """
    Tier 3: Semantic / Archived Memory

    Stores compressed tool results for retrieval.

    Unlike episodic memory which stores conversation summaries,
    this stores the actual CONTENT that was researched:
    hotel options, flight results, weather conditions, etc.

    This is queried separately from episodic memory because
    it answers different types of questions:

    Episodic answers: "What happened?" "What did we decide?"
    Semantic answers: "What options did we find?" "What were the prices?"

    AGENT-CENTRIC:
    This class stores tool results from any type of tool.
    It does not know about travel specifically.
    A code agent could store function documentation here.
    A medical agent could store lab result summaries here.
    """

    def __init__(self):
        os.makedirs(CHROMA_PATH, exist_ok=True)

        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Compressed tool results and research",
                "hnsw:space": "cosine"
            }
        )

        print(
            f"[SemanticMemory] Ready. "
            f"Contains {self.collection.count()} entries."
        )

    def add(
        self,
        compressed_result: str,
        tool_name: str,
        query_used: str,
        turn_number: int = 0,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Store a compressed tool result.

        Parameters:
          compressed_result: The already-compressed tool output
                            (compressed by compressor.py before storing)
          tool_name:         Which tool produced this result
          query_used:        The query that was passed to the tool
          turn_number:       Which conversation turn this came from
          metadata:          Optional extra metadata

        Returns:
          Unique ID for this entry

        Example:
          semantic.add(
            compressed_result="Tokyo hotels: Shinjuku Park $120★4.2,
                               Hilton $220★4.6, Capsule Inn $55★3.9",
            tool_name="places_search",
            query_used="hotels in Tokyo",
            turn_number=5
          )
        """
        if not compressed_result or not compressed_result.strip():
            return ""

        memory_id = f"sem_{uuid.uuid4().hex[:12]}"

        # The text we embed combines the query AND the result
        # This makes retrieval work better:
        # Future query "hotels Tokyo" finds this entry
        # because we embedded "hotels in Tokyo" + the results
        embeddable_text = f"Query: {query_used}\nResult: {compressed_result}"

        entry_metadata = {
            "tool_name": tool_name,
            "query_used": query_used,
            "turn_number": turn_number,
            "created_at": datetime.now().isoformat(),
            "stale": False,
            "type": "tool_result"
        }

        if metadata:
            entry_metadata.update(metadata)

        vector = embed(embeddable_text)

        self.collection.add(
            documents=[compressed_result],
            embeddings=[vector],
            metadatas=[entry_metadata],
            ids=[memory_id]
        )

        print(
            f"[SemanticMemory] Stored {tool_name} result: "
            f"{memory_id}"
        )
        print(f"  Content: {compressed_result[:80]}...")

        return memory_id

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tool_filter: Optional[str] = None,
        exclude_stale: bool = True
    ) -> list:
        """
        Retrieve relevant archived tool results.

        Parameters:
          query:        Current user message or topic
          top_k:        Maximum results to return
          tool_filter:  If set, only retrieve results from this tool
                       e.g., tool_filter="places_search"
          exclude_stale: Skip stale entries

        Returns:
          List of dicts with: id, text, similarity, tool_name, metadata
        """
        if self.collection.count() == 0:
            return []

        actual_top_k = min(top_k, self.collection.count())
        query_vector = embed(query)

        # Build where filter
        # ChromaDB supports AND conditions via $and
        where_conditions = []
        if exclude_stale:
            where_conditions.append({"stale": {"$eq": False}})
        if tool_filter:
            where_conditions.append({"tool_name": {"$eq": tool_filter}})

        where_filter = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=actual_top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[SemanticMemory] Query error: {e}")
            return []

        retrieved = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for doc, meta, dist, rid in zip(
            documents, metadatas, distances, ids
        ):
            similarity = 1.0 - (dist / 2.0)

            if similarity < SIMILARITY_THRESHOLD:
                continue

            retrieved.append({
                "id": rid,
                "text": doc,
                "similarity": round(similarity, 3),
                "tool_name": meta.get("tool_name", "unknown"),
                "query_used": meta.get("query_used", ""),
                "metadata": meta
            })

        if retrieved:
            print(
                f"[SemanticMemory] Retrieved {len(retrieved)} "
                f"archived results"
            )
        else:
            print("[SemanticMemory] No relevant archives found")

        return retrieved

    def mark_stale_by_content(self, substring: str) -> int:
        """
        Mark archived entries mentioning substring as stale.
        Mirror of the same method in EpisodicMemory.
        Called by StaleDetector when context is cancelled.
        """
        if self.collection.count() == 0:
            return 0

        try:
            all_entries = self.collection.get(
                include=["documents", "metadatas"]
            )
        except Exception as e:
            print(f"[SemanticMemory] Error getting entries: {e}")
            return 0

        documents = all_entries.get("documents", [])
        metadatas = all_entries.get("metadatas", [])
        ids = all_entries.get("ids", [])

        stale_count = 0
        substring_lower = substring.lower()

        for doc, meta, entry_id in zip(documents, metadatas, ids):
            if substring_lower in doc.lower():
                if meta.get("stale", False):
                    continue

                updated_meta = dict(meta)
                updated_meta["stale"] = True
                updated_meta["staled_at"] = datetime.now().isoformat()
                updated_meta["stale_reason"] = f"Cancelled: {substring}"

                self.collection.update(
                    ids=[entry_id],
                    metadatas=[updated_meta]
                )
                stale_count += 1

        if stale_count > 0:
            print(
                f"[SemanticMemory] Marked {stale_count} "
                f"archived entries stale for '{substring}'"
            )

        return stale_count

    def get_count(self) -> dict:
        """Return count stats."""
        total = self.collection.count()
        if total == 0:
            return {"total": 0, "active": 0, "stale": 0}
        try:
            stale = self.collection.get(
                where={"stale": {"$eq": True}}
            )
            stale_count = len(stale.get("ids", []))
            return {
                "total": total,
                "active": total - stale_count,
                "stale": stale_count
            }
        except Exception:
            return {"total": total, "active": total, "stale": 0}

    def reset(self):
        """Delete all archived memories."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("[SemanticMemory] Reset complete")
        except Exception as e:
            print(f"[SemanticMemory] Reset error: {e}")