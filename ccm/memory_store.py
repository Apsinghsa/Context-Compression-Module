# ccm/memory_store.py
#
# Tier 1: Working Memory
#
# AGENT-CENTRIC DESIGN:
#   Stores ANY type of fact, completely domain agnostic.
#   Does not know about travel, allergies, budgets, or any
#   specific topic. Only knows about priority levels.
#
# WHAT IT DOES:
#   - Stores facts classified as critical/important/contextual
#   - Persists to disk as JSON so memory survives restarts
#   - Always injects critical facts into every prompt
#   - Provides clean text formatting for LLM consumption
#   - Enforces a token budget so it cannot overflow context

import os

os.makedirs("data", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)

import json
import tiktoken
from datetime import datetime
from typing import Any, Optional

MEMORY_FILE_PATH = "data/working_memory.json"
MAX_WORKING_MEMORY_TOKENS = 400  # Hard limit for prompt injection

DEFAULT_MEMORY = {
    "facts": {
        "critical": [],
        "important": [],
        "contextual": []
    },
    "decisions": [],
    "cancelled": [],
    "turn_count": 0,
    "conversation_id": "",
    "last_updated": ""
}


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split()) * 4 // 3


class WorkingMemory:
    """
    Tier 1: Working Memory
    
    General-purpose, domain-agnostic fact store.
    
    Stores facts as structured items with priority levels.
    Critical facts are ALWAYS in every prompt.
    Important facts are usually in every prompt.
    Contextual facts are stored but retrieved via RAG when needed.
    
    This class knows nothing about travel, allergies, hotels,
    or any specific topic. It only knows about priority levels.
    Swap the travel agent for a medical agent and this class
    works identically without any changes.
    """

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self.memory = self._load()

    def _load(self) -> dict:
        """Load from disk or initialize fresh."""
        if os.path.exists(MEMORY_FILE_PATH):
            try:
                with open(MEMORY_FILE_PATH, "r") as f:
                    loaded = json.load(f)
                    # Handle old format gracefully
                    if "facts" not in loaded:
                        print("[Memory] Old format detected, migrating")
                        return DEFAULT_MEMORY.copy()
                    print("[Memory] Loaded from disk")
                    return loaded
            except Exception as e:
                print(f"[Memory] Load error: {e}. Fresh start.")
                return DEFAULT_MEMORY.copy()
        return DEFAULT_MEMORY.copy()

    def _save(self):
        """Persist to disk."""
        self.memory["last_updated"] = datetime.now().isoformat()
        try:
            with open(MEMORY_FILE_PATH, "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"[Memory] Save error: {e}")

    def add_facts(self, facts: list):
        """
        Add extracted facts to memory.
        
        Each fact must have: key, value, category, priority
        Handles deduplication by key.
        Updates value if key already exists and value changed.
        """
        if not facts:
            return

        changed = False

        for fact in facts:
            key = fact.get("key", "").strip()
            value = fact.get("value", "").strip()
            priority = fact.get("priority", "contextual")
            category = fact.get("category", "information")

            if not key or not value:
                continue

            if priority not in ["critical", "important", "contextual"]:
                priority = "contextual"

            # Search all priority levels for existing key
            found = False
            for p in ["critical", "important", "contextual"]:
                for i, existing in enumerate(self.memory["facts"][p]):
                    if existing.get("key") == key:
                        # Key exists — update value if changed
                        if existing.get("value") != value:
                            print(
                                f"[Memory] Update [{p}] {key}: "
                                f"'{existing['value']}' → '{value}'"
                            )
                            self.memory["facts"][p][i]["value"] = value
                            self.memory["facts"][p][i]["category"] = category
                            changed = True
                        found = True
                        break
                if found:
                    break

            if not found:
                # New fact — add to appropriate priority bucket
                new_fact = {
                    "key": key,
                    "value": value,
                    "category": category,
                    "priority": priority,
                    "added_at_turn": self.memory["turn_count"]
                }
                self.memory["facts"][priority].append(new_fact)
                print(f"[Memory] New [{priority}] {key}: {value}")
                changed = True

        if changed:
            self._save()

    def remove_by_key(self, key: str):
        """Remove a fact by its key. Used by stale detector."""
        removed = False
        for priority in ["critical", "important", "contextual"]:
            before = len(self.memory["facts"][priority])
            self.memory["facts"][priority] = [
                f for f in self.memory["facts"][priority]
                if f.get("key") != key
            ]
            if len(self.memory["facts"][priority]) < before:
                print(f"[Memory] Removed fact: {key}")
                removed = True
        if removed:
            self._save()

    def remove_by_value_substring(self, substring: str):
        """
        Remove facts whose value contains a substring.
        Used by stale detector for broader cancellations.
        e.g., remove all facts mentioning "Bali"
        """
        removed_keys = []
        for priority in ["critical", "important", "contextual"]:
            to_remove = [
                f for f in self.memory["facts"][priority]
                if substring.lower() in f.get("value", "").lower()
            ]
            for f in to_remove:
                removed_keys.append(f["key"])
            self.memory["facts"][priority] = [
                f for f in self.memory["facts"][priority]
                if substring.lower() not in f.get("value", "").lower()
            ]
        if removed_keys:
            print(f"[Memory] Removed facts mentioning '{substring}': {removed_keys}")
            self._save()
        return removed_keys

    def add_cancelled(self, item: str):
        """Record something as explicitly cancelled."""
        if item not in self.memory["cancelled"]:
            self.memory["cancelled"].append(item)
            self._save()
            print(f"[Memory] Marked as cancelled: {item}")

    def add_decision(self, decision: str):
        """Record a confirmed decision."""
        if decision not in self.memory["decisions"]:
            self.memory["decisions"].append(decision)
            self._save()

    def get(self, key: str, default=None) -> Any:
        """Get fact value by key."""
        for priority in ["critical", "important", "contextual"]:
            for fact in self.memory["facts"][priority]:
                if fact.get("key") == key:
                    return fact.get("value", default)
        return default

    def get_all(self) -> dict:
        """Full memory snapshot."""
        return self.memory.copy()

    def get_all_facts_as_text_list(self) -> list:
        """
        Return all facts as plain text strings.
        Used by RAG system to embed and store in ChromaDB.
        """
        texts = []
        for priority in ["critical", "important", "contextual"]:
            for fact in self.memory["facts"][priority]:
                texts.append(fact.get("value", ""))
        return [t for t in texts if t]

    def get_critical_facts(self) -> list:
        return self.memory["facts"]["critical"]

    def get_important_facts(self) -> list:
        return self.memory["facts"]["important"]

    def increment_turn(self):
        self.memory["turn_count"] += 1
        self._save()

    def format_for_prompt(self) -> str:
        """
        Format memory for injection into LLM prompts.
        
        KEY DESIGN DECISION:
        This method presents facts as neutral statements.
        It does NOT add domain-specific instructions.
        It does NOT say "check for shellfish" or "watch budget".
        
        The LLM receives facts and uses its own reasoning
        to determine how each fact affects the current response.
        
        This is what makes the system agent-centric:
        the compression layer is dumb about domain knowledge,
        the LLM layer is smart about domain knowledge.
        
        Also enforces MAX_WORKING_MEMORY_TOKENS budget.
        """
        lines = []
        total_tokens = 0

        # Critical facts — always include
        critical = self.memory["facts"]["critical"]
        if critical:
            header = "[CRITICAL CONSTRAINTS]"
            lines.append(header)
            total_tokens += count_tokens(header)
            for fact in critical:
                line = f"  • {fact['value']}"
                line_tokens = count_tokens(line)
                if total_tokens + line_tokens > MAX_WORKING_MEMORY_TOKENS:
                    lines.append("  • [additional constraints truncated]")
                    break
                lines.append(line)
                total_tokens += line_tokens

        # Important facts — include if budget allows
        important = self.memory["facts"]["important"]
        if important:
            header = "[USER PREFERENCES]"
            header_tokens = count_tokens(header)
            if total_tokens + header_tokens < MAX_WORKING_MEMORY_TOKENS:
                lines.append(header)
                total_tokens += header_tokens
                for fact in important:
                    line = f"  • {fact['value']}"
                    line_tokens = count_tokens(line)
                    if total_tokens + line_tokens > MAX_WORKING_MEMORY_TOKENS:
                        break
                    lines.append(line)
                    total_tokens += line_tokens

        # Decisions — include if budget allows
        decisions = self.memory["decisions"]
        if decisions:
            header = "[DECISIONS MADE]"
            header_tokens = count_tokens(header)
            if total_tokens + header_tokens < MAX_WORKING_MEMORY_TOKENS:
                lines.append(header)
                total_tokens += header_tokens
                for decision in decisions[-5:]:  # Last 5 decisions
                    line = f"  • {decision}"
                    line_tokens = count_tokens(line)
                    if total_tokens + line_tokens > MAX_WORKING_MEMORY_TOKENS:
                        break
                    lines.append(line)
                    total_tokens += line_tokens

        # Cancelled items — important for stale context test
        cancelled = self.memory["cancelled"]
        if cancelled:
            header = "[CANCELLED — DO NOT REVISIT]"
            header_tokens = count_tokens(header)
            if total_tokens + header_tokens < MAX_WORKING_MEMORY_TOKENS:
                lines.append(header)
                total_tokens += header_tokens
                for item in cancelled:
                    line = f"  • {item}"
                    line_tokens = count_tokens(line)
                    if total_tokens + line_tokens > MAX_WORKING_MEMORY_TOKENS:
                        break
                    lines.append(line)
                    total_tokens += line_tokens

        if not lines:
            return "[NO USER PREFERENCES CAPTURED YET]"

        return "\n".join(lines)

    def reset(self):
        """Reset for new conversation."""
        self.memory = DEFAULT_MEMORY.copy()
        self._save()
        print("[Memory] Reset complete")