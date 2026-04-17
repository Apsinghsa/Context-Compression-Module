# ccm/stale_detector.py
#
# AGENT-CENTRIC DESIGN:
#   Works on ANY memory key, not just destinations/preferences.
#   Does not know about travel concepts.
#   Detects overrides based on memory key matching,
#   not hardcoded field names.

import os
import json
from groq import Groq
from dotenv import load_dotenv
from travel_agent.prompts import STALE_DETECTION_PROMPT

load_dotenv()

# Fast pre-filter — skip LLM call if none of these are present
# These are language patterns, not domain concepts
OVERRIDE_SIGNALS = [
    "scratch", "forget", "cancel", "instead", "actually",
    "change", "never mind", "drop", "not anymore",
    "changed my mind", "no longer", "skip", "disregard",
    "replace", "switch", "swap", "rather", "different"
]


class StaleDetector:
    """
    Detects and removes stale/overridden context from memory.

    Domain agnostic — works on memory keys and values,
    not on specific travel concepts like destinations.

    When user says "actually forget Paris, let's do Rome instead":
    1. Detects the override signal
    2. Asks LLM which memory keys are being overridden
    3. Removes those keys from working memory
    4. Marks overridden items as cancelled
    5. Marks ChromaDB memories mentioning those items as stale
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def _has_override_signal(self, message: str) -> bool:
        """Fast keyword check before spending an LLM call."""
        message_lower = message.lower()
        return any(signal in message_lower for signal in OVERRIDE_SIGNALS)

    def check_and_clean(
        self,
        user_message: str,
        working_memory,
        episodic_memory=None
    ) -> dict:
        """
        Check message for overrides and clean memory accordingly.

        Parameters:
          user_message:    Current user input
          working_memory:  WorkingMemory instance
          episodic_memory: Optional EpisodicMemory instance
                          If provided, marks stale items there too

        Returns:
          dict with has_override, overridden_keys, reason
        """
        # Skip LLM call if no override signals
        if not self._has_override_signal(user_message):
            return {
                "has_override": False,
                "overridden_keys": [],
                "reason": ""
            }

        print("[StaleDetector] Override signal detected, checking...")

        # Get current memory to check against
        current_memory = working_memory.get_all()

        # Build flat view of current facts for the LLM
        # Just keys and values — LLM does not need full structure
        flat_memory = {}
        for priority in ["critical", "important", "contextual"]:
            for fact in current_memory["facts"][priority]:
                flat_memory[fact["key"]] = fact["value"]

        # Add decisions and cancelled for completeness
        flat_memory["_decisions"] = current_memory.get("decisions", [])
        flat_memory["_cancelled"] = current_memory.get("cancelled", [])

        prompt = STALE_DETECTION_PROMPT.format(
            message=user_message,
            current_memory=json.dumps(flat_memory, indent=2)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You detect when users override previous statements. "
                            "Return only valid JSON."
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

            raw = response.choices[0].message.content.strip()

            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            result = json.loads(raw)

            # Apply the overrides to memory
            if result.get("has_override") and result.get("overridden_keys"):
                overridden = result["overridden_keys"]
                print(f"[StaleDetector] Overriding keys: {overridden}")

                for key in overridden:
                    # Get the value before removing
                    # so we can mark it cancelled
                    old_value = working_memory.get(key)

                    # Remove from working memory
                    working_memory.remove_by_key(key)

                    # If the value is meaningful, mark as cancelled
                    if old_value and isinstance(old_value, str):
                        working_memory.add_cancelled(old_value)

                        # Also mark stale in episodic memory if provided
                        if episodic_memory:
                            count = episodic_memory.mark_stale_by_content(
                                old_value
                            )
                            if count > 0:
                                print(
                                    f"[StaleDetector] Marked {count} "
                                    f"episodic memories as stale"
                                )

            return result

        except json.JSONDecodeError as e:
            print(f"[StaleDetector] JSON error: {e}")
            return {
                "has_override": False,
                "overridden_keys": [],
                "reason": ""
            }

        except Exception as e:
            print(f"[StaleDetector] Error: {e}")
            return {
                "has_override": False,
                "overridden_keys": [],
                "reason": ""
            }