# ccm/extractor.py
#
# AGENT-CENTRIC DESIGN:
#   Extracts facts from ANY user message in ANY domain.
#   Uses abstract classification (priority/category) not
#   domain-specific fields (allergy/budget/destination).
#
#   The same extractor works for:
#   Travel agent — extracts destinations, budgets, allergies
#   Medical agent — extracts symptoms, medications, allergies
#   Legal agent  — extracts case details, deadlines, constraints
#   Code agent   — extracts requirements, tech stack, constraints
#
#   No changes needed to this file for different domains.

import os
import json
from groq import Groq
from dotenv import load_dotenv
from travel_agent.prompts import EXTRACTION_PROMPT

load_dotenv()


class MemoryExtractor:
    """
    Domain-agnostic fact extractor.

    Takes raw user messages and returns structured facts
    ready to be stored in WorkingMemory.

    Uses the LLM to understand meaning and classify facts
    by priority and category — not by specific keywords.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # 8B model is sufficient for structured extraction
        # with a well-designed prompt
        self.model = "llama-3.1-8b-instant"

    def extract(self, user_message: str, current_memory: dict) -> list:
        """
        Extract facts from a single user message.

        Returns list of fact dicts.
        Each dict: {key, value, category, priority}
        Returns empty list if nothing to extract.

        Does not modify memory — just returns what was found.
        """
        # Quick pre-check: very short messages rarely have
        # new facts worth storing
        if len(user_message.strip()) < 10:
            return []

        prompt = EXTRACTION_PROMPT.format(
            message=user_message,
            current_memory=json.dumps(
                current_memory.get("facts", {}),
                indent=2
            )
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract facts and return JSON. "
                            "Return only valid JSON. "
                            "Never include markdown, explanation, or "
                            "text outside the JSON structure."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.0  # Deterministic extraction
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown if model added it
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            # Find JSON object in response
            # Sometimes model adds a sentence before the JSON
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            parsed = json.loads(raw)
            facts = parsed.get("facts", [])

            # Validate each fact has required fields
            valid_facts = []
            for f in facts:
                if (
                    isinstance(f, dict) and
                    f.get("key") and
                    f.get("value") and
                    f.get("priority") in ["critical", "important", "contextual"]
                ):
                    valid_facts.append(f)
                else:
                    print(f"[Extractor] Skipped malformed fact: {f}")

            if valid_facts:
                print(f"[Extractor] Found {len(valid_facts)} facts:")
                for f in valid_facts:
                    print(
                        f"  [{f['priority']:11}] "
                        f"{f['key']:30} → {f['value']}"
                    )
            else:
                print("[Extractor] No new facts in this message")

            return valid_facts

        except json.JSONDecodeError as e:
            print(f"[Extractor] JSON parse failed: {e}")
            print(f"[Extractor] Raw response: {raw[:200]}")
            return []

        except Exception as e:
            print(f"[Extractor] Error: {e}")
            return []

    def extract_and_update(
        self,
        user_message: str,
        working_memory
    ) -> list:
        """
        Extract facts and store them in working memory.

        This is the main entry point called by CCM core.
        Returns list of what was extracted for logging.
        """
        current_state = working_memory.get_all()
        facts = self.extract(user_message, current_state)

        if facts:
            working_memory.add_facts(facts)

        return facts