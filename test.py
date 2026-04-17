# test_full_system.py
# Tests the complete system end to end
# Run this to verify everything works before evaluation

import sys
import os

# Silence ChromaDB telemetry before anything else
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

sys.path.append('.')

def test_1_memory_extraction():
    """Test that facts are extracted with correct priority."""
    print("\n" + "="*55)
    print("TEST 1: Memory Extraction Priority")
    print("="*55)

    from ccm.memory_store import WorkingMemory
    from ccm.extractor import MemoryExtractor

    memory = WorkingMemory()
    memory.reset()
    extractor = MemoryExtractor()

    msg = (
        "I want to plan a 10-day trip to Tokyo and Kyoto. "
        "My total budget is $3000 maximum. "
        "I am severely allergic to shellfish — medical requirement. "
        "I prefer a relaxed pace with maximum 2 activities per day."
    )

    facts = extractor.extract_and_update(msg, memory)

    print(f"\nExtracted {len(facts)} facts:")
    for f in facts:
        print(f"  [{f['priority']:11}] {f['key']}: {f['value']}")

    print("\nFormatted for prompt:")
    print(memory.format_for_prompt())

    # Verify critical facts
    critical = memory.get_critical_facts()
    critical_values = [f['value'].lower() for f in critical]

    allergy_critical = any('shellfish' in v for v in critical_values)
    print(f"\nShellfish allergy is CRITICAL: {allergy_critical}")

    if not allergy_critical:
        print("⚠️  WARNING: Allergy not marked critical!")
        print("    Check EXTRACTION_PROMPT priority rules")
    else:
        print("✅ TEST 1 PASSED")

    return allergy_critical


def test_2_stale_detection():
    """Test stale context detection and cleanup."""
    print("\n" + "="*55)
    print("TEST 2: Stale Context Detection")
    print("="*55)

    from ccm.memory_store import WorkingMemory
    from ccm.stale_detector import StaleDetector
    from ccm.episodic_memory import EpisodicMemory
    from ccm.semantic_memory import SemanticMemory

    memory = WorkingMemory()
    memory.reset()
    episodic = EpisodicMemory()
    episodic.reset()
    semantic = SemanticMemory()
    semantic.reset()
    detector = StaleDetector()

    # Add Bali plans
    memory.add_facts([{
        "key": "destination_primary",
        "value": "Bali beach vacation",
        "category": "decision",
        "priority": "important"
    }])

    # Store episodic memory about Bali
    # Note: text says "Bali resorts" not "Bali beach vacation"
    # This tests whether our word-level matching works
    episodic.add(
        "Researched Bali resorts. Found beach hotels $150-300/night.",
        turn_range=(1, 3)
    )
    semantic.add(
        "Bali flights: $950 roundtrip from NYC. Peak season prices.",
        tool_name="web_search",
        query_used="flights to Bali",
        turn_number=2
    )

    print("Added Bali plans to memory and ChromaDB")

    # User cancels Bali
    pivot_msg = (
        "Actually scratch Bali completely. "
        "Let us do Switzerland instead."
    )
    result = detector.check_and_clean(
        pivot_msg, memory, episodic, semantic
    )

    print(f"\nOverride detected: {result['has_override']}")
    print(f"Overridden keys: {result.get('overridden_keys', [])}")

    # Check Bali is gone from episodic
    import time
    time.sleep(0.5)  # Small delay for ChromaDB to update
    bali_episodic = episodic.retrieve("Bali resorts")
    bali_semantic = semantic.retrieve("Bali flights")

    print(f"Bali in episodic after stale: {len(bali_episodic)} (want 0)")
    print(f"Bali in semantic after stale: {len(bali_semantic)} (want 0)")
    print(f"\nMemory cancelled list: {memory.get_all().get('cancelled', [])}")

    passed = (
        result['has_override'] and
        len(bali_episodic) == 0 and
        len(bali_semantic) == 0
    )
    print(f"\n{'✅ TEST 2 PASSED' if passed else '❌ TEST 2 FAILED'}")
    return passed

def test_3_tool_compression():
    """Test that tool results are compressed correctly."""
    print("\n" + "="*55)
    print("TEST 3: Tool Result Compression")
    print("="*55)

    from ccm.compressor import ToolCompressor
    from travel_agent.tools import places_search

    compressor = ToolCompressor()

    raw = places_search("Tokyo", "restaurants")
    raw_str = str(raw)
    raw_tokens = len(raw_str) // 4

    constraints = ["severely allergic to shellfish"]
    compressed = compressor.compress(raw, "places_search", constraints)
    compressed_tokens = len(compressed) // 4

    print(f"Raw tokens (approx):        {raw_tokens}")
    print(f"Compressed tokens (approx): {compressed_tokens}")
    ratio = raw_tokens / max(compressed_tokens, 1)
    print(f"Compression ratio:          {ratio:.1f}x")
    print(f"\nCompressed output:\n{compressed}")

    # Check shellfish mentioned in compressed output
    shellfish_flagged = (
        'shellfish' in compressed.lower() or
        '⚠️' in compressed or
        'allergy' in compressed.lower()
    )
    print(f"\nShellfish flagged in compressed: {shellfish_flagged}")

    passed = ratio > 2.0
    print(f"{'✅ TEST 3 PASSED' if passed else '❌ TEST 3 FAILED'}")
    return passed


def test_4_rag_retrieval():
    """Test RAG retrieval finds relevant memories."""
    print("\n" + "="*55)
    print("TEST 4: RAG Retrieval")
    print("="*55)

    from ccm.episodic_memory import EpisodicMemory
    from ccm.semantic_memory import SemanticMemory
    from ccm.retriever import Retriever

    ep = EpisodicMemory()
    sem = SemanticMemory()
    ep.reset()
    sem.reset()

    # Store some memories
    ep.add(
        "User has shellfish allergy. Budget set at $3000.",
        turn_range=(0, 1)
    )
    ep.add(
        "Booked ANA flight NYC to Tokyo for $780 direct.",
        turn_range=(2, 3)
    )
    ep.add(
        "Searched Tokyo hotels. Shinjuku Park $120 shortlisted.",
        turn_range=(4, 5)
    )
    sem.add(
        "Tsukiji restaurants: Sushi Dai (shellfish heavy, avoid), "
        "Odayasu (safe, traditional Japanese, $$)",
        tool_name="places_search",
        query_used="restaurants near Tsukiji",
        turn_number=6
    )

    retriever = Retriever(ep, sem, use_reranking=False)

    # Query about restaurants — should find allergy memory
    print("\nQuery: 'find dinner restaurants Tsukiji'")
    results = retriever.retrieve(
        "find dinner restaurants Tsukiji",
        n_episodic=3,
        n_semantic=2
    )

    print(f"Episodic retrieved: {len(results['episodic'])}")
    for r in results['episodic']:
        print(f"  [{r['similarity']:.2f}] {r['text'][:60]}")

    print(f"Semantic retrieved: {len(results['semantic'])}")
    for r in results['semantic']:
        print(f"  [{r['similarity']:.2f}] {r['text'][:60]}")

    passed = (
        len(results['episodic']) > 0 or
        len(results['semantic']) > 0
    )
    print(f"\n{'✅ TEST 4 PASSED' if passed else '❌ TEST 4 FAILED'}")
    return passed


def test_5_ccm_agent_allergy():
    """
    THE KEY TEST:
    State allergy in turn 1.
    Ask about restaurants in turn 4.
    Agent must remember and warn about shellfish.
    """
    print("\n" + "="*55)
    print("TEST 5: CCM Agent Remembers Allergy (4 turns)")
    print("="*55)

    from travel_agent.agent import CCMAgent

    agent = CCMAgent(use_reranking=False)
    agent.reset()

    turns = [
        "I want to plan a trip to Tokyo and Kyoto. "
        "Budget is $3000 maximum. "
        "I am severely allergic to shellfish.",

        "Find me flights from New York to Tokyo in June",

        "Search for hotels in Shinjuku area please",

        # KEY TURN — must warn about shellfish
        "Find me the best dinner spots near Tsukiji market"
    ]

    responses = []
    for i, msg in enumerate(turns):
        print(f"\nTurn {i+1}: {msg[:60]}...")
        result = agent.chat(msg)
        responses.append(result['response'])
        print(f"Tokens: {result['tokens_in_context']}")
        print(f"Response preview: {result['response'][:150]}")

    # Check final response
    final = responses[-1].lower()
    allergy_words = [
        'shellfish', 'allergy', 'allergic',
        'seafood', '⚠️', 'warning', 'avoid'
    ]
    allergy_remembered = any(w in final for w in allergy_words)

    print(f"\n--- FULL FINAL RESPONSE ---")
    print(responses[-1])
    print(f"\nAllergy mentioned in response: {allergy_remembered}")

    print(
        f"\n{'✅ TEST 5 PASSED — CCM remembered allergy!' if allergy_remembered else '❌ TEST 5 FAILED — allergy forgotten'}"
    )
    return allergy_remembered


def test_6_baseline_fails():
    """
    Show baseline agent forgetting allergy.
    This is the BEFORE state for the demo.
    """
    print("\n" + "="*55)
    print("TEST 6: Baseline Agent (should forget allergy)")
    print("="*55)

    from travel_agent.baseline_agent import BaselineAgent

    agent = BaselineAgent()
    agent.reset()

    turns = [
        "I want to plan a trip to Tokyo. "
        "Budget is $3000. "
        "I am severely allergic to shellfish.",

        "Find flights from New York to Tokyo",
        "Search for hotels in Tokyo",
        "What is the weather in Tokyo?",
        "Find restaurants near Tsukiji market"
    ]

    responses = []
    tokens_per_turn = []

    for i, msg in enumerate(turns):
        print(f"\nTurn {i+1}: {msg[:60]}...")
        result = agent.chat(msg)
        responses.append(result['response'])
        tokens_per_turn.append(result['tokens_in_context'])
        print(f"Tokens: {result['tokens_in_context']}")

    final = responses[-1].lower()
    allergy_words = ['shellfish', 'allergy', 'allergic', 'warning']
    allergy_remembered = any(w in final for w in allergy_words)

    print(f"\nToken growth: {tokens_per_turn}")
    print(f"Baseline remembered allergy: {allergy_remembered}")
    print(f"(We WANT this to be False to show the problem)")

    print(f"\n{'✅ Baseline correctly FAILED (proves the problem)' if not allergy_remembered else '⚠️  Baseline passed — problem not visible yet (run with more turns)'}")

    return not allergy_remembered  # Pass if baseline fails


if __name__ == "__main__":
    print("\n" + "="*55)
    print("FULL SYSTEM TEST")
    print("="*55)

    results = {}

    results['memory_extraction'] = test_1_memory_extraction()
    results['stale_detection'] = test_2_stale_detection()
    results['compression'] = test_3_tool_compression()
    results['rag_retrieval'] = test_4_rag_retrieval()
    results['ccm_allergy'] = test_5_ccm_agent_allergy()
    results['baseline_fails'] = test_6_baseline_fails()

    print("\n" + "="*55)
    print("FINAL RESULTS")
    print("="*55)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    total = sum(results.values())
    print(f"\n{total}/{len(results)} tests passed")