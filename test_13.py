# test_ccm_core.py
import sys
sys.path.append('.')

from ccm.ccm_core import ContextCompressionModule
from travel_agent.tools import web_search, places_search

print("=" * 60)
print("TESTING CCM CORE — END TO END")
print("=" * 60)

ccm = ContextCompressionModule(use_reranking=False)
ccm.reset()

# Simulate turn 1
print("\n--- TURN 1: User states constraints ---")
msg1 = ("I want to plan a 10-day trip to Tokyo and Kyoto. "
        "Budget is $3000. I am severely allergic to shellfish.")
context1 = ccm.process_user_message(msg1)
print("\nAssembled context:")
print(context1)

# Simulate tool call
print("\n--- TOOL CALL: Flight search ---")
raw_result = web_search("flights from New York to Tokyo")
compressed = ccm.process_tool_result(
    "web_search", raw_result, "flights NYC to Tokyo"
)
print(f"Compressed: {compressed}")

# Simulate agent response
ccm.process_agent_response(
    msg1,
    "I found great flights to Tokyo! ANA offers direct flights "
    "for $780. Given your $3000 budget, this leaves $2220 for "
    "hotels and activities. Shall I search for hotels next?"
)

# Simulate turn 2
print("\n--- TURN 2: Follow up ---")
msg2 = "Yes find hotels in Shinjuku area please"
context2 = ccm.process_user_message(msg2)
print(f"\nContext tokens: {ccm.assembler.get_last_token_count()}")

# Get memory state
state = ccm.get_memory_state()
print("\n--- MEMORY STATE ---")
print(f"Turn count: {state['turn_count']}")
print(f"Episodic memories: {state['episodic_count']}")
print(f"Semantic memories: {state['semantic_count']}")
print(f"Token metrics: {state['token_metrics']}")
print(f"\nWorking memory:\n{state['working_memory']}")

print("\n✅ CCM Core working end-to-end")