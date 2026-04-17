# test_evaluation_quick.py
# Runs just Test A (shellfish allergy) on CCM agent only
# Fastest way to verify evaluation works

import sys
sys.path.append('.')

from evaluation.run_evaluation import run_full_evaluation
from evaluation.test_conversations import TEST_A_SHELLFISH_ALLERGY

print("Running Test A only on CCM agent...")
print("This tests if shellfish allergy is remembered at turn 15")
print("Takes about 3-4 minutes\n")

results = run_full_evaluation(
    run_baseline=False,
    tests_to_run=[TEST_A_SHELLFISH_ALLERGY]
)

ccm = results['ccm'][0]
print(f"\nTest A passed: {ccm['passed']}")
print(f"Key response: {ccm['key_response'][:300]}")