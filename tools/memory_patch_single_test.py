import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from leave_benchmark_cases import _build_cases
from run_system import run_system
from engine.policy_engine import policy_checker
from utils.config import POLICY_JSON_PATH
import json

cases = _build_cases()
# Try to find a memory case where stored + requested > cap so memory will block.
from interpreter.llm_interpreter import heuristic_extract
from utils.config import POLICY_LIMITS

case = None
for c in cases:
    patch = c.get('_memory_store_patch')
    if not patch:
        continue
    # find stored value under patch.leave.*_already_taken_this_period
    leave = patch.get('leave') or {}
    if not isinstance(leave, dict):
        continue
    # check known keys
    for key, stored in leave.items():
        try:
            stored_i = int(stored)
        except Exception:
            continue
        # infer requested days from text
        inferred = heuristic_extract(c.get('input') or '')
        req = inferred.get('days_requested')
        if req is None:
            # fallback: try to parse a digit in the input
            import re
            m = re.search(r"(\d+)\s+days?", (c.get('input') or '').lower())
            if m:
                req = int(m.group(1))
        if req is None:
            continue
        # map stored key to cap
        if 'casual' in key:
            cap = POLICY_LIMITS.get('casual_max_approved_days', 3)
        elif 'sick' in key:
            cap = 7
        else:
            cap = POLICY_LIMITS.get('annual_days_pro_rata_cap', 30)
        if stored_i + req > cap:
            case = c
            break
    if case:
        break

if case is None:
    # fallback: first memory_reject
    case = next((c for c in cases if c.get('category') == 'memory_reject'), None)

if case is None:
    print('No memory_reject case found in CASES')
    sys.exit(1)

cid = case['case_id']
req = case['input']
patch = case.get('_memory_store_patch')
print(f'Case: {cid}')
print('Request:', req)
print('Memory patch present:', bool(patch))
if patch:
    print('Patch:', json.dumps(patch, ensure_ascii=False))

print('\n-- RUN without memory_patch --')
out1 = run_system(req, memory_store_patch=None)
print('Decision:', out1.get('decision'))
print('Extracted JSON:', out1.get('json'))
print('Explanation:', (out1.get('explanation') or '')[:400])
pol1 = policy_checker(out1.get('json') or {}, POLICY_JSON_PATH)
print('Skipped policies:', json.dumps(pol1.get('skipped_policies', []), ensure_ascii=False))

if patch:
    print('\n-- RUN with memory_patch --')
    out2 = run_system(req, memory_store_patch=patch)
    print('Decision:', out2.get('decision'))
    print('Extracted JSON:', out2.get('json'))
    print('Explanation:', (out2.get('explanation') or '')[:400])
    pol2 = policy_checker(out2.get('json') or {}, POLICY_JSON_PATH)
    print('Skipped policies:', json.dumps(pol2.get('skipped_policies', []), ensure_ascii=False))

print('\nDone')
