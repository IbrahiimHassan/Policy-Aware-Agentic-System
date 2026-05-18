import sys
from pathlib import Path

# ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from leave_benchmark_cases import _build_cases
from interpreter.llm_interpreter import heuristic_extract

# list of disagreement case ids from results/disagreement_baselineB_vs_pipeline.csv
case_ids = [
    "TC_004",
    "TC_012",
    "TC_021",
    "TC_025",
    "TC_067",
    "TC_071",
    "TC_103",
    "TC_104",
    "TC_105",
    "TC_106",
    "TC_107",
    "TC_108",
    "TC_109",
    "TC_110",
    "TC_111",
    "TC_112",
    "TC_138",
]

cases = {c['case_id']: c for c in _build_cases()}

for cid in case_ids:
    c = cases.get(cid)
    if c is None:
        print(cid, "-- not found in CASES")
        continue
    text = c['input']
    inferred = heuristic_extract(text)
    print(cid, "->", inferred, "| note:", c.get('notes'))
