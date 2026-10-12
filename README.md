# A Policy-Aware Agentic System for Compliance Reasoning

Bachelor thesis, Media Engineering and Technology Faculty, German University in Cairo.
**Grade: A+** | Supervisor: Dr. Nourhan Ehab

## Overview
Compliance automation tends to fail in predictable ways. Rule-based systems apply thresholds exactly but cannot read free text. Language models read free text fluently but drift on numeric rules, contradict themselves across runs, and forget earlier context.

This project keeps each component responsible for what it does best. Language models handle language only: extracting structured fields from a request and explaining the final verdict in plain words. Every decision in between is made by deterministic logic.

## Architecture
A LangGraph pipeline with shared typed state and conditional routing:

| Node | Role |
|------|------|
| **Interpreter** | LLM extracts structured fields from the request using constrained decoding (strict JSON schema) |
| **Validation** | Deterministic checks catch malformed input before it reaches the policy engine |
| **Policy Matching** | Evaluates formal policy conditions against the extracted data, with no model involved |
| **Memory** | Tracks cumulative usage such as leave balances and borrowing limits, separate from the statutory rules |
| **Decision** | Produces the final verdict from the matched policies and memory state |
| **Human in the Loop** | Cases that do not resolve cleanly are escalated to a reviewer with the full audit trail |
| **Explanation** | LLM turns the verdict into plain language, grounded only in the triggered policy text |

`graph.png` in the repo root shows the compiled pipeline graph.

## Evaluation
Tested on **400 cases across two domains**: employee leave management under Egyptian private-sector labor law, and library book circulation. Two LLM-only baselines (`baseline_a.py`, `baseline_b.py`) using the same underlying model were used for comparison.

| Metric | Leave management | Library circulation |
|--------|------------------|---------------------|
| Accuracy | 81.0% | 87.5% |
| Macro F1 | 0.794 | 0.850 |
| Memory-dependent cases | 100% | 100% |
| Run-to-run consistency | 92.0% | 99.5% |

- **Memory-dependent cases:** the strongest baseline scored 20% on leave and 0% on circulation.
- **Explanations:** 18 of 20 sampled explanations traced directly back to the policy text they cited.
- **Ablation:** removing the interpreter costs 42 accuracy points on leave and 38.5 on circulation. Removing memory and validation together costs more than removing either alone, which shows they cover different failure modes.

## Tech stack
- Python
- LangGraph
- OpenAI API (GPT-4o mini, with Structured Outputs)
- JSON schema validation

## Repository structure
```
Policy-Aware-Agentic-System/
├── engine/                      # Deterministic policy engine
├── graph/                       # LangGraph pipeline definition and shared state
├── interpreter/                 # LLM interpreter (constrained JSON extraction)
├── nodes/                       # Validation, policy matching, memory, decision, human review, explanation
├── policies/                    # Formal policy definitions for each domain
├── tools/                       # Helper tools used by the pipeline
├── utils/                       # Shared utilities
├── results/                     # Saved evaluation outputs
│
├── main.py                      # Entry point
├── run_system.py                # Runs the full pipeline on a request
├── ui.py                        # User interface
│
├── baseline_a.py                # LLM-only baseline A
├── baseline_b.py                # LLM-only baseline B
├── evaluation.py                # Metrics and evaluation runner
├── evaluation_dataset.py        # Evaluation dataset
├── gen_benchmark.py             # Benchmark case generator
├── leave_benchmark_cases.py     # Leave management benchmark cases
├── batch1_eval.py               # Evaluation batch 1
├── batch2_eval.py               # Evaluation batch 2
├── batch3_eval.py               # Evaluation batch 3
├── test_policies.py             # Policy unit tests
├── evaluation_checkpoint_main.csv   # Evaluation checkpoint
├── evaluation_results.xlsx      # Aggregated evaluation results
│
├── graph.png                    # Pipeline graph visualization
├── .env.example                 # Environment variable template
└── .gitignore
```

## Run it
1. Clone the repository:
```bash
git clone https://github.com/IbrahiimHassan/Policy-Aware-Agentic-System.git
cd Policy-Aware-Agentic-System
```
2. Install the dependencies:
```bash
pip install langgraph openai jsonschema python-dotenv
```
3. Set your OpenAI API key. Copy the template and fill in your key:
```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=your_key_here
```
4. Run the pipeline:
```bash
python main.py
```

### Reproduce the evaluation
```bash
python test_policies.py     # check the policy definitions
python batch1_eval.py       # run evaluation batches
python batch2_eval.py
python batch3_eval.py
python evaluation.py        # compute metrics
```
Baselines can be run separately with `python baseline_a.py` and `python baseline_b.py`. Saved outputs are in `results/` and `evaluation_results.xlsx`.

## Key idea
Keep language models on language. Hand decisions to deterministic logic.

## Author
Ibrahim Hassan | [LinkedIn](https://linkedin.com/in/ibrahim-hassan)
