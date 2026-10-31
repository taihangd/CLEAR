# Trajectory Recovery via Constrained LLM Reasoning and Difficulty-Aware Routing

Reference implementation and supplementary material for cluster-level road-network trajectory recovery (`C1` applicability-gated transition retrieval with $G_5 / G_5^{\text{kin}}$ boundary clamping, `C2` constrained road-intent planning and three-tier bounded Dijkstra verification, and `C3` corridor subgraph pruning $\mathcal{G}_q$ with difficulty-aware multi-LLM routing).

---

## Supplementary Material

- **[`supplementary_material.pdf`](supplementary_material.pdf)**: Complete technical appendix accompanying the paper, including:
  - **Appendix A**: Extended five-split multi-metric benchmark ledgers (`UrbanVehicle` & `Carla`) and LCSS timestamp-projection invariance analysis.
  - **Appendix B**: Full mathematical proofs for Proposition 1 ($G_5 / G_5^{\text{kin}}$ boundary safety) and Proposition 2 ($\mathcal{G}_q$ lossless Tier 1–3 shortest-path equivalence).
  - **Appendix C**: `C2` structured prompt contract, three-tier verification ablation, and multi-seed synthetic outlier robustness.
  - **Appendix D**: `C3` 18-dimensional router feature specification, experience-bank cold-start scaling, and corridor subgraph pruning breakdown.

---

## Repository Structure

```text
.
├── supplementary_material.pdf # Technical appendix (Appendices A-D)
├── main.py                    # Main C1 -> C3 -> C2 pipeline entrypoint
├── traj_rec_st_cot.py         # LLM client and C2 structured road-intent interface
├── multi_llm_router.py        # C3 dual-regressor router with quality tolerance delta
├── cluster_algorithm.py       # Upstream visual-spatiotemporal clustering (SigCluster)
├── common.py                  # Directed road-graph construction and trajectory utilities
├── load_data.py               # Observation feature loader
├── eval.py                    # Multi-metric evaluator (Macro/Micro F1, STLC, LCSS, Vol-MSE)
├── utils/                     # Core C1, C2, and C3 algorithmic modules
├── config/uv.yaml             # Example dataset and model configuration template
└── tests/                     # Unit tests for C1, C2, and C3 core logic
```

---

## Setup & Configuration

1. Install dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```

2. Configure dataset paths in [`config/uv.yaml`](config/uv.yaml) and export the provider API keys (OpenAI and Alibaba Cloud DashScope):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export DASHSCOPE_API_KEY="your-dashscope-api-key"
   ```

---

## Quick Verification

1. Run the `C1`–`C3` core unit test suite:
   ```bash
   pytest -q
   # or:
   python3 tests/test_traj_rec_core.py
   ```

2. Run the trajectory recovery pipeline with the example configuration [`config/uv.yaml`](config/uv.yaml):
   ```bash
   python3 main.py --cfg_file config/uv.yaml --allow-api
   ```
