# Zigzag Persistent Homology Engine for ETFs

Implements zigzag persistent homology (Carlsson & de Silva 2010) on ETF correlation distance matrices. Tracks topological features (connected components and cycles) as the filtration parameter moves both forward and backward. The per‑ETF score captures participation in regime cycles missed by standard persistence.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Distance matrix = 1 - |correlation|
- Zigzag persistence approximated via forward/backward filtrations
- Score = average distance × (1 + clustering coefficient)
- Walk‑forward backtest validates predictive power
- Three‑tab Streamlit dashboard (auto best, manual, backtest)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-persistent-homology-zigzag-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (runtime ~5‑10 minutes)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- The score combines a vertex’s average distance (centrality) and its clustering coefficient.
- High score → the ETF is central and lies in many cycles (loops) that persist when the filtration goes backward.
- This captures “regime cycles” that standard persistence ignores.

## Requirements

See `requirements.txt`.
