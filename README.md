# QGA: Quantum Geometric Alpha

## Heat Kernel Signatures, p-adic Market Trees, Ricci Liquidity Curvature, Persistent Homology, and Path-Integral-Inspired Market Simulation

QGA is a quantitative research platform that represents financial markets as evolving geometric and topological structures. The project explores whether market stress, regime transitions, and asset dislocations can be captured by combining graph diffusion, hierarchical ultrametric structure, discrete curvature, persistent topology, and Monte Carlo path ensembles.

## Project Overview

QGA treats market behavior as more than a sequence of prices. Instead, it frames financial data as geometry and topology that evolve over time. The system uses mathematical signatures to describe: market hierarchy, diffusion of stress, liquidity curvature, regime structure, and probabilistic path ensembles.

The result is a research-oriented framework for building composite market signals and evaluating them through historical backtesting, visualization, and scenario analysis.

## Core Research Question

Can geometric, topological, and probabilistic signatures reveal market stress, regime transitions, or asset dislocations before they become obvious in traditional indicators?

This project is designed to test that question using a structured collection of mathematical engines, rather than to provide guaranteed market timing.

## Mathematical Engines

- **p-adic-inspired ultrametric market tree**
  - Constructs a hierarchical market topology where asset similarity is derived from shared market structure.
  - This is a hierarchy-inspired ultrametric model, not a literal implementation of p-adic arithmetic.

- **heat kernel diffusion on asset graphs**
  - Uses heat diffusion over a correlation graph to measure how volatility and stress propagate across the market.
  - Diffused heat provides a market stress signature that can highlight concentration effects.

- **discrete Ricci curvature liquidity stress**
  - Computes a practical discrete curvature proxy on asset graphs.
  - Negative curvature is interpreted as a fragile or stressed connectivity region, while positive curvature indicates structural support.

- **persistent homology regime detection**
  - Applies topological data analysis to identify market regimes and structural complexity.
  - Regimes are represented through persistent homology summaries and rule-based labels.

- **path-integral-inspired Monte Carlo simulator**
  - Simulates many future paths for an asset and scores them with an economic action functional.
  - Path likelihoods are weighted by lower action scores to create a probabilistic terminal distribution.
  - This is a metaphorical path-integral framework, not a literal quantum prediction model.

- **unified geometric signal engine**
  - Combines curvature, heat, topology, path probabilities, and ultrametric isolation into a single interpretable score.
  - The score is intended as a research signal rather than investment advice.

- **backtesting engine**
  - Converts geometric signals into portfolio weights, rebalances positions, applies transaction costs, and compares performance against benchmarks.
  - Provides a historical simulation framework for evaluating the signal pipeline.

- **visualization dashboard**
  - Generates charts for price history, return correlations, heat diffusion, curvature, topology, signal rankings, equity curves, drawdown, and rolling Sharpe.
  - Supports research interpretation with clean, professional plots.

## Repository Structure

- `src/data` — data loading, cleaning, and feature engineering utilities
- `src/geometry` — market graph, heat kernel, p-adic tree, and curvature modules
- `src/topology` — persistent homology and regime signature analysis
- `src/quantum` — path-integral-inspired simulation and probabilistic distribution modeling
- `src/signals` — composite geometric signal construction and asset scoring
- `src/backtest` — historical simulation engine, benchmarks, and performance metrics
- `src/visualization` — dashboard and plotting routines for research reports
- `tests` — unit tests for core modules and algorithms
- `docs` — research notes, mathematical framework, experiment design, and limitations
- `notebooks` — interactive examples and exploratory analysis notebooks

## Installation

```bash
git clone https://github.com/Elsbernd09/qga-heatkernel.git
cd qga-heatkernel
python -m pip install -r requirements.txt
```

## How to Run

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the full demo:

```bash
python scripts/run_qga_demo.py
```

3. Run the tests:

```bash
pytest
```

4. Run the notebook-style research walkthrough:

```bash
python notebooks/qga_research_demo.py
```

## Quick Start

```python
import yfinance as yf
import pandas as pd
from src.visualization.dashboard import QGADashboard
from src.geometry.heat_kernel import HeatKernelDiffusion
from src.geometry.ricci_curvature import RicciCurvatureEngine
from src.signals.geometric_signal import GeometricSignalEngine
from src.backtest.engine import BacktestEngine

# Load sample market data
symbols = ["AAPL", "MSFT", "SPY"]
prices = yf.download(symbols, start="2023-01-01", end="2024-01-01")["Adj Close"]
returns = prices.pct_change().dropna()

# Compute heat diffusion and curvature
heat = HeatKernelDiffusion.diffuse_heat(...)
curvature = RicciCurvatureEngine.curvature_time_series(returns)

# Generate signals and run backtest
signal_engine = GeometricSignalEngine()
signal_table = signal_engine.generate_signal_table(...)

backtest_engine = BacktestEngine()
report = backtest_engine.backtest_report(prices, signal_table.set_index("date"))

# Visualize results
dashboard = QGADashboard()
fig, ax = dashboard.plot_equity_curve(report["strategy_results"]["equity_curve"], benchmark_curve=report["strategy_results"]["equity_curve"])
```

> Note: The example above is illustrative; actual function inputs should match the data shapes produced by this repository's modules.

## Example Output

In a typical research workflow, QGA produces the following outputs:

- **geometric signal table** — composite scores for each asset with signal labels and confidence ratings
- **heat diffusion scores** — asset-level heat intensity indicating stress concentration
- **curvature stress scores** — Ricci curvature approximations for liquidity and structural fragility
- **topology regime label** — persistent homology regime summaries such as calm, fragmented, or collapsed
- **path-integral probability summary** — expected return, upside/downside probabilities, and distribution quantiles
- **equity curve** — historical strategy value over time
- **drawdown** — peak-to-trough risk profile
- **rolling Sharpe** — risk-adjusted return trajectory

## Research Methodology

The QGA pipeline is designed as a research workflow:

1. Market data → feature engineering
2. Geometric transformation of asset relationships
3. Topology and curvature analysis
4. Probabilistic path simulation
5. Composite signal construction
6. Backtest evaluation

This pipeline emphasizes structural market evidence rather than deterministic forecasting.

## Limitations

- Historical backtests do not guarantee future performance.
- Data sourced from `yfinance` or similar vendors may contain gaps, corporate action effects, or inaccuracies.
- The discrete Ricci curvature implementation is an approximation, not a mathematically exact curvature measurement.
- The p-adic tree is hierarchy-inspired and should not be interpreted as literal p-adic arithmetic.
- The path-integral module is Monte Carlo inspired and not a literal quantum mechanical model.
- There is risk of overfitting to historical data, especially when many model features are combined.
- Transaction costs, slippage, and survivorship bias can materially affect real-world performance.

## Future Work

Potential next steps include:

- intraday and higher-frequency market data
- order book and liquidity modeling
- integration with a dedicated Ollivier-Ricci curvature library
- improved regime labeling and topological feature selection
- portfolio optimization and risk-constrained allocation
- structured stress testing and scenario analysis
- a live research dashboard for monitoring structural market signals
- formal research paper expansion and peer review

## Disclaimer

This repository is intended for education and quantitative research only. It is not financial advice, investment guidance, or a promise of returns.
