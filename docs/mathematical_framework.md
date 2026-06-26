# QGA Mathematical Framework

## 1. Abstract

QGA is a research platform that represents financial markets as evolving geometric and topological structures. The framework combines hierarchical ultrametric topology, graph diffusion, discrete curvature, persistent homology, and Monte Carlo path ensembles to produce composite market signals. The objective is to explore whether structural market signatures capture stress, regime transitions, and asset dislocations that are not visible in traditional time-series models.

## 2. Motivation

Traditional financial time-series models typically treat each asset as an isolated scalar process or they rely on low-dimensional linear correlations. Such models often miss higher-order structure that emerges from market hierarchy, network fragility, liquidity concentration, and regime topology.

A purely statistical time-series view can fail to distinguish:

- hierarchical relationships among assets in sectors and industries,
- network bottlenecks that create fragility when liquidity is concentrated,
- topological signatures of regime change that arise from loops and connected components,
- latent structure beyond contemporaneous correlation.

QGA is motivated by the hypothesis that market behavior can be better understood when data are embedded in a geometric and topological space, and when signals are derived from the structure of that space rather than from scalar price series alone.

## 3. Market Data as a Geometric Object

Assets are represented as nodes in a geometric object whose shape evolves over time. Several representations are used in parallel:

1. **Nodes and features**: Each asset is associated with a feature vector such as returns, volatility, or correlation-based measures.
2. **Point clouds**: Feature vectors form points in a high-dimensional space; distances between points reveal similarity structure.
3. **Graphs**: Assets become vertices with weighted edges representing similarity, correlation, or co-movement.

In this view, the market is not simply a set of independent price series. It is a structured object whose connectivity, diffusion, curvature, and topological persistence contain information about the underlying regime.

## 4. p-adic-Inspired Ultrametric Market Trees

### 4.1 Market Hierarchy

Assets are arranged in a hierarchical tree that reflects market taxonomy: broad market, asset classes, sectors, sub-sectors, and individual securities. The tree is built from a domain knowledge hierarchy or from clustering of structural features.

### 4.2 Ultrametric Distance

The ultrametric distance between two assets is determined by the depth of their lowest common ancestor (LCA) in the hierarchy. Formally, for assets $x$ and $y$:

$$
\mathrm{d}_p(x, y) = p^{-\mathrm{depth}(\mathrm{LCA}(x, y))},
$$

where $p > 1$ is a base parameter and $\mathrm{depth}(\mathrm{LCA}(x,y))$ measures the hierarchical level of the shared ancestor.

### 4.3 Lowest Common Ancestor

The lowest common ancestor is the deepest tree node that is an ancestor of both assets. If two assets share a deep ancestor, they are structurally similar; if their shared ancestor is near the root, they are more distant.

### 4.4 Hierarchical Distance Usefulness

Hierarchical distance is useful because it reflects market structure that is not apparent from pairwise correlation alone. It provides a coarse-grained similarity measure that captures sectoral and industry relationships, and it identifies assets that are isolated from stressed clusters.

### 4.5 p-adic-Inspired, Not Literal p-adic Arithmetic

The model is described as p-adic-inspired because it employs ultrametric distances and hierarchical scaling similar to p-adic metrics. It does not perform formal p-adic number arithmetic. The emphasis is on hierarchical similarity rather than algebraic p-adic operations.

## 5. Heat Kernel Diffusion on Asset Graphs

### 5.1 Graph Laplacian

A graph is constructed from asset correlations or similarity measures. The weighted graph Laplacian $L$ is defined as:

$$
L = D - W,
$$

where $W$ is the adjacency matrix of edge weights and $D$ is the diagonal degree matrix.

### 5.2 Heat Diffusion Equation

Heat diffusion on the graph is modeled by the solution of the heat equation.
For an initial heat vector $u(0)$, the diffused state at time $t$ is:

$$
u(t) = \exp(-tL) \, u(0).
$$

This expression defines the heat kernel on the graph.

### 5.3 Volatility or Stress as Heat

In the financial interpretation, volatility, return stress, or liquidity stress is treated as an initial heat distribution on assets. Diffusion spreads this stress through the graph according to the graph Laplacian.

### 5.4 Diffusion Across Correlated Assets

Heat diffusion captures how stress propagates across correlated assets. Highly connected subgraphs can rapidly share heat, producing concentrated stress profiles. Conversely, diffuse structures indicate dispersion of stress.

## 6. Discrete Ricci Curvature and Liquidity Fragility

### 6.1 Curvature as a Proxy for Graph Stability

Ricci curvature measures the extent to which geodesics converge or diverge. On a discrete asset graph, curvature is approximated by comparing local transport distances with direct edge distances.

### 6.2 Negative Curvature as Bottlenecks or Fragility

Negative curvature is interpreted as an indicator of fragility or bottlenecks in the graph. When local transport between neighborhoods is larger than the direct edge distance, the region behaves like a vulnerable conduit where perturbations may amplify.

### 6.3 Simplified Ollivier-Ricci Intuition

Ollivier-Ricci curvature uses probability measures on neighboring nodes and measures how those distributions contract under the graph metric. In simplified form, QGA uses a proxy that penalizes large neighbor transport distances relative to edge distance.

### 6.4 Limitations

This curvature is a practical approximation. It is not a full continuous Ricci curvature computation, and it should be interpreted as a structural liquidity proxy rather than a rigorous geometric invariant.

## 7. Persistent Homology for Regime Detection

### 7.1 Point Clouds

Market features can be embedded as points in a high-dimensional space. A point cloud is formed from asset returns, correlations, or graph-based distances.

### 7.2 H0 Connected Components

H0 homology tracks connected components as a scale parameter changes. In financial data, the birth and death of components reflect clustering and regime separation.

### 7.3 H1 Loops

H1 homology captures loops or cycles in the data. Loop structures may correspond to rotating market sectors or cyclic transitions between states.

### 7.4 Persistence Diagrams

Persistence diagrams record the birth and death times of topological features across scales. They provide a summary of structural stability and complexity.

### 7.5 Wasserstein Distance and Regime Similarity

Distances between persistence diagrams, such as the Wasserstein distance, quantify regime similarity. Small distances indicate topologically similar market states, while large distances suggest regime shifts.

## 8. Path-Integral-Inspired Return Simulation

### 8.1 Monte Carlo Paths

The path-integral module generates many simulated future price paths using geometric Brownian motion. Each simulated path is evaluated and weighted.

### 8.2 Economic Action Functional

An economic action functional assigns a scalar cost to each path. The action penalizes undesirable path properties such as high realized volatility, large drawdown, and negative terminal return, and it incorporates exogenous stress measures.

### 8.3 Lower Action Paths Receive Higher Probability

Paths are weighted by a softmax over negative action. The probability of a path is:

$$
P(\text{path}) \propto \exp(-S[\text{path}]),
$$

where $S[\text{path}]$ is the action functional.

### 8.4 Imaginary-Time Path Integral Inspiration

The weighting scheme is inspired by the structure of imaginary-time path integrals in physics, where paths are weighted by $\exp(-S)$ rather than $\exp(iS)$. In QGA, this provides a metaphorical foundation for treating lower-action paths as more probable. It is explicitly not a claim of literal quantum prediction.

## 9. Unified Geometric Signal

The unified signal engine aggregates multiple structural components into a single score. The core components are:

- curvature score,
- heat diffusion score,
- topology regime score,
- ultrametric isolation score,
- path-integral probability score.

Each component is normalized and combined with weights to produce a score in the range $[-100, 100]$. The score is translated into signal labels such as Strong Long, Long, Neutral, Risk-Off, and Strong Short. Component agreement also produces a confidence metric.

## 10. Backtesting Methodology

### 10.1 Rebalance Frequency

Backtesting is performed with discrete rebalancing intervals: daily, weekly, or monthly. The chosen frequency determines how often portfolio weights are updated based on the latest available signals.

### 10.2 Long-Only and Long-Short Modes

The strategy supports long-only allocation to top-ranked assets and long-short allocation with dollar neutrality. In long-short mode, the highest-ranked assets are longed and the lowest-ranked assets are shorted with equal absolute exposure.

### 10.3 Transaction Costs

Transaction costs are applied to capture trading friction. Turnover is computed between rebalancing periods, and costs are deducted from returns to approximate realistic strategy drag.

### 10.4 Performance Metrics

Standard performance metrics are computed from the strategy equity curve:

- total return,
- annualized return,
- annualized volatility,
- Sharpe ratio,
- maximum drawdown,
- hit rate,
- best day and worst day.

Benchmark comparison further contextualizes performance against a passive index or an equal-weight portfolio.

## 11. Limitations and Risks

QGA is a research framework. The following limitations are important:

- historical backtests do not guarantee future performance,
- data quality issues, missing values, corporate actions, and survivorship bias can distort results,
- the discrete Ricci curvature proxy is an approximation and not a full geometric invariant,
- the ultrametric tree is a hierarchy-inspired model, not literal p-adic arithmetic,
- the path-integral module is Monte Carlo inspired, not a literal quantum mechanical model,
- combining many engineered features increases the risk of overfitting,
- transaction costs, slippage, and liquidity impact may be underestimated.

## 12. Conclusion

QGA is not a magic arbitrage machine. It is a technical framework for testing whether geometric and topological market signatures contain useful information.

The project is intended as a research platform for exploring market structure through rigorous mathematical lenses, with an emphasis on honesty, precision, and practical limitations.
