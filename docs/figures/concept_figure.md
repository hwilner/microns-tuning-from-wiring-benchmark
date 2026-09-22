# Concept figure (Mermaid fallback) — Paper 1: Predicting Tuning from Wiring

> The canonical vector figure is `concept_figure.svg` in this directory. This file is a
> faithful Mermaid rendition kept as a text-based fallback (the PNG raster could not be
> committed through the available API tooling).

```mermaid
flowchart LR
    A[EM connectome<br/>MICrONS: >200k cells<br/>~523M synapses] --> B[Connectome graph:<br/>neurons = nodes<br/>synapses = weighted edges]
    B --> C[Graph neural network<br/>message passing layers]
    C --> D[Predicted tuning:<br/>orientation · direction<br/>receptive field]
    E[Wiring-free baseline<br/>position, area, layer] --> F{Held-out R²<br/>benchmark leaderboard}
    G[Degree-preserving<br/>null graph] --> F
    D --> F
    F --> H[Honest negative result at<br/>~1.3 edges/neuron:<br/>baseline 0.06 vs GNN 0.04]
```
