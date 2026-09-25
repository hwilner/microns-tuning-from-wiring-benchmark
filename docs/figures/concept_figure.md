# Concept figure (Mermaid fallback) — Paper 1: Predicting Tuning from Wiring

> **Note.** The rendered concept figure is [`01-concept-schematic.png`](01-concept-schematic.png) in this folder — a generated scientific illustration. This file keeps the Mermaid source of the same diagram so it remains editable and re-renderable.

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
