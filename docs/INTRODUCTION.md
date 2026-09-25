# Introduction — Predicting Neuronal Tuning from Wiring (MICrONS Benchmark)

![Concept figure: from electron microscopy to a connectome graph, through a graph neural network, to predicted neuronal tuning evaluated against baselines and null graphs](figures/01-concept-schematic.png)

*Figure 1. The pipeline at a glance: the EM-derived wiring diagram becomes a graph of neurons (nodes) and synapses (weighted edges); a graph neural network passes messages along the wiring to predict each neuron's tuning; predictions are scored honestly against wiring-free baselines and degree-preserving null graphs. (Mermaid fallback: `figures/concept_figure.md`.)*

## Background

A central ambition of neuroscience is to explain brain function in terms of brain structure. Connectomics — the complete mapping of neuronal wiring at synaptic resolution — has long promised to make this ambition concrete. The field began with the serial-section electron-microscopy reconstruction of the 302-neuron nervous system of *Caenorhabditis elegans* by White and colleagues, a heroic thirteen-year effort that produced the first complete connectome of any nervous system [1]. For decades, *C. elegans* remained the only complete connectome, in part because electron-microscopy reconstruction of even a cubic millimeter of mammalian brain requires petabytes of imagery and machine-learning-based segmentation to trace individual axons and dendrites through densely packed tissue.

That barrier fell with the MICrONS (Machine Intelligence from Cortical Networks) project, funded by IARPA and executed by the Allen Institute, Baylor College of Medicine, and Princeton University. MICrONS produced a dense reconstruction of approximately one cubic millimeter of mouse visual cortex containing more than 200,000 cells and roughly half a billion (523 million) synaptic connections [2,3]. Crucially, before the tissue was extracted for EM, the same cortical volume was imaged with two-photon calcium imaging in an awake mouse viewing parametric and naturalistic visual stimuli, yielding in vivo functional recordings — orientation and direction tuning, receptive-field properties — for roughly 75,000 of the same neurons [2]. This co-registration of structure and function at single-cell resolution makes MICrONS the first large-scale *functional connectomics* dataset in a mammal.

The MICrONS data enable a question that could not previously be asked at scale in cortex: **how much of a neuron's function is determined by its wiring?** Early analyses showed that anatomy is informative about function — Ding et al. found a general "like-to-like" wiring rule in which connected excitatory neurons share similar visual responses, and demonstrated that functional properties can be predicted, to a degree, from anatomical connectivity [4]. Meanwhile, in the fly, connectome-constrained recurrent network models have shown that wiring diagrams can strongly constrain predicted neural activity across the visual system [5,6]. The mammalian counterpart of this program — learned, predictive models of tuning from connectivity — remains largely unexplored.

Methodologically, a connectome is a directed, weighted graph: neurons are nodes, synapses are edges. Graph neural networks (GNNs) learn node representations by iteratively aggregating information over neighborhoods, and are therefore a natural class of models for predicting node-level properties (tuning) from graph structure (wiring). The field was catalyzed by Kipf and Welling's graph convolutional networks [7] and extended to large, inductive settings by sampling-based aggregation in GraphSAGE [9] and to attention-weighted neighborhoods by graph attention networks [10]. GNNs have transformed machine learning on graphs but have only recently been applied to connectomic data. In parallel, deep learning has already proven its value for modeling visual cortical function from the *input* side: task-optimized deep networks are among the best quantitative models of neural responses in visual cortex [11], and can even be used to control neuronal firing through synthesized images [8]. What is missing is the complementary, structure-driven approach: predicting the same functional properties from measured connectivity rather than from stimulus-response fitting.

## Prior work and gap

Prior work falls into four camps. (1) Descriptive functional connectomics: the MICrONS flagship papers characterized the dataset and uncovered correlational wiring rules (e.g., like-to-like connectivity), but were not framed as a predictive machine-learning benchmark with held-out test sets and baselines [2,3,4]. (2) Connectome-constrained simulation: in *Drosophila*, Lappalainen et al. showed that deep networks constrained by the FlyWire connectome predict neural activity across the fly visual system [5,6] — but fly models are hand-built simulations, not learned predictive models trained on matched functional data, and no mammalian equivalent exists. (3) Input-driven deep models of visual cortex [8,11], which ignore anatomy entirely. (4) Classical connectomics [1,12], which mapped wiring without paired functional recordings.

The gap: there is **no public, reproducible benchmark** that asks how well neuronal tuning (orientation/direction selectivity, receptive fields) can be predicted from EM-derived connectivity alone in mammalian cortex, with fixed splits, strong baselines, and topology-null controls. Without such a benchmark, claims that "wiring determines function" cannot be quantified or compared across models.

## Research questions

1. How well does local synaptic connectivity — alone, without any stimulus information — predict a neuron's orientation tuning, direction selectivity, and receptive-field properties?
2. How much predictive power do learned GNN representations add over hand-crafted connectivity statistics (in/out degree, synapse counts, cell-type composition of partners) and over degree-preserving topology nulls?
3. Which functional properties are predictable from wiring at all, and which appear to require non-connectivity information (e.g., development, plasticity, neuromodulation)?

## Data

| Resource | Contents | Scale | Access |
|---|---|---|---|
| MICrONS mm³ EM connectome | Directed synaptic graph, cell types, morphology | >200k cells, ~0.5B synapses | Open (microns-explorer.org) |
| MICrONS functional imaging | In vivo 2-photon calcium responses to drifting gratings, natural movies | ~75k neurons co-registered with EM | Open (MICrONS Explorer / DANDI) |
| FlyWire whole-brain connectome | Adult *Drosophila* brain wiring (reference point) | ~139k neurons, 54.5M synapses | Open (codex.flywire.ai) |
| *C. elegans* connectome | Complete nematode wiring (historical anchor) | 302 neurons, ~7k connections | Open (literature) |

## Methods

We represent the MICrONS connectome as a directed, weighted graph (edge weight = synapse count; node features = cell type, layer, morphology summaries) and train GNNs (GCN-style message passing [7], sampling-based aggregation [9], and attention variants [10], implemented in PyTorch Geometric) to regress orientation selectivity index, direction selectivity index, and receptive-field parameters for each functionally characterized neuron. Baselines include: (i) hand-crafted connectivity statistics with gradient-boosted trees; (ii) cell-type-only prediction (no wiring); (iii) degree-preserving randomized graphs that destroy fine connectivity while retaining global statistics. We define fixed, versioned train/validation/test splits (by cortical area and depth strata) so so that downstream work compares identical numbers. Evaluation uses held-out R², rank correlation of tuning curves, and calibration analysis, with uncertainty from ensembling.

## Expected contributions

1. The first open, reproducible **function-from-wiring benchmark** on the MICrONS dataset: data loaders, fixed splits, evaluation harness, and baseline results.
2. A quantitative answer to how much of visual tuning is recoverable from connectivity alone, per area and per cell class.
3. Trained reference models and feature extractors that later studies can interpret, transfer, and scale.

## Scope and boundary

This repository contains planning, software, and synthetic-data tests for the benchmark. Prediction is not mechanism: strong predictive accuracy does not establish that wiring *causes* tuning, only that it *contains information about* it; biological interpretation, cross-area/cross-species transfer, and scaling are deferred to later work. We do not attempt connectome-constrained dynamical simulation of the full circuit.

## References

1. White, J. G., Southgate, E., Thomson, J. N. & Brenner, S. The structure of the nervous system of the nematode *Caenorhabditis elegans*. *Philosophical Transactions of the Royal Society of London. Series B, Biological Sciences* 314, 1–340 (1986). DOI: 10.1098/rstb.1986.0056
2. MICrONS Consortium et al. Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 640, 435–447 (2025). DOI: 10.1038/s41586-025-08790-w
3. Turner, N. L., Macrina, T., Bae, J. A. et al. Reconstruction of neocortex: Organelles, compartments, cells, circuits, and activity. *Cell* 185, 1082–1100 (2022). DOI: 10.1016/j.cell.2022.01.023
4. Ding, Z., Fahey, P. G., Papadopoulos, S. et al. Functional connectomics reveals general wiring rule in mouse visual cortex. *Nature* 640, 459–469 (2025). DOI: 10.1038/s41586-025-08840-3
5. Lappalainen, J. K. et al. Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634, 1132–1140 (2024). DOI: 10.1038/s41586-024-07939-3
6. Dorkenwald, S., Matsliah, A., Sterling, A. R. et al. Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138 (2024). DOI: 10.1038/s41586-024-07558-y
7. Kipf, T. N. & Welling, M. Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)* (2017). arXiv:1609.02907
8. Bashivan, P., Kar, K. & DiCarlo, J. J. Neural population control via deep image synthesis. *Science* 364, eaav9436 (2019). DOI: 10.1126/science.aav9436
9. Hamilton, W. L., Ying, R. & Leskovec, J. Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems 30 (NeurIPS)* (2017). arXiv:1706.02216
10. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P. & Bengio, Y. Graph attention networks. *International Conference on Learning Representations (ICLR)* (2018). arXiv:1710.10903
11. Yamins, D. L. K., Hong, H., Cadieu, C. F., Solomon, E. A., Seibert, D. & DiCarlo, J. J. Performance-optimized hierarchical models predict neural responses in higher visual cortex. *Proceedings of the National Academy of Sciences* 111, 8619–8624 (2014). DOI: 10.1073/pnas.1403112111
12. Cook, S. J., Jarrell, T. A., Brittin, C. A. et al. Whole-animal connectomes of both *Caenorhabditis elegans* sexes. *Nature* 571, 63–71 (2019). DOI: 10.1038/s41586-019-1352-7
