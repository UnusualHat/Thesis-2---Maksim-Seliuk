"""Loss-landscape visualization (RQ3).

Gated to toy models by design: full-data slices and the naive-vs-filter-normalized
comparison are cheap and meaningful on small models, where they answer the actual RQ3
question -- *what correct visualization shows about different optimizers' solutions, and
how naive visualization misleads* (Li et al., 2018). On the LLM only the benchmark
(RQ1/RQ2) runs.
"""
