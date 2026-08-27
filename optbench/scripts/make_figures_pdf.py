"""Assemble the thesis result figures (RQ1 / RQ2 / RQ3) into one flip-through PDF.

One figure per page with a short caption. Skips any figure that isn't present yet
(e.g. the second-model robustness figures while that run is still in progress).
Run: .venv\\Scripts\\python.exe scripts\\make_figures_pdf.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

FIG = "results/figures"
OUT = "reports/figures_overview.pdf"

# (path, short title, one-line caption) in reading order.
FIGURES = [
    (f"{FIG}/rq1_gpt2_lora_rq1_val_curves.png",
     "RQ1 - Validation curves (GPT-2 + LoRA)",
     "Best config per optimizer, mean +-std over 5 seeds. AdamW lowest/fastest; the others converge close together."),
    (f"{FIG}/rq1_gpt2_lora_rq2_tunability.png",
     "RQ2 - Tunability: quality vs learning rate",
     "Left: full range (Sophia diverges at high LR). Right: zoom near optima (AdamW broad valley, Lion narrow, SGD separate high-LR basin)."),
    (f"{FIG}/rq3_filternorm_slices.png",
     "RQ3 - Filter-normalized loss slices (Li et al., 2018)",
     "Fair, comparable 2D slices of each optimizer's solution. AdamW flattest, Lion sharpest."),
    (f"{FIG}/rq3_naive_vs_filternorm.png",
     "RQ3 - Naive vs filter-normalized directions",
     "Same AdamW solution, same total step size: naive (global) normalization makes the minimum look sharper. Naive visualization misleads."),
    (f"{FIG}/rq3_interpolation.png",
     "RQ3 - 1D interpolation between solutions",
     "AdamW -> SGD solutions: a loss barrier (~0.027) means they sit in different basins (not linearly connected)."),
    # Second-model robustness (appears once that run finishes):
    (f"{FIG}/rq1b_smollm2_lora_rq2_tunability.png",
     "RQ1 robustness - Tunability on SmolLM2-360M",
     "Does the tunability picture transfer to a second, Llama-style model?"),
    (f"{FIG}/rq1b_smollm2_lora_rq1_val_curves.png",
     "RQ1 robustness - Validation curves on SmolLM2-360M",
     "Does the quality ranking transfer to a second model?"),
]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    included = 0
    with PdfPages(OUT) as pdf:
        for path, title, caption in FIGURES:
            if not os.path.exists(path):
                print(f"skip (not found): {path}")
                continue
            img = mpimg.imread(path)
            h, w = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(11, 8.5))  # landscape Letter
            ax = fig.add_axes([0.03, 0.06, 0.94, 0.82])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
            fig.text(0.5, 0.03, caption, ha="center", va="bottom", fontsize=9, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
            included += 1
            print(f"added: {path}")
    print(f"\n{included} figures -> {OUT}")


if __name__ == "__main__":
    main()
