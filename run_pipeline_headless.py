"""Headless wrapper for run_pipeline.py.

Sets the matplotlib backend to Agg (non-interactive) before any import so
plot_segmentation_results / plot_pipeline_result do not block on plt.show()
when no display is available — e.g. when launching from WSL into the Windows
Python interpreter (which does not inherit Unix env vars set on the command
line). Behaviour identical to running run_pipeline.py directly otherwise.
"""

import os
import sys
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Force UTF-8 stdout so the κ glyph in the concave-point summary does not
# crash the per-image loop under the Windows cp1252 default console codec.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from run_pipeline import main


if __name__ == "__main__":
    main(debugging=True, image_path="all", classify=True)
