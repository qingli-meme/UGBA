#!/usr/bin/env python
"""Apply the minimal RPI integration to UGBA/run_adaptive.py.

Usage (from UGBA repository root):
    python /path/to/apply_rpi_patch.py

The script is intentionally conservative: it only edits run_adaptive.py and
stops if the expected official-main code markers are not found.
"""

from pathlib import Path
import sys


RUN = Path("run_adaptive.py")


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            "Patch marker {!r} expected exactly once, found {}. "
            "Your UGBA revision may differ; apply patches/run_adaptive.patch manually.".format(label, count)
        )
    return text.replace(old, new, 1)


def main():
    if not RUN.exists():
        raise SystemExit("run_adaptive.py not found. Run this script from the UGBA repository root.")

    text = RUN.read_text()
    if "selection_method == 'rpi'" in text and "import rpi_selection as rpis" in text:
        print("RPI patch already appears to be applied; no changes made.")
        return

    old_selection_args = """parser.add_argument('--selection_method', type=str, default='none',\n                    choices=['loss','conf','cluster','none','cluster_degree'],\n                    help='Method to select idx_attach for training trojan model (none means randomly select)')\n"""
    new_selection_args = """parser.add_argument('--selection_method', type=str, default='none',\n                    choices=['loss','conf','cluster','none','cluster_degree','rpi'],\n                    help='Method to select idx_attach for training trojan model (none means randomly select)')\n# RPI selection setting -- only used when --selection_method rpi\nparser.add_argument('--rpi_surrogate_model', type=str, default='GCN',\n                    choices=['GCN','GAT','GraphSage'],\n                    help='Clean surrogate used only for offline RPI scoring')\nparser.add_argument('--rpi_mc_samples', type=int, default=4,\n                    help='Number of topology samples for RPI; first sample is the clean graph')\nparser.add_argument('--rpi_edge_drop', type=float, default=0.10,\n                    help='Undirected clean-edge drop probability for robust RPI samples')\nparser.add_argument('--rpi_kappa', type=float, default=0.0,\n                    help='Required target decision margin used in the minimum propagation cost')\nparser.add_argument('--rpi_eps', type=float, default=1e-12,\n                    help='Numerical stabilizer in RPI denominator')\nparser.add_argument('--rpi_max_candidates', type=int, default=0,\n                    help='Optional cap for large graphs; 0 scores all eligible candidates')\nparser.add_argument('--rpi_no_class_balance', action='store_true', default=False,\n                    help='Disable class-balanced final Bottom-K selection')\nparser.add_argument('--rpi_score_dir', type=str, default='./rpi_scores',\n                    help='Directory used to save per-node RPI diagnostics')\nparser.add_argument('--rpi_verbose_every', type=int, default=100,\n                    help='Print RPI scoring progress every N candidates; <=0 disables progress prints')\n"""
    text = replace_once(text, old_selection_args, new_selection_args, "selection arguments")

    text = replace_once(
        text,
        "import heuristic_selection as hs\n",
        "import heuristic_selection as hs\nimport rpi_selection as rpis\n",
        "RPI import",
    )

    old_tail = """    idx_attach = torch.LongTensor(idx_attach).to(device)\nprint(\"idx_attach: {}\".format(idx_attach))\n"""
    new_tail = """    idx_attach = torch.LongTensor(idx_attach).to(device)\nelif(args.selection_method == 'rpi'):\n    idx_attach = rpis.robust_propagation_selection(\n        args=args,\n        data=data,\n        idx_train=idx_train,\n        idx_val=idx_val,\n        idx_clean_test=idx_clean_test,\n        unlabeled_idx=unlabeled_idx,\n        train_edge_index=train_edge_index,\n        size=size,\n        device=device,\n    )\nprint(\"idx_attach: {}\".format(idx_attach))\n"""
    text = replace_once(text, old_tail, new_tail, "selection branch")

    backup = RUN.with_suffix(".py.rpi_backup")
    if not backup.exists():
        backup.write_text(RUN.read_text())
    RUN.write_text(text)
    print("Applied RPI patch to run_adaptive.py")
    print("Backup: {}".format(backup))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
