#!/usr/bin/env python
"""Build the Phase-2 12x5 fixed-node utility groups from an RPI CSV."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (float(row["mean_rpi_cost"]), int(row["node_id"])))

    n = len(rows)
    strata = [
        ("low", 0),
        ("q20", round(0.20 * (n - 1)) - 5),
        ("q40", round(0.40 * (n - 1)) - 5),
        ("q60", round(0.60 * (n - 1)) - 5),
        ("q80", round(0.80 * (n - 1)) - 5),
        ("high", n - 10),
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    group_id = 1
    for stratum, start in strata:
        ten = rows[max(0, min(start, n - 10)):][:10]
        for half in range(2):
            group = ten[half * 5:(half + 1) * 5]
            group_path = output_dir / "group_{:02d}_{}.txt".format(group_id, stratum)
            group_path.write_text("\n".join(row["node_id"] for row in group) + "\n")
            manifest.append({
                "group_id": group_id,
                "stratum": stratum,
                "file": str(group_path),
                "node_ids": [int(row["node_id"]) for row in group],
                "mean_degree": sum(float(row["degree"]) for row in group) / 5,
                "mean_gap": sum(float(row["mean_margin_gap"]) for row in group) / 5,
                "mean_sensitivity": sum(float(row["mean_grad_norm"]) for row in group) / 5,
                "mean_rpi_cost": sum(float(row["mean_rpi_cost"]) for row in group) / 5,
            })
            group_id += 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)
    for item in manifest:
        print(
            "group_{:02d} {:>4} nodes={} mean_rpi={:.8g}".format(
                item["group_id"], item["stratum"], item["node_ids"],
                item["mean_rpi_cost"],
            )
        )


if __name__ == "__main__":
    main()
