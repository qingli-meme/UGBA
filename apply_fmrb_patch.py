#!/usr/bin/env python3
"""
Patch UGBA/run_adaptive.py to add FMRB while leaving the original Backdoor class untouched.

Run from the UGBA repository root:
    python apply_fmrb_patch.py
"""

from pathlib import Path
import shutil
import sys


path = Path("run_adaptive.py")
if not path.exists():
    raise SystemExit("run_adaptive.py not found. Run this script from the UGBA repository root.")

text = path.read_text(encoding="utf-8")

backup = path.with_suffix(".py.fmrb_backup")
if not backup.exists():
    shutil.copy2(path, backup)
    print(f"[backup] {backup}")

# 1) Import
old = "from models.backdoor import Backdoor\n"
new = (
    "from models.backdoor import Backdoor\n"
    "from models.distributed_backdoor import DistributedRelayBackdoor\n"
)
if "from models.distributed_backdoor import DistributedRelayBackdoor" not in text:
    if old not in text:
        raise SystemExit("Cannot find Backdoor import anchor.")
    text = text.replace(old, new, 1)

# 2) CLI arguments, inserted immediately before --test_model
arg_anchor = "parser.add_argument('--test_model', type=str, default='GCN',"
if "--attack_method" not in text:
    if arg_anchor not in text:
        raise SystemExit("Cannot find --test_model parser anchor.")
    args_block = """parser.add_argument('--attack_method', type=str, default='ugba',
                    choices=['ugba','fmrb'],
                    help='Backdoor implementation: original UGBA or focused multi-relay backdoor')
parser.add_argument('--relay_method', type=str, default='pfr',
                    choices=['pfr','gain','low_degree','random'],
                    help='Relay selection rule used by FMRB')
parser.add_argument('--relay_count', type=int, default=-1,
                    help='Number of relays/payloads per victim; <=0 reuses --trigger_size')
parser.add_argument('--relay_eps', type=float, default=1e-12,
                    help='Numerical epsilon in relay scoring')
"""
    text = text.replace(arg_anchor, args_block + arg_anchor, 1)

# 3) Model instantiation
old_model = "model = Backdoor(args,device)\n"
new_model = """if args.attack_method == 'fmrb':
    model = DistributedRelayBackdoor(args,device)
else:
    model = Backdoor(args,device)
"""
if "model = DistributedRelayBackdoor(args,device)" not in text:
    if old_model not in text:
        raise SystemExit("Cannot find original Backdoor instantiation.")
    text = text.replace(old_model, new_model, 1)

# 4) 1-by-1 trigger injection
old_inject_1 = (
    "induct_x, induct_edge_index,induct_edge_weights = "
    "model.inject_trigger(relabeled_node_idx,poison_x[sub_induct_nodeset],"
    "sub_induct_edge_index,sub_induct_edge_weights,device)"
)
new_inject_1 = """if args.attack_method == 'fmrb':
                        eligible_relay_mask = (sub_induct_nodeset < data.x.shape[0]).to(device)
                        induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(
                            relabeled_node_idx,
                            poison_x[sub_induct_nodeset],
                            sub_induct_edge_index,
                            sub_induct_edge_weights,
                            device,
                            eligible_relay_mask=eligible_relay_mask,
                        )
                    else:
                        induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(
                            relabeled_node_idx,
                            poison_x[sub_induct_nodeset],
                            sub_induct_edge_index,
                            sub_induct_edge_weights,
                            device,
                        )"""
if "eligible_relay_mask = (sub_induct_nodeset < data.x.shape[0])" not in text:
    if old_inject_1 not in text:
        raise SystemExit("Cannot find 1-by-1 inject_trigger anchor.")
    text = text.replace(old_inject_1, new_inject_1, 1)

# 5) Overall trigger injection
old_inject_all = (
    "induct_x, induct_edge_index,induct_edge_weights = "
    "model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)"
)
new_inject_all = """if args.attack_method == 'fmrb':
                eligible_relay_mask = (
                    torch.arange(poison_x.shape[0], device=device) < data.x.shape[0]
                )
                induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(
                    idx_atk,
                    poison_x,
                    induct_edge_index,
                    induct_edge_weights,
                    device,
                    eligible_relay_mask=eligible_relay_mask,
                )
            else:
                induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(
                    idx_atk,
                    poison_x,
                    induct_edge_index,
                    induct_edge_weights,
                    device,
                )"""
if "torch.arange(poison_x.shape[0], device=device) < data.x.shape[0]" not in text:
    if old_inject_all not in text:
        raise SystemExit("Cannot find overall inject_trigger anchor.")
    text = text.replace(old_inject_all, new_inject_all, 1)

path.write_text(text, encoding="utf-8")
print("[ok] patched run_adaptive.py")
print("[next] python -m py_compile run_adaptive.py models/distributed_backdoor.py relay_selection.py")
