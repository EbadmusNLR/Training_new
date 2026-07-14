#!/usr/bin/env python3
"""Full-matrix physics for the datakit scenario format (basis=pu).

The datakit corpus stores admittances as FULL row-major matrices per component
(Load/Gen/PV/Storage 4x4, Line/Cap/Reactor/Vsource 8x8, Transformer 12x12) and
per-terminal currents I_*_bus{t}_pu of width FC=4. Connectivity is one
(component,'bus{t}','node') edge per ACTIVE conductor slot; active slots are the
leading ones (slot 0..count-1), so edge position within a component's terminal
equals its conductor slot.

Physics (datakit DATA_STRUCTURE.md):
    passive:  I_into = Y @ V
    active:   I_into = Y @ V - Icomp        (Icomp on terminal-1 slots)
    KCL:      sum of into-element currents at each non-ground node = 0

Everything is a batched dense matmul — GPU-friendly, no packed-triangle expand.
"""
from __future__ import annotations

import torch

FC = 4  # fixed conductors per terminal

# store -> (json Y-field prefix, n_terminals, icomp_slots)
STORES: dict[str, tuple[str, int, int]] = {
    "line": ("Yline", 2, 0),
    "capacitor": ("Ycap", 2, 0),
    "reactor": ("Yreactor", 2, 0),
    "transformer": ("Yxfmr", 3, 0),
    "vsource": ("Ysource", 2, 8),
    "load": ("Yload", 1, 4),
    "generator": ("Ygen", 1, 4),
    "pvsystem": ("Ypv", 1, 4),
    "storage": ("Ystorage", 1, 4),
}


def store_size(data, store: str) -> int:
    """Component count for a store (PyG cannot infer num_nodes here)."""
    st = data[store]
    prefix = STORES[store][0]
    for f in (f"{prefix}_r_pu", "I_r_bus1_pu"):
        if f in st:
            return st[f].shape[0]
    return 0


def node_count(data) -> int:
    return data["node"].V_r_init_pu.shape[0]


def terminal_slot(comp_ids: torch.Tensor) -> torch.Tensor:
    """Conductor slot (0..count-1) of each edge, from its position within the
    component's contiguous edge group. Works on batched graphs (each feeder's
    edges are grouped by component in order, offsets keep them consecutive)."""
    if comp_ids.numel() == 0:
        return comp_ids.new_zeros(0)
    _, counts = torch.unique_consecutive(comp_ids, return_counts=True)
    return torch.cat([torch.arange(int(c), device=comp_ids.device) for c in counts])


def _y_full(store_data, prefix: str, dim: int, dtype):
    n = store_data[f"{prefix}_r_pu"].shape[0]
    yr = store_data[f"{prefix}_r_pu"].to(dtype).reshape(n, dim, dim)
    yi = store_data[f"{prefix}_i_pu"].to(dtype).reshape(n, dim, dim)
    return yr, yi


def local_voltages(data, store: str, nterm: int, vr: torch.Tensor, vi: torch.Tensor):
    """Assemble per-component terminal voltages [n, nterm*FC] from node V via the
    terminal edges (inactive slots stay 0; their Y rows/cols are 0)."""
    n = store_size(data, store)
    dim = nterm * FC
    Vlr = vr.new_zeros((n, dim))
    Vli = vi.new_zeros((n, dim))
    for t in range(1, nterm + 1):
        rel = (store, f"bus{t}", "node")
        if rel not in data.edge_types:
            continue
        ei = data[rel].edge_index
        comp, node = ei[0], ei[1]
        if comp.numel() == 0:
            continue
        col = (t - 1) * FC + terminal_slot(comp)
        Vlr[comp, col] = vr[node]
        Vli[comp, col] = vi[node]
    return Vlr, Vli


def element_currents(data, store: str, vr: torch.Tensor, vi: torch.Tensor,
                     yr_full=None, yi_full=None, icomp_r=None, icomp_i=None):
    """Into-element terminal currents I=Y@V (−Icomp for active), [n, nterm*FC]
    complex as (Ir, Ii). Y/Icomp default to the stored truth; pass overrides to
    decode from predicted admittance/compensation."""
    prefix, nterm, nic = STORES[store]
    dim = nterm * FC
    dtype = vr.dtype
    st = data[store]
    if yr_full is None:
        yr_full, yi_full = _y_full(st, prefix, dim, dtype)
    Vlr, Vli = local_voltages(data, store, nterm, vr, vi)
    Ir = torch.bmm(yr_full, Vlr.unsqueeze(-1)).squeeze(-1) - torch.bmm(yi_full, Vli.unsqueeze(-1)).squeeze(-1)
    Ii = torch.bmm(yr_full, Vli.unsqueeze(-1)).squeeze(-1) + torch.bmm(yi_full, Vlr.unsqueeze(-1)).squeeze(-1)
    if nic:
        icr = st.Icomp_r_pu.to(dtype) if icomp_r is None else icomp_r
        ici = st.Icomp_i_pu.to(dtype) if icomp_i is None else icomp_i
        # Icomp occupies the first `nic` flattened slots ([bus1|bus2|...] order):
        # 4 for one-terminal devices, 8 for the two-terminal vsource.
        w = min(nic, Ir.shape[1])
        Ir[:, :w] = Ir[:, :w] - icr[:, :w]
        Ii[:, :w] = Ii[:, :w] - ici[:, :w]
    return Ir, Ii


def stored_currents(data, store: str, dtype=torch.float32):
    """Stored truth into-element currents [n, nterm*FC] as (Ir, Ii)."""
    prefix, nterm, _ = STORES[store]
    st = data[store]
    n = st[f"{prefix}_r_pu"].shape[0] if f"{prefix}_r_pu" in st else st.I_r_bus1_pu.shape[0]
    Ir = torch.zeros(n, nterm * FC, dtype=dtype)
    Ii = torch.zeros(n, nterm * FC, dtype=dtype)
    for t in range(1, nterm + 1):
        if f"I_r_bus{t}_pu" in st:
            Ir[:, (t - 1) * FC:t * FC] = st[f"I_r_bus{t}_pu"].to(dtype)
            Ii[:, (t - 1) * FC:t * FC] = st[f"I_i_bus{t}_pu"].to(dtype)
    return Ir, Ii


def nodal_kcl_residual(data, per_store_I: dict) -> torch.Tensor:
    """Nodal KCL residual [N,2] = sum of into-element terminal currents at each
    node (ground node 0 excluded). per_store_I[store]=(Ir,Ii) of [n, nterm*FC]."""
    n_node = node_count(data)
    dev = data["node"].V_r_init_pu.device
    rr = torch.zeros(n_node, device=dev)
    ri = torch.zeros(n_node, device=dev)
    for store, (Ir, Ii) in per_store_I.items():
        _, nterm, _ = STORES[store]
        for t in range(1, nterm + 1):
            rel = (store, f"bus{t}", "node")
            if rel not in data.edge_types:
                continue
            ei = data[rel].edge_index
            comp, node = ei[0], ei[1]
            if comp.numel() == 0:
                continue
            col = (t - 1) * FC + terminal_slot(comp)
            rr.index_add_(0, node, Ir[comp, col].to(rr.dtype))
            ri.index_add_(0, node, Ii[comp, col].to(ri.dtype))
    res = torch.stack([rr, ri], dim=1)
    # node 0 is ground (Gnd/gnd.0); exclude it from KCL
    res[0] = 0.0
    return res


if __name__ == "__main__":
    import sys, glob, os
    sys.path.insert(0, "/kfs2/projects/gogpt/Ebadmus/datakit")
    from core.scenario_store import FeederScenarios

    fs = sorted(glob.glob("/kfs2/projects/gogpt/Ebadmus/training_data/SMART-DS_1000/*/static.pt"),
                key=os.path.getsize)
    probe = fs[:3] + fs[len(fs) // 2: len(fs) // 2 + 2]  # small + medium feeders
    agg = {}
    kcl_max = 0.0
    for p in probe:
        data = FeederScenarios(os.path.dirname(p))[0]
        vr, vi = data["node"].V_r_pu.double(), data["node"].V_i_pu.double()
        per_store = {}
        for store in STORES:
            if store not in data.node_types or store_size(data, store) == 0:
                continue
            Ir, Ii = element_currents(data, store, vr, vi)
            Tr, Ti = stored_currents(data, store, dtype=torch.float64)
            num = float((Ir - Tr).abs().sum() + (Ii - Ti).abs().sum())
            den = float(Tr.abs().sum() + Ti.abs().sum() + 1e-12)
            a = agg.setdefault(store, [0.0, 0.0, 0])
            a[0] += num; a[1] += den; a[2] += store_size(data, store)
            per_store[store] = (Tr, Ti)
        res = nodal_kcl_residual(data, per_store)
        kcl_max = max(kcl_max, float(res.abs().max()))
    print(f"probed {len(probe)} feeders")
    for store, (num, den, n) in agg.items():
        print(f"  {store:12s} n={n:6d}  I=Y@V vs stored WAPE={num/den:.2e}")
    print(f"  global KCL |residual| max over feeders={kcl_max:.2e}")
