"""Supervised, edge-state, and validated physical objectives."""
from __future__ import annotations

import torch

from .legacy import SPECS, i_offset, physics, y_width
from .tree_current import decode_tree_line_currents


def balanced_reconstruction_loss(
    batch, preds, weights: dict, field_std: dict | None = None,
    balance_stores: float = 0.0,
):
    """Normalize each physical field independently before weighting.

    A single entry-count average lets hundreds of component columns drown out
    the few bus-voltage targets. Field balancing makes the task definition
    explicit and remains stable as topology size/component mix changes.
    """
    dev = preds["node"].device
    sums = {k: torch.zeros((), device=dev) for k in ("voltage", "y", "icomp", "ibus")}
    counts = {k: torch.zeros((), device=dev) for k in sums}
    store_parts = {k: [] for k in ("y", "icomp", "ibus")}
    nd = batch["node"]
    mv = nd.msk_v.unsqueeze(1)
    sums["voltage"] += ((preds["node"] - nd.dv.to(preds["node"].dtype)).pow(2) * mv).sum()
    counts["voltage"] += 2 * mv.sum()
    for store in SPECS:
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ny, ni = y_width(store), i_offset(store)
        err = preds[store] - st.x_true.to(preds[store].dtype)
        if field_std is not None:
            err = err / field_std[store].to(err.dtype).clamp_min(1e-12)
        err2 = err.pow(2)
        for name, cols in (
            ("y", slice(0, ny)), ("icomp", slice(ny, ni)), ("ibus", slice(ni, None))
        ):
            mask = st.msk[:, cols]
            value_sum = (err2[:, cols] * mask).sum()
            value_count = mask.sum()
            sums[name] += value_sum
            counts[name] += value_count
            if balance_stores > 0 and bool(value_count > 0):
                store_parts[name].append(value_sum / value_count)
    parts = {k: sums[k] / counts[k].clamp_min(1) for k in sums}
    if balance_stores > 0:
        alpha = float(balance_stores)
        if not 0 <= alpha <= 1:
            raise ValueError("loss.balance_stores must be in [0, 1]")
        for name, values in store_parts.items():
            if values:
                store_mean = torch.stack(values).mean()
                parts[name] = (1 - alpha) * parts[name] + alpha * store_mean
    total = sum(float(weights.get(k, 0.0)) * value for k, value in parts.items())
    return total, parts


def edge_voltage_loss(batch, aux: dict, drop_floor: float = 1e-4):
    """Supervise terminal dV proposals and complex line terminal drops."""
    nd = batch["node"]
    total = nd.dv.new_zeros(())
    count = nd.dv.new_zeros(())
    drop_total = nd.dv.new_zeros(())
    drop_count = nd.dv.new_zeros(())
    for store in SPECS:
        es = batch[(store, "conn", "node")]
        comp, node, slot = es.edge_index[0], es.edge_index[1], es.slot
        pred = aux["edge_dv"][store]
        if not node.numel():
            continue
        # Every incidence is a physically meaningful local voltage state. Give
        # masked nodes full weight and visible nodes a small anchoring weight.
        weight = torch.where(nd.msk_v[node], 1.0, 0.1).to(pred.dtype).unsqueeze(1)
        total = total + (weight * (pred - nd.dv[node].to(pred.dtype)).pow(2)).sum()
        count = count + 2 * weight.sum()
        if store != "line":
            continue
        # Match conductor slot s at terminal 1 with slot s at terminal 2.
        n_comp = batch[store].num_nodes
        slot_pred = pred.new_zeros(n_comp, 8, 2)
        slot_node = torch.full((n_comp, 8), -1, dtype=torch.long, device=node.device)
        slot_pred[comp, slot] = pred
        slot_node[comp, slot] = node
        for s in range(4):
            valid = (slot_node[:, s] >= 0) & (slot_node[:, 4 + s] >= 0)
            if not valid.any():
                continue
            n1, n2 = slot_node[valid, s], slot_node[valid, 4 + s]
            pdrop = slot_pred[valid, s] - slot_pred[valid, 4 + s]
            tdrop = nd.dv[n1].to(pred.dtype) - nd.dv[n2].to(pred.dtype)
            scale = tdrop.norm(dim=1, keepdim=True).clamp_min(drop_floor)
            drop_total = drop_total + torch.nn.functional.smooth_l1_loss(
                pdrop / scale, tdrop / scale, reduction="sum"
            )
            drop_count = drop_count + 2 * valid.sum()
    return total / count.clamp_min(1), drop_total / drop_count.clamp_min(1)


def physical_ibus_wape_loss(
    batch, preds, clamp: float, stores=None, min_truth_abs: float = 0.0,
    graph_mask=None,
):
    """Differentiable aggregate Ibus WAPE in pu, never feature coordinates."""
    num = preds["node"].new_zeros(())
    den = preds["node"].new_zeros(())
    for store in (stores or SPECS):
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ni = i_offset(store)
        mask = st.msk[:, ni:]
        if graph_mask is not None:
            comp_batch = getattr(st, "batch", None)
            if comp_batch is None:
                comp_batch = torch.zeros(
                    st.num_nodes, dtype=torch.long, device=mask.device
                )
            mask = mask & graph_mask[comp_batch].unsqueeze(1)
        if not mask.any():
            continue
        pred = physics.decode(preds[store][:, ni:], st.scale[:, ni:], clamp)
        truth = physics.decode_truth(st.x_true[:, ni:], st.scale[:, ni:])
        if min_truth_abs > 0:
            mask = mask & (truth.abs() >= min_truth_abs)
        num = num + (pred - truth).abs()[mask].sum()
        den = den + truth.abs()[mask].sum()
    return num / den.clamp_min(1e-12)


def pf_graph_mask(batch) -> torch.Tensor:
    """Identify exactly posed PF masks inside a mixed-task batch."""
    nd = batch["node"]
    node_batch = getattr(nd, "batch", None)
    if node_batch is None:
        node_batch = torch.zeros(nd.num_nodes, dtype=torch.long, device=nd.msk_v.device)
    n_graph = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
    keep = torch.ones(n_graph, dtype=torch.bool, device=node_batch.device)
    required_v = ~nd.ground & ~nd.slack
    bad_node = required_v & ~nd.msk_v
    if bad_node.any():
        keep[node_batch[bad_node].unique()] = False
    for store in SPECS:
        st = batch[store]
        if st.num_nodes == 0:
            continue
        ny, ni = y_width(store), i_offset(store)
        bad = st.msk[:, :ni].any(1) | (st.act[:, ni:] & ~st.msk[:, ni:]).any(1)
        if bad.any():
            comp_batch = getattr(st, "batch", None)
            if comp_batch is None:
                comp_batch = torch.zeros(
                    st.num_nodes, dtype=torch.long, device=bad.device
                )
            keep[comp_batch[bad].unique()] = False
    return keep


def objective(batch, raw_preds, aux, cfg: dict, s_kcl: float):
    clamp = float(cfg["loss"]["feat_clamp"])
    preds = physics.clamp_structural_zeros(batch, raw_preds)
    mask_loss, metrics = physics.mask_loss_and_metrics(batch, preds, clamp, raw_preds=raw_preds)
    recon, recon_parts = balanced_reconstruction_loss(
        batch, preds, cfg["loss"].get(
            "recon_weights", {"voltage": 1.0, "y": 1.0, "icomp": 1.0, "ibus": 1.0}
        ), aux.get("field_std"), float(cfg["loss"].get("balance_stores", 0.0))
    )
    x_bar, vr, vi = physics.completed(batch, preds)
    elem, kcl, pmetrics = physics.physics_losses(batch, x_bar, vr, vi, clamp, s_kcl)
    edge, drop = edge_voltage_loss(batch, aux, float(cfg["loss"].get("drop_floor", 1e-4)))
    lc = cfg["loss"]
    ibus_wape = physical_ibus_wape_loss(batch, preds, clamp)
    line_wape = physical_ibus_wape_loss(batch, preds, clamp, ("line",))
    reactor_wape = preds["node"].new_zeros(())
    if float(cfg["loss"].get("lambda_reactor_wape", 0.0)):
        reactor_wape = physical_ibus_wape_loss(batch, preds, clamp, ("reactor",))
    transformer_wape = preds["node"].new_zeros(())
    if float(cfg["loss"].get("lambda_transformer_wape", 0.0)):
        transformer_wape = physical_ibus_wape_loss(
            batch, preds, clamp, ("transformer",)
        )
    tree_wape = preds["node"].new_zeros(())
    tree_line_wape = preds["node"].new_zeros(())
    if float(lc.get("lambda_tree_wape", 0.0)) or float(
        lc.get("lambda_tree_line_wape", 0.0)
    ):
        tree_preds = decode_tree_line_currents(batch, preds, clamp)
        only_pf = pf_graph_mask(batch) if lc.get("tree_pf_only", False) else None
        tree_wape = physical_ibus_wape_loss(
            batch, tree_preds, clamp, graph_mask=only_pf
        )
        tree_line_wape = physical_ibus_wape_loss(
            batch, tree_preds, clamp, ("line",),
            float(lc.get("tree_min_truth_abs", 0.0)),
            graph_mask=only_pf,
        )
    loss = (
        float(lc.get("lambda_recon", lc.get("lambda_mask", 1.0))) * recon
        + float(lc.get("lambda_edge", 0.0)) * edge
        + float(lc.get("lambda_drop", 0.0)) * drop
        + float(lc.get("lambda_elem", 0.0)) * elem
        + float(lc.get("lambda_kcl", 0.0)) * kcl
        + float(lc.get("lambda_ibus_wape", 0.0)) * ibus_wape
        + float(lc.get("lambda_line_wape", 0.0)) * line_wape
        + float(lc.get("lambda_reactor_wape", 0.0)) * reactor_wape
        + float(lc.get("lambda_transformer_wape", 0.0)) * transformer_wape
        + float(lc.get("lambda_tree_wape", 0.0)) * tree_wape
        + float(lc.get("lambda_tree_line_wape", 0.0)) * tree_line_wape
    )
    logs = {
        "loss": loss.item(), "loss_mask": mask_loss.item(), "loss_recon": recon.item(),
        **{f"loss_{k}": value.item() for k, value in recon_parts.items()},
        "loss_edge": edge.item(), "loss_drop": drop.item(),
        "loss_elem": elem.item(), "loss_kcl": kcl.item(),
        "loss_ibus_wape": ibus_wape.item(),
        "loss_line_wape": line_wape.item(),
        "loss_reactor_wape": reactor_wape.item(),
        "loss_transformer_wape": transformer_wape.item(),
        "loss_tree_wape": tree_wape.item(),
        "loss_tree_line_wape": tree_line_wape.item(),
        **metrics, **pmetrics,
    }
    return loss, preds, logs
