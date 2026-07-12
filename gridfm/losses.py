"""Supervised, edge-state, and validated physical objectives."""
from __future__ import annotations

import torch

from .legacy import SPECS, physics


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
        weight = torch.where(nd.msk_v[node], 1.0, 0.1).unsqueeze(1)
        total = total + (weight * (pred - nd.dv[node].float()).pow(2)).sum()
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
            tdrop = nd.dv[n1].float() - nd.dv[n2].float()
            scale = tdrop.norm(dim=1, keepdim=True).clamp_min(drop_floor)
            drop_total = drop_total + torch.nn.functional.smooth_l1_loss(
                pdrop / scale, tdrop / scale, reduction="sum"
            )
            drop_count = drop_count + 2 * valid.sum()
    return total / count.clamp_min(1), drop_total / drop_count.clamp_min(1)


def objective(batch, raw_preds, aux, cfg: dict, s_kcl: float):
    clamp = float(cfg["loss"]["feat_clamp"])
    preds = physics.clamp_structural_zeros(batch, raw_preds)
    mask_loss, metrics = physics.mask_loss_and_metrics(batch, preds, clamp, raw_preds=raw_preds)
    x_bar, vr, vi = physics.completed(batch, preds)
    elem, kcl, pmetrics = physics.physics_losses(batch, x_bar, vr, vi, clamp, s_kcl)
    edge, drop = edge_voltage_loss(batch, aux, float(cfg["loss"].get("drop_floor", 1e-4)))
    lc = cfg["loss"]
    loss = (
        float(lc.get("lambda_mask", 1.0)) * mask_loss
        + float(lc.get("lambda_edge", 0.0)) * edge
        + float(lc.get("lambda_drop", 0.0)) * drop
        + float(lc.get("lambda_elem", 0.0)) * elem
        + float(lc.get("lambda_kcl", 0.0)) * kcl
    )
    logs = {
        "loss": loss.item(), "loss_mask": mask_loss.item(),
        "loss_edge": edge.item(), "loss_drop": drop.item(),
        "loss_elem": elem.item(), "loss_kcl": kcl.item(),
        **metrics, **pmetrics,
    }
    return loss, preds, logs

