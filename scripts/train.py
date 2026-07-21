#!/usr/bin/env python3
"""Train EdgeStateGridFM with topology-held-out checkpoint selection."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets, fit_feature_stats
from gridfm.config import load_config
from gridfm.legacy import physics
from gridfm.losses import objective
from gridfm.model import EdgeStateGridFM, load_compatible_state
from gridfm.tree_current import decode_tree_line_currents


def foundation_selection_score(task_metrics: dict) -> float:
    """Score aggregate tasks and scale-normalized component tails, fail closed."""
    required = (
        ("pf", "V_wape_pct"), ("pf", "Ibus_wape_pct"),
        ("se_known", "V_wape_pct"), ("se_known", "Ibus_wape_pct"),
        ("param_one", "Y_wape_pct"), ("injection", "Icomp_wape_pct"),
    )
    values = [
        task_metrics.get(task, {}).get(key, float("inf"))
        for task, key in required
    ]
    # Raw family WAPE is undefined when its physical truth denominator is zero
    # (notably storage Y). Family-scale WAPE remains meaningful there.
    for task, role in (("param_one", "_Y"), ("injection", "_Icomp")):
        values.extend(
            float(value)
            for key, value in task_metrics.get(task, {}).items()
            if key.startswith("field_")
            and role in key
            and key.endswith("_scale_wape_pct")
        )
    for task in ("random_safe", "random"):
        if task in task_metrics:
            values.extend(
                task_metrics[task].get(f"{field}_wape_pct", float("inf"))
                for field in ("V", "Y", "Icomp", "Ibus")
            )
    return max(values) if values else float("inf")


def evaluate_task_lenses(dataset, task_fields: dict, evaluate) -> dict:
    """Evaluate each mask through a synchronous loader backed by ``dataset``.

    A persistent-worker DataLoader owns a forked copy of the dataset, so parent
    ``mask_cfg`` changes are invisible to it. The caller must therefore pass an
    evaluator over a zero-worker loader.
    """
    original_mask = dataset.mask_cfg
    metrics = {}
    try:
        for task in task_fields:
            task_mask = {**original_mask, "mixture": {task: 1.0}}
            if task == "random":
                # Canonical all-field stress mask. The random-safe training
                # config deliberately has zero base Y/Icomp probabilities.
                task_mask.update({
                    "p_voltage": 0.30,
                    "p_current": 0.15,
                    "p_icomp": 0.15,
                    "p_admittance": 0.10,
                    "p_terminal": 0.05,
                    "p_component": 0.0,
                })
            dataset.mask_cfg = task_mask
            metrics[task] = evaluate()
    finally:
        dataset.mask_cfg = original_mask
    return metrics


def loader(
    dataset, batch: int, workers: int, shuffle: bool,
    samples: int | None = None, prefetch_factor: int = 2,
):
    sampler = None
    if shuffle and samples is not None and samples < len(dataset):
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=samples)
        shuffle = False
    worker_args = {}
    if workers:
        worker_args.update(
            multiprocessing_context="fork",
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, sampler=sampler,
        num_workers=workers, pin_memory=True,
        **worker_args,
    )


def init_wandb(cfg: dict, out: Path):
    wc = cfg.get("wandb", {})
    if not wc.get("enabled", False):
        return None
    # wandb key: prefer this repo's .env; fall back to the legacy location only
    # if it happens to exist (DG_FM_Training is no longer a dependency).
    env_path = ROOT / ".env"
    if not env_path.is_file():
        env_path = ROOT.parent / "DG_FM_Training" / ".env"
    if "WANDB_API_KEY" not in os.environ and env_path.is_file():
        for line in env_path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().lower() in {"wandb_api_key", "wandb_key"}:
                os.environ["WANDB_API_KEY"] = value.strip().strip("\"'")
                break
    try:
        import wandb
    except ImportError:
        print("warning: wandb unavailable; continuing with JSONL logs", flush=True)
        return None
    id_path = out / "wandb_run_id.txt"
    run_id = id_path.read_text().strip() if id_path.is_file() else wandb.util.generate_id()
    id_path.write_text(run_id + "\n")
    return wandb.init(
        project=wc.get("project", "Training_new_GridFM"), name=wc.get("name", out.name),
        id=run_id, resume="allow", dir=str(out), tags=wc.get("tags"), config=cfg,
    )


def run_epoch(model, batches, cfg, device, s_kcl, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    logs: dict[str, list[float]] = {}
    amp = training and device.type == "cuda" and cfg["train"].get("amp") == "bf16"
    wape_sums: dict[str, float] = {}
    for batch in batches:
        batch = batch.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                raw, aux = model(batch, return_aux=True)
            # Physical inverse transforms stay outside autocast. Model outputs may
            # be float32 while the persisted truth and scales remain float64.
            raw = {k: v.float() if v.dtype in (torch.float16, torch.bfloat16) else v
                   for k, v in raw.items()}
            aux["edge_dv"] = {
                k: v.float() if v.dtype in (torch.float16, torch.bfloat16) else v
                for k, v in aux["edge_dv"].items()
            }
            loss, preds, row = objective(batch, raw, aux, cfg, s_kcl)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            optimizer.step()
            scheduler.step()
        for key, value in row.items():
            logs.setdefault(key, []).append(float(value))
        if not training:
            clamp = float(cfg["loss"]["feat_clamp"])
            variants = {"": preds}
            if float(cfg["loss"].get("lambda_tree_wape", 0.0)) or float(
                cfg["loss"].get("lambda_tree_line_wape", 0.0)
            ):
                tree = decode_tree_line_currents(batch, preds, clamp)
                variants["tree_"] = physics.kcl_decode_vsource(batch, tree, clamp)
            for prefix, variant in variants.items():
                for key, value in physics.percentage_error_sums(
                    batch, variant, clamp
                ).items():
                    name = prefix + key
                    wape_sums[name] = wape_sums.get(name, 0.0) + value
    out = {key: statistics.fmean(values) for key, values in logs.items()}
    if not training:
        for key in {k[:-4] for k in wape_sums if k.endswith("_num")}:
            den = wape_sums.get(f"{key}_den", 0.0)
            if den > 0:
                out[f"{key}_wape_pct"] = 100.0 * wape_sums[f"{key}_num"] / den
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit-feeders", type=int)
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument(
        "--task-mixture", type=json.loads,
        help='JSON task probabilities, e.g. {"pf":0.35,"se_known":0.25,"injection":0.4}',
    )
    ap.add_argument("--recon-voltage-weight", type=float)
    ap.add_argument("--recon-icomp-weight", type=float)
    ap.add_argument("--recon-ibus-weight", type=float)
    ap.add_argument("--init-ckpt", type=Path)
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument(
        "--exact-metadata", choices=("none", "line", "transformer", "generator", "shunts", "load", "pvsystem", "vsource", "storage", "both", "all"),
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.task_mixture is not None:
        if not isinstance(args.task_mixture, dict) or not args.task_mixture:
            ap.error("--task-mixture must be a non-empty JSON object")
        total = sum(float(value) for value in args.task_mixture.values())
        if any(float(value) < 0 for value in args.task_mixture.values()) or abs(total - 1.0) > 1e-8:
            ap.error("--task-mixture probabilities must be nonnegative and sum to 1")
        cfg["mask"]["mixture"] = {
            str(key): float(value) for key, value in args.task_mixture.items()
        }
    recon_weights = cfg["loss"].setdefault("recon_weights", {})
    for name, value in (
        ("voltage", args.recon_voltage_weight),
        ("icomp", args.recon_icomp_weight),
        ("ibus", args.recon_ibus_weight),
    ):
        if value is not None:
            if value < 0:
                ap.error(f"--recon-{name}-weight must be nonnegative")
            recon_weights[name] = value
    if args.init_ckpt is not None and args.scratch:
        ap.error("--init-ckpt and --scratch are mutually exclusive")
    if args.init_ckpt is not None:
        cfg["train"]["init_ckpt"] = str(args.init_ckpt)
    elif args.scratch:
        cfg["train"].pop("init_ckpt", None)
    if args.exact_metadata is not None:
        cfg["model"]["exact_line_metadata"] = args.exact_metadata in ("line", "both", "all")
        cfg["model"]["exact_transformer_metadata"] = args.exact_metadata in (
            "transformer", "both", "all"
        )
        cfg["model"]["exact_generator_metadata"] = args.exact_metadata in ("generator", "all")
        cfg["model"]["exact_capacitor_metadata"] = args.exact_metadata in ("shunts", "all")
        cfg["model"]["exact_reactor_metadata"] = args.exact_metadata in ("shunts", "all")
        cfg["model"]["exact_load_metadata"] = args.exact_metadata in ("load", "all")
        cfg["model"]["exact_pvsystem_metadata"] = args.exact_metadata in ("pvsystem", "all")
        cfg["model"]["exact_vsource_metadata"] = args.exact_metadata in ("vsource", "all")
        cfg["model"]["exact_storage_metadata"] = args.exact_metadata in ("storage", "all")
    cfg["data"]["exact_line_metadata"] = bool(
        cfg["model"].get("exact_line_metadata", False)
    )
    cfg["data"]["exact_transformer_metadata"] = bool(
        cfg["model"].get("exact_transformer_metadata", False)
    )
    cfg["data"]["exact_generator_metadata"] = bool(
        cfg["model"].get("exact_generator_metadata", False)
    )
    cfg["data"]["exact_capacitor_metadata"] = bool(
        cfg["model"].get("exact_capacitor_metadata", False)
    )
    cfg["data"]["exact_reactor_metadata"] = bool(
        cfg["model"].get("exact_reactor_metadata", False)
    )
    cfg["data"]["exact_load_metadata"] = bool(
        cfg["model"].get("exact_load_metadata", False)
    )
    cfg["data"]["exact_pvsystem_metadata"] = bool(
        cfg["model"].get("exact_pvsystem_metadata", False)
    )
    cfg["data"]["exact_vsource_metadata"] = bool(
        cfg["model"].get("exact_vsource_metadata", False)
    )
    cfg["data"]["exact_storage_metadata"] = bool(
        cfg["model"].get("exact_storage_metadata", False)
    )
    if args.limit_feeders is not None:
        cfg["data"]["limit_feeders"] = args.limit_feeders
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    seed = int(cfg["train"]["seed"])
    torch.manual_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out or cfg["train"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    bundle = build_strict_datasets(cfg["data"], cfg["mask"], seed)
    print(
        f"strict datasets: train={len(bundle.train)} seen={len(bundle.seen)} "
        f"unseen={len(bundle.unseen)} test={len(bundle.test)}; "
        f"feeders={len(bundle.train_feeders)}/{len(bundle.unseen_feeders)}/"
        f"{len(bundle.test_feeders)}; build={time.time()-t0:.1f}s",
        flush=True,
    )
    split_manifest = {
        "train": bundle.train_feeders, "seen": bundle.train_feeders,
        "unseen": bundle.unseen_feeders, "test": bundle.test_feeders,
    }
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    import yaml
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    model_cfg = dict(cfg["model"])
    model_dtype = model_cfg.pop("dtype", "float32")
    model = EdgeStateGridFM(**model_cfg).to(device)
    if model_dtype == "float64":
        model = model.double()
    elif model_dtype != "float32":
        raise ValueError(f"unsupported model.dtype={model_dtype}")
    if cfg["train"].get("init_ckpt"):
        init = torch.load(Path(cfg["train"]["init_ckpt"]), map_location=device, weights_only=False)
        load_compatible_state(model, init["model"])
        print(f"initialized model from {cfg['train']['init_ckpt']}", flush=True)
    if bool(model_cfg.get("normalize_features")):
        # A warm checkpoint may have been trained under a different global
        # scaler/corpus. Refit after loading so stale normalization buffers do
        # not overwrite the train-only statistics of the current corpus.
        stats = fit_feature_stats(
            bundle.train, float(cfg["data"].get("feature_min_std", 1e-8))
        )
        model.set_feature_stats(stats)
        print("fitted current train-only per-column feature normalization", flush=True)
    if cfg["train"].get("freeze_except_field_heads", False):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.field_head.parameters():
            parameter.requires_grad_(True)
        print("froze backbone and voltage heads; training field heads only", flush=True)
    if cfg["train"].get("freeze_except_feedback", False):
        # Train only the KCL-residual feedback path and the voltage-producing
        # heads.  The converged backbone is a delicate optimum that diverges when
        # perturbed by the stiff physics signal (R2/KCL1); freezing it lets the
        # feedback learn to correct voltage without destabilizing the model.
        trainable = (
            model.kcl_feedback_mlp, model.node_head, model.node_edge_gate,
            model.edge_dv_head, model.node_norm,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in trainable:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        print("froze backbone; training KCL feedback + voltage heads only", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model parameters: {total_params:,}; trainable: {trainable_params:,}", flush=True
    )
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    batch_size = int(cfg["train"]["batch_size"])
    workers = int(cfg["data"].get("num_workers", 0))
    samples = int(cfg["data"].get("samples_per_epoch", len(bundle.train)))
    prefetch_factor = int(cfg["data"].get("prefetch_factor", 2))
    if prefetch_factor < 1:
        raise ValueError("data.prefetch_factor must be >= 1")
    train_loader = loader(
        bundle.train, batch_size, workers, True, samples, prefetch_factor
    )
    # Evaluation is short and infrequent. Separate persistent pools here
    # multiply worker/pin-memory threads across co-located GPU jobs and caused
    # reproducible epoch-5 failures; keep both evaluation loaders synchronous.
    seen_loader = loader(
        bundle.seen, batch_size, 0, False, prefetch_factor=prefetch_factor
    )
    unseen_loader = loader(
        bundle.unseen, batch_size, 0, False, prefetch_factor=prefetch_factor
    )
    # Task masks change between passes. Keep this loader synchronous so each
    # pass observes the current parent-dataset mask rather than a stale fork.
    task_loader = loader(bundle.unseen, batch_size, 0, False)
    steps = math.ceil(min(samples, len(bundle.train)) / batch_size)
    total_steps = max(1, int(cfg["train"]["epochs"]) * steps)
    warmup = min(int(cfg["train"].get("warmup_steps", 0)), total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: step / max(1, warmup) if step < warmup else 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))
        ),
    )
    scaler = json.loads((Path(cfg["data"]["root"]) / "feature_scaler.json").read_text())
    s_kcl = statistics.median(v["I_scale"] for v in scaler["current"].values())
    if getattr(model, "kcl_feedback_enabled", False):
        model.s_kcl = model.s_kcl.new_tensor(float(s_kcl))

    start, best_v, best_i, best_foundation = (
        1, float("inf"), float("inf"), float("inf")
    )
    last_path = out / "last.pt"
    if args.resume and last_path.is_file():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start = int(ck["epoch"]) + 1
        best_v, best_i = float(ck["best_v"]), float(ck["best_i"])
        best_foundation = float(ck.get("best_foundation", float("inf")))

    wb = init_wandb(cfg, out)
    log_path = out / "log.jsonl"
    for epoch in range(start, int(cfg["train"]["epochs"]) + 1):
        epoch_started = time.perf_counter()
        bundle.train.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, cfg, device, s_kcl, opt, sched)
        train_sec = time.perf_counter() - epoch_started
        seen_metrics = unseen_metrics = {}
        task_metrics = {}
        eval_started = time.perf_counter()
        if epoch % int(cfg["train"]["eval_every"]) == 0 or epoch == int(cfg["train"]["epochs"]):
            with torch.no_grad():
                seen_metrics = run_epoch(model, seen_loader, cfg, device, s_kcl)
                unseen_metrics = run_epoch(model, unseen_loader, cfg, device, s_kcl)
                task_fields = cfg["train"].get("foundation_task_fields", {})
                if task_fields:
                    task_metrics = evaluate_task_lenses(
                        bundle.unseen,
                        task_fields,
                        lambda: run_epoch(model, task_loader, cfg, device, s_kcl),
                    )
        eval_sec = time.perf_counter() - eval_started
        epoch_sec = time.perf_counter() - epoch_started
        rec = {
            "epoch": epoch, "lr": sched.get_last_lr()[0],
            "train_sec": train_sec, "eval_sec": eval_sec, "epoch_sec": epoch_sec,
            "train": train_metrics, "seen": seen_metrics, "unseen": unseen_metrics,
            "unseen_tasks": task_metrics,
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"epoch {epoch:03d} [{epoch_sec:.1f}s train={train_sec:.1f}s "
            f"eval={eval_sec:.1f}s] train={train_metrics.get('loss', float('nan')):.4e} "
            f"seen V/I={seen_metrics.get('V_wape_pct', float('nan')):.4f}%/"
            f"{seen_metrics.get('tree_Ibus_wape_pct', seen_metrics.get('Ibus_wape_pct', float('nan'))):.3f}% "
            f"unseen V/I={unseen_metrics.get('V_wape_pct', float('nan')):.4f}%/"
            f"{unseen_metrics.get('tree_Ibus_wape_pct', unseen_metrics.get('Ibus_wape_pct', float('nan'))):.3f}%",
            flush=True,
        )
        if wb is not None:
            flat = {"epoch": epoch, "lr": rec["lr"]}
            for split in ("train", "seen", "unseen"):
                for key, value in rec[split].items():
                    flat[f"{split}/{key}"] = value
            for task, values in task_metrics.items():
                for key, value in values.items():
                    flat[f"unseen_tasks/{task}/{key}"] = value
            wb.log(flat, step=epoch)
        score_split = unseen_metrics or seen_metrics
        v = score_split.get("V_wape_pct", float("inf"))
        i = score_split.get(
            "tree_Ibus_wape_pct", score_split.get("Ibus_wape_pct", float("inf"))
        )
        task_fields = cfg["train"].get("foundation_task_fields", {})
        selection_weights = cfg["train"].get("foundation_selection_weights", {})
        weighted = [
            (float(weight), score_split.get(f"{field}_wape_pct", float("inf")))
            for field, weight in selection_weights.items()
            if float(weight) > 0
        ]
        foundation = foundation_selection_score(task_metrics) if task_fields else (
            sum(weight * value for weight, value in weighted)
            / sum(weight for weight, _ in weighted)
            if weighted else float("inf")
        )
        state = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "epoch": epoch,
            "best_v": min(best_v, v), "best_i": min(best_i, i),
            "best_foundation": min(best_foundation, foundation), "cfg": cfg,
        }
        torch.save(state, last_path)
        if v < best_v:
            best_v = v
            torch.save(state, out / "best_voltage.pt")
        if i < best_i:
            best_i = i
            torch.save(state, out / "best_current.pt")
        if foundation < best_foundation:
            best_foundation = foundation
            torch.save(state, out / "best_foundation.pt")
    if wb is not None:
        wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
