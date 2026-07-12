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
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gridfm.data import build_strict_datasets
from gridfm.legacy import physics
from gridfm.losses import objective
from gridfm.model import EdgeStateGridFM


def loader(dataset, batch: int, workers: int, shuffle: bool, samples: int | None = None):
    sampler = None
    if shuffle and samples is not None and samples < len(dataset):
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=samples)
        shuffle = False
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, sampler=sampler,
        num_workers=workers, pin_memory=True,
        multiprocessing_context="fork" if workers else None,
    )


def init_wandb(cfg: dict, out: Path):
    wc = cfg.get("wandb", {})
    if not wc.get("enabled", False):
        return None
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
    amp = device.type == "cuda" and cfg["train"].get("amp") == "bf16"
    wape_sums: dict[str, float] = {}
    for batch in batches:
        batch = batch.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                raw, aux = model(batch, return_aux=True)
            # Physical inverse transforms stay outside autocast. Model outputs may
            # be float32 while the persisted truth and scales remain float64.
            raw = {k: v.float() for k, v in raw.items()}
            aux["edge_dv"] = {k: v.float() for k, v in aux["edge_dv"].items()}
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
            for key, value in physics.percentage_error_sums(
                batch, preds, float(cfg["loss"]["feat_clamp"])
            ).items():
                wape_sums[key] = wape_sums.get(key, 0.0) + value
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
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
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
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    model = EdgeStateGridFM(**cfg["model"]).to(device)
    if cfg["train"].get("init_ckpt"):
        init = torch.load(Path(cfg["train"]["init_ckpt"]), map_location=device, weights_only=False)
        model.load_state_dict(init["model"])
        print(f"initialized model from {cfg['train']['init_ckpt']}", flush=True)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    batch_size = int(cfg["train"]["batch_size"])
    workers = int(cfg["data"].get("num_workers", 0))
    samples = int(cfg["data"].get("samples_per_epoch", len(bundle.train)))
    train_loader = loader(bundle.train, batch_size, workers, True, samples)
    seen_loader = loader(bundle.seen, batch_size, workers, False)
    unseen_loader = loader(bundle.unseen, batch_size, workers, False)
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

    start, best_v, best_i = 1, float("inf"), float("inf")
    last_path = out / "last.pt"
    if args.resume and last_path.is_file():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start = int(ck["epoch"]) + 1
        best_v, best_i = float(ck["best_v"]), float(ck["best_i"])

    wb = init_wandb(cfg, out)
    log_path = out / "log.jsonl"
    for epoch in range(start, int(cfg["train"]["epochs"]) + 1):
        bundle.train.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, cfg, device, s_kcl, opt, sched)
        seen_metrics = unseen_metrics = {}
        if epoch % int(cfg["train"]["eval_every"]) == 0 or epoch == int(cfg["train"]["epochs"]):
            with torch.no_grad():
                seen_metrics = run_epoch(model, seen_loader, cfg, device, s_kcl)
                unseen_metrics = run_epoch(model, unseen_loader, cfg, device, s_kcl)
        rec = {
            "epoch": epoch, "lr": sched.get_last_lr()[0],
            "train": train_metrics, "seen": seen_metrics, "unseen": unseen_metrics,
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"epoch {epoch:03d} train={train_metrics.get('loss', float('nan')):.4e} "
            f"seen V/I={seen_metrics.get('V_wape_pct', float('nan')):.4f}%/"
            f"{seen_metrics.get('Ibus_wape_pct', float('nan')):.3f}% "
            f"unseen V/I={unseen_metrics.get('V_wape_pct', float('nan')):.4f}%/"
            f"{unseen_metrics.get('Ibus_wape_pct', float('nan')):.3f}%",
            flush=True,
        )
        if wb is not None:
            flat = {"epoch": epoch, "lr": rec["lr"]}
            for split in ("train", "seen", "unseen"):
                for key, value in rec[split].items():
                    flat[f"{split}/{key}"] = value
            wb.log(flat, step=epoch)
        score_split = unseen_metrics or seen_metrics
        v = score_split.get("V_wape_pct", float("inf"))
        i = score_split.get("Ibus_wape_pct", float("inf"))
        state = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "epoch": epoch,
            "best_v": min(best_v, v), "best_i": min(best_i, i), "cfg": cfg,
        }
        torch.save(state, last_path)
        if v < best_v:
            best_v = v
            torch.save(state, out / "best_voltage.pt")
        if i < best_i:
            best_i = i
            torch.save(state, out / "best_current.pt")
    if wb is not None:
        wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
