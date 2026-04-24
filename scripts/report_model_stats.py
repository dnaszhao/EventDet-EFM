import argparse
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_model_module


def build_config(model_name: str, dataset_name: str, experiment_group: str, experiment_name: str):
    config_dir = str(REPO_ROOT / "config")
    overrides = [
        f"model={model_name}",
        f"dataset={dataset_name}",
        f"+experiment/{experiment_group}={experiment_name}",
    ]
    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        cfg = compose(config_name="train", overrides=overrides)
    dynamically_modify_train_config(cfg)
    return cfg


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    non_trainable = total - trainable
    estimated_size_mb = total * 4 / (1024 ** 2)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "estimated_size_mb_fp32": estimated_size_mb,
    }


def benchmark_inference(module, cfg, device: torch.device, batch_size: int, warmup: int, runs: int):
    input_channels = cfg.model.backbone.input_channels
    input_hw = cfg.model.backbone.in_res_hw
    x = torch.randn(batch_size, input_channels, input_hw[0], input_hw[1], device=device)

    module = module.to(device)
    module.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    with torch.no_grad():
        for _ in range(warmup):
            module(x, previous_states=None, retrieve_detections=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        times_ms = []
        for _ in range(runs):
            start = time.perf_counter()
            module(x, previous_states=None, retrieve_detections=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    mean_ms = sum(times_ms) / len(times_ms)
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device=device) / 1024 / 1024

    return {
        "input_shape": [batch_size, input_channels, input_hw[0], input_hw[1]],
        "mean_latency_ms": mean_ms,
        "fps": fps,
        "peak_memory_mb": peak_mem_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Report parameter count and dummy-input inference stats.")
    parser.add_argument("--model", default="rnndet")
    parser.add_argument("--dataset", required=True, help="Example: gen4x0.01_ss")
    parser.add_argument("--experiment-group", required=True, help="Example: gen4")
    parser.add_argument("--experiment", required=True, help="Example: efm_tiny_win3060.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    cfg = build_config(
        model_name=args.model,
        dataset_name=args.dataset,
        experiment_group=args.experiment_group,
        experiment_name=args.experiment,
    )
    module = fetch_model_module(cfg)
    param_stats = count_parameters(module)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    print("=== Config ===")
    print(f"model={args.model}")
    print(f"dataset={args.dataset}")
    print(f"experiment_group={args.experiment_group}")
    print(f"experiment={args.experiment}")
    print(f"device={requested_device}")
    print(f"vit_size={cfg.model.backbone.vit_size}")
    print(f"in_res_hw={list(cfg.model.backbone.in_res_hw)}")
    print(f"num_classes={cfg.model.head.num_classes}")

    print("\n=== Parameters ===")
    print(f"total_params={param_stats['total']}")
    print(f"trainable_params={param_stats['trainable']}")
    print(f"non_trainable_params={param_stats['non_trainable']}")
    print(f"estimated_size_mb_fp32={param_stats['estimated_size_mb_fp32']:.3f}")

    if args.skip_benchmark:
        return

    bench_stats = benchmark_inference(
        module=module,
        cfg=cfg,
        device=requested_device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
    )
    print("\n=== Benchmark ===")
    print(f"input_shape={bench_stats['input_shape']}")
    print(f"mean_latency_ms={bench_stats['mean_latency_ms']:.3f}")
    print(f"fps={bench_stats['fps']:.3f}")
    print(f"peak_memory_mb={bench_stats['peak_memory_mb']:.3f}")


if __name__ == "__main__":
    main()
