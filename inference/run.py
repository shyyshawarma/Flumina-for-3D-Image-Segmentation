import argparse
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from inference import ROOT


DATASETS = ("AMOS-2022", "BraTS-2025-GLI", "KiTS-2023", "LiTS", "MAMA-MIA")
MODELS = ("Flumina-S", "Flumina-B", "Flumina-M", "Flumina-L")


def boolean(value: str) -> bool:
    """
    Parse a command-line boolean without case sensitivity.

    Args:
        value (str): true, false, yes, no, 1, or 0.

    Returns:
        bool: Boolean represented by the accepted token.
    """
    match value.lower():
        case "true" | "1" | "yes":
            return True
        case "false" | "0" | "no":
            return False
        case _:
            raise argparse.ArgumentTypeError("Expected True or False")


def parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for checkpoint evaluation.

    Returns:
        argparse.ArgumentParser: Parser for dataset, model, inference, and output options.
    """
    result = argparse.ArgumentParser(
        description="Evaluate formal Flumina checkpoints on processed 3D volumes."
    )
    result.add_argument(
        "--dataset", required=True, choices=(*DATASETS, "AMOS-2022-CT", "AMOS-2022-MRI")
    )
    result.add_argument("--model", required=True, choices=MODELS)
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument(
        "--config",
        type=Path,
        help="Saved config.json; discovered beside the checkpoint or in its run directory",
    )
    result.add_argument(
        "--data-root", type=Path, help="Processed dataset directory or shared parent"
    )
    result.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_dir",
        type=Path,
        default=ROOT / "runs",
    )
    result.add_argument(
        "--inference_augument",
        "--inference-augument",
        type=boolean,
        default=True,
        metavar="True|False",
    )
    result.add_argument(
        "--metrics", choices=("benchmark", "legacy"), default="benchmark"
    )
    result.add_argument("--device", default="auto")
    result.add_argument("--crop-size", type=int, nargs=3, metavar=("D", "H", "W"))
    result.add_argument("--overlap", type=float)
    result.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"))
    result.add_argument("--seed", type=int)
    result.add_argument("--membership-file", type=Path)
    result.add_argument("--max-cases", type=int)
    result.add_argument("--cpu-threads", type=int, default=4)
    result.add_argument("--save-predictions", action="store_true")
    result.add_argument(
        "--amos-subsets",
        action="store_true",
        help="Also summarize CT and MRI cases from the same predictions",
    )
    result.add_argument("--dry-run", "--preflight", dest="dry_run", action="store_true")
    return result


def configure_runtime() -> None:
    """
    Create repository-local cache directories and disable Python bytecode writes.
    """
    cache = ROOT / ".cache"
    for variable, name in {
        "TMPDIR": "tmp",
        "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": "xdg",
        "TORCH_EXTENSIONS_DIR": "extensions",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "MPLCONFIGDIR": "matplotlib",
        "HF_HOME": "huggingface",
    }.items():
        location = cache / name
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)
    tempfile.tempdir = os.environ["TMPDIR"]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    Combine saved evaluation settings with explicit command-line overrides.

    Args:
        args (argparse.Namespace): Parsed inference arguments.

    Returns:
        dict[str, Any]: Validated dataset, model, channel, class, crop, precision, overlap, and seed settings; crop_size is a
            three-axis tuple.
    """
    from inference.core.checkpoint import model_variant
    from inference.core.data import physical_dataset, read_json
    from source.dataset.runtime_contracts import get_dataset_contract

    config_path = args.config
    if config_path is None:
        for parent in tuple(args.checkpoint.resolve().parents)[:4]:
            candidate = parent / "config.json"
            if candidate.is_file():
                config_path = candidate
                break
    config = {} if config_path is None else read_json(config_path)
    for name, default in (("split_type", "Normal"), ("train_ratio", None)):
        value = config.pop(name, default)
        if value != default:
            raise ValueError(f"Saved configuration uses unsupported {name}: {value!r}")
    dataset = physical_dataset(args.dataset)
    contract = get_dataset_contract(dataset)
    configured_dataset = config.get("dataset_name", config.get("dataset"))
    if configured_dataset and physical_dataset(configured_dataset) != dataset:
        raise ValueError("Saved configuration dataset differs from --dataset")
    variant = model_variant(args.model)
    config.update(dataset_name=dataset, model_name="flumina", model_variant=variant)
    for key, expected in (
        ("in_channels", contract.in_channels),
        ("num_classes", contract.num_classes),
    ):
        if key in config and config[key] != expected:
            raise ValueError(
                f"Saved {key} differs from the {dataset} label/channel contract"
            )
        config[key] = expected
    crop = args.crop_size if args.crop_size is not None else config.get("crop_size")
    if isinstance(crop, int):
        crop = (crop,) * 3
    if (
        not isinstance(crop, (tuple, list))
        or len(crop) != 3
        or any(type(size) is not int or size <= 0 for size in crop)
    ):
        raise ValueError(
            "Provide saved config.json with crop_size, --config, or explicit --crop-size D H W"
        )
    config["crop_size"] = tuple(crop)
    config["overlap"] = (
        args.overlap if args.overlap is not None else config.get("overlap", 0.5)
    )
    if not 0 <= config["overlap"] < 1:
        raise ValueError("overlap must be in [0, 1)")
    config["mixed_precision"] = args.mixed_precision or config.get(
        "mixed_precision", "no"
    )
    if config["mixed_precision"] not in {"no", "bf16", "fp16"}:
        raise ValueError("mixed_precision must be no, bf16, or fp16")
    config["seed"] = args.seed if args.seed is not None else config.get("seed", 42)
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if config_path is not None:
        config["config_file"] = config_path.name
    return config


def resolve_device(requested: str) -> str:
    """
    Normalize a CPU or available CUDA device selection.

    Args:
        requested (str): auto, cpu, cuda, or cuda:<index>.

    Returns:
        str: cpu or cuda:<index>; auto prefers the first available CUDA device.
    """
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    match device.type:
        case "cuda":
            index = device.index or 0
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ValueError("Requested CUDA device is unavailable")
            return f"cuda:{index}"
        case "cpu":
            return "cpu"
        case _:
            raise ValueError("Device must be auto, cpu, cuda, or cuda:<index>")


def predict(model, inputs, config: Mapping[str, Any], inference_augument: bool = True):
    """
    Assign class labels from Gaussian-blended sliding-window logits.

    Args:
        model (torch.nn.Module): Model mapping volume windows to raw class logits.
        inputs (torch.Tensor): Floating-point volume batch of shape (B, C, D, H, W).
        config (Mapping[str, Any]): num_classes, crop_size, and overlap settings.
        inference_augument (bool): Average logits over eight inverse-aligned spatial flip views before selecting classes.

    Returns:
        torch.Tensor: Integer class labels of shape (B, D, H, W) on the input device.
    """
    from source.core.inference import sliding_window_inference
    from source.inference import inference_with_augmentation

    def predictor(volume):
        """
        Run Gaussian sliding-window inference for one spatial view.

        Args:
            volume (torch.Tensor): Floating-point view of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Float32 logits of shape (B, K, D, H, W) when augmentation is enabled, otherwise integer class labels
                of shape (B, D, H, W).
        """
        options = {"return_logits": True} if inference_augument else {}
        return sliding_window_inference(
            model,
            volume,
            int(config["num_classes"]),
            config["crop_size"],
            overlap=config["overlap"],
            blend_mode="gaussian",
            **options,
        )

    output = inference_with_augmentation(inputs, predictor, inference_augument)
    return output.argmax(dim=1) if inference_augument else output


def autocast(device: str, precision: str):
    """
    Select the requested precision context for the inference device.

    Args:
        device (str): CPU or CUDA device identifier.
        precision (str): Precision mode: no, bf16, or fp16.

    Returns:
        torch.autocast | nullcontext: BF16 or CUDA FP16 context; no and CPU FP16 leave autocasting unchanged.
    """
    import torch

    if precision == "bf16":
        return torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
    if precision == "fp16" and device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def score_case(prediction, target, dataset: str, case, metrics: str):
    """
    Evaluate one categorical prediction under the selected metric protocol.

    Args:
        prediction (np.ndarray): Integer class map of shape (D, H, W).
        target (np.ndarray): Categorical reference on the same processed grid.
        dataset (str): Evaluation dataset or benchmark AMOS modality view.
        case (Case): Case spacing, identifier, and label integrity metadata.
        metrics (str): Metric protocol, benchmark or legacy.

    Returns:
        tuple[MetricRecord, ...]: Region records from inference.core.metrics for benchmark scoring or source.evaluation for
            legacy scoring.
    """
    if metrics == "benchmark":
        from inference.core.metrics import evaluate_case_records

        return evaluate_case_records(
            prediction,
            target,
            dataset,
            spacing=case.spacing,
            case_id=case.case_id,
        )
    from source.evaluation import evaluate_case

    if dataset not in DATASETS:
        raise ValueError("Legacy metrics require the combined AMOS dataset")
    metadata = dict(case.metadata)
    metadata["label_support.dropped_nonzero_label_voxels"] = case.metadata[
        "dropped_nonzero_label_voxels"
    ]
    metadata["label_support.integrity_status"] = case.metadata["data_integrity_status"]
    return evaluate_case(
        dataset,
        prediction,
        target,
        spacing=case.spacing,
        case_id=case.case_id,
        metadata=metadata,
    )


def finish_run(
    output: Path, records, case_by_id: Mapping, config: Mapping, args, epoch: int
):
    """
    Write per-case CSVs and region summaries for one evaluation cohort.

    Args:
        output (Path): Directory for the cohort's CSV files.
        records (Sequence[MetricRecord]): Nonempty records from the selected protocol, using inference.core.metrics.MetricRecord
            or source.evaluation.MetricRecord.
        case_by_id (Mapping[str, Case]): Case descriptors indexed by record case ID.
        config (Mapping[str, Any]): Evaluation configuration containing the run seed.
        args (argparse.Namespace): Selected metric protocol, model, and checkpoint.
        epoch (int): Checkpoint epoch written to output rows.

    Returns:
        dict[str, dict[str, float]]: Unscaled region means indexed by metric and region.
    """
    from inference.core.output import atomic_text, render_csv, summary_row

    if args.metrics == "benchmark":
        from inference.core.metrics import aggregate_records

        summaries = aggregate_records(records)
    else:
        from source.evaluation import MetricAccumulator

        accumulator = MetricAccumulator()
        accumulator.update(records)
        summaries = {}
        for row in accumulator.aggregate():
            summaries.setdefault(row.metric, {})[row.region] = row.value
    header = tuple(records[0].to_dict())
    rows = []
    for record in records:
        case = case_by_id[record.case_id]
        rows.append(
            [
                epoch,
                args.checkpoint.name,
                args.model,
                config["seed"],
                *case.spacing,
                case.metadata["data_integrity_status"],
                case.metadata["dropped_nonzero_label_voxels"],
                *(record.to_dict()[key] for key in header),
            ]
        )
    atomic_text(
        output / "test_results_per_case.csv",
        render_csv(
            (
                "epoch",
                "checkpoint",
                "model",
                "seed",
                "spacing_axis0_mm",
                "spacing_axis1_mm",
                "spacing_axis2_mm",
                "case_data_integrity_status",
                "case_dropped_nonzero_label_voxels",
                *header,
            ),
            rows,
        ),
    )
    for metric, values in summaries.items():
        scale = 100.0 if metric in {"DSC", "IoU"} else 1.0
        atomic_text(
            output / f"{metric}-{args.model}.csv",
            render_csv(
                ("Epoch", *values, "Average"),
                [summary_row(epoch, values, scale)],
            ),
        )
    dice = summaries["DSC" if args.metrics == "benchmark" else "dice"]
    atomic_text(
        output / "test_results.csv",
        render_csv(("Epoch", *dice, "Average"), [summary_row(epoch, dice)]),
    )
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run inference preflight or evaluate a checkpoint and write result artifacts.

    Preflight validates settings and data without loading checkpoint weights.

    Args:
        argv (Sequence[str] | None): Command-line tokens; None reads the process arguments.

    Returns:
        int: Zero for completion or preflight success, or one for handled evaluation errors.
    """
    args = parser().parse_args(argv)
    configure_runtime()
    import torch
    from inference.core.checkpoint import load_model, read_checkpoint
    from inference.core.data import load_case, prepare_data, sha256_file
    from inference.core.metrics import metric_protocol
    from inference.core.output import write_json
    from source.evaluation import get_task
    from source.config.setup_seed import setup_seed

    output = None
    metadata = {}
    try:
        if args.cpu_threads <= 0 or (
            args.max_cases is not None and args.max_cases <= 0
        ):
            raise ValueError("cpu-threads and max-cases must be positive")
        if args.amos_subsets and (
            args.dataset != "AMOS-2022" or args.metrics != "benchmark"
        ):
            raise ValueError("amos-subsets requires AMOS-2022 and benchmark metrics")
        if args.metrics == "legacy" and args.dataset not in DATASETS:
            raise ValueError("Legacy metrics require the combined AMOS dataset")
        config = load_config(args)
        data_root = (
            args.data_root
            or os.environ.get("FLUMINA_DATA_ROOT")
            or os.environ.get("DATA_PATH")
            or config.get("data_path")
        )
        if data_root is None:
            raise ValueError("Provide --data-root or FLUMINA_DATA_ROOT")
        data = prepare_data(
            args.dataset, config, data_root, membership_file=args.membership_file
        )
        device = resolve_device(args.device)
        if not args.checkpoint.is_file():
            raise FileNotFoundError("Checkpoint file does not exist")
        task = get_task(config["dataset_name"])
        metadata = {
            "status": "preflight_ready" if args.dry_run else "running",
            "dataset": args.dataset,
            "model": args.model,
            "checkpoint": args.checkpoint.name,
            "config": config,
            "data": data.provenance,
            "device": device,
            "metrics": args.metrics,
            "metric_protocol": metric_protocol()
            if args.metrics == "benchmark"
            else {
                "id": task.protocol_id,
                "metrics": task.metrics,
                "aggregation": task.aggregation,
            },
            "inference_augument": args.inference_augument,
            "prediction_rule": "argmax of eight inverse-aligned Gaussian-weighted logit volumes"
            if args.inference_augument
            else "argmax of Gaussian-weighted raw logits",
            "effective_precision": "no"
            if config["mixed_precision"] == "fp16" and device == "cpu"
            else config["mixed_precision"],
            "partial": args.max_cases is not None or args.membership_file is not None,
        }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            args.output_dir
            / args.dataset
            / args.model
            / f"seed{config['seed']}_{stamp}_{uuid4().hex[:8]}"
        )
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "inference.json", metadata)
        if args.dry_run:
            print(
                f"PREFLIGHT READY: {args.dataset}, {args.model}, {len(data.cases)} cases, {device}; "
                "checkpoint loading not checked"
            )
            return 0
        torch.set_num_threads(args.cpu_threads)
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        setup_seed(
            config["seed"], deterministic=bool(config.get("deterministic", False))
        )
        checkpoint = read_checkpoint(args.checkpoint)
        metadata["checkpoint_sha256"] = sha256_file(args.checkpoint)
        model = load_model(args.model, config, checkpoint, device)
        epoch = int(checkpoint.metadata.get("epoch", 0))
        metadata["epoch"] = epoch
        cases = data.cases[: args.max_cases]
        records = []
        for index, case in enumerate(cases, 1):
            print(
                f"[{args.dataset} / {args.model}] {index}/{len(cases)} {case.case_id}",
                flush=True,
            )
            image, target = load_case(
                case,
                in_channels=config["in_channels"],
                num_classes=config["num_classes"],
            )
            inputs = (
                torch.from_numpy(image).unsqueeze(0).to(device, dtype=torch.float32)
            )
            with torch.inference_mode(), autocast(device, config["mixed_precision"]):
                prediction = (
                    predict(model, inputs, config, args.inference_augument)[0]
                    .cpu()
                    .numpy()
                )
            records.extend(
                score_case(
                    prediction,
                    target,
                    args.dataset,
                    case,
                    args.metrics,
                )
            )
            if args.save_predictions:
                import numpy as np

                destination = output / "predictions"
                destination.mkdir(exist_ok=True)
                np.save(
                    destination / f"{case.case_id}.npy",
                    prediction.astype(np.uint8),
                    allow_pickle=False,
                )
            metadata["evaluated_case_count"] = index
            write_json(output / "inference.json", metadata)
            del inputs, image, prediction, target
        case_by_id = {case.case_id: case for case in cases}
        metadata["summaries"] = finish_run(
            output, records, case_by_id, config, args, epoch
        )
        if args.amos_subsets:
            from dataclasses import replace
            from inference.core.metrics import MRI_UNAVAILABLE_REGIONS

            metadata["amos_subsets"] = {}
            for modality in ("CT", "MRI"):
                selected = {
                    case.case_id
                    for case in cases
                    if case.metadata["modality"] == modality
                }
                subset = []
                for row in records:
                    if row.case_id not in selected:
                        continue
                    changes: dict[str, str | float | bool] = {
                        "dataset": f"AMOS-2022-{modality}"
                    }
                    if modality == "MRI" and row.region in MRI_UNAVAILABLE_REGIONS:
                        if row.gt_voxels:
                            raise ValueError(
                                "AMOS MRI target contains an unannotated label"
                            )
                        changes.update(
                            value=float("nan"), included=False, status="not_annotated"
                        )
                    subset.append(replace(row, **changes))
                if subset:
                    metadata["amos_subsets"][modality] = finish_run(
                        output / modality, subset, case_by_id, config, args, epoch
                    )
        metadata["status"] = "completed_partial" if metadata["partial"] else "completed"
        write_json(output / "inference.json", metadata)
        print(f"Completed {len(cases)} cases: {output}", flush=True)
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as error:
        if output is not None:
            metadata.update(status="failed", error_type=type(error).__name__)
            write_json(output / "inference.json", metadata)
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
