from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models import build_model
from source.config import TrainingConfig, setup_seed
from source.dataset import (
    setup_evaluation_dataloader,
    setup_train_val_dataloaders,
    validate_training_splits,
)
from source.dataset.runtime_contracts import get_dataset_contract
from source.evaluation_plan import build_evaluation_plan, preflight_evaluation_plan
from source.logger import Logger
from source.test import evaluate_checkpoints
from source.train import TrainingResult, prepare_run_directory, train_model


def resolve_device(requested: str) -> torch.device:
    """
    Resolve automatic device selection and validate CUDA availability.

    Args:
        requested (str): Device name, CUDA index, or "auto".

    Returns:
        torch.device: Requested device, or available CUDA/CPU for "auto".
    """

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def run(args: TrainingConfig) -> TrainingResult:
    """
    Train a model and evaluate the selected checkpoints on held-out cohorts.

    Args:
        args (TrainingConfig): Training, checkpoint, and evaluation settings.

    Returns:
        TrainingResult: Training artifacts and final update count.
    """

    run_dir = prepare_run_directory(args)
    evaluation_tasks = build_evaluation_plan(args.dataset_name)
    validate_training_splits(args)
    if evaluation_tasks:
        preflight_evaluation_plan(args, evaluation_tasks)
    setup_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    model = build_model(
        args.model_name,
        in_channels=args.in_channels,
        out_channels=args.num_classes,
        crop_size=args.crop_size,
        variant=args.model_variant,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.compile:
        model = cast(nn.Module, torch.compile(model, mode="default"))

    train_loader, val_loader = setup_train_val_dataloaders(args)
    with Logger(
        run_dir / "run.log",
        run_dir / "metrics.jsonl",
        clear_metrics=args.resume is None,
    ) as logger:
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        try:
            logger.log_config(args.as_dict())
            result = train_model(
                model,
                args,
                train_loader,
                val_loader,
                device=device,
                writer=writer,
                logger=logger,
                _run_directory_prepared=True,
            )
            if len(result.top_checkpoints) != args.top_k:
                raise RuntimeError(
                    f"Training produced {len(result.top_checkpoints)} ranked checkpoints; "
                    f"top_k={args.top_k} requires at least that many validation epochs"
                )
            evaluation_top_k = args.resolved_evaluation_top_k
            evaluation_checkpoints = result.top_checkpoints[:evaluation_top_k]
            logger.write(
                f"Final evaluation uses ranks 1..{evaluation_top_k} from "
                f"the {args.top_k} retained checkpoints.\n"
            )
            if evaluation_tasks:
                preflight_evaluation_plan(args, evaluation_tasks)
            for task in evaluation_tasks:
                contract = get_dataset_contract(task.evaluated_dataset_name)
                evaluation_args = replace(
                    args,
                    dataset_name=contract.dataset_name,
                    in_channels=contract.in_channels,
                    num_classes=contract.num_classes,
                )
                membership_file = task.membership_path
                test_loader, selection = setup_evaluation_dataloader(
                    evaluation_args,
                    dataset_name=contract.dataset_name,
                    membership_file=membership_file,
                    direct_dataset_name=args.dataset_name,
                )
                evaluation_kwargs = {
                    "output_dir": run_dir,
                    "top_k": evaluation_top_k,
                    "device": device,
                    "subset": task.list_key,
                    "logger": logger,
                    "results_path": run_dir / task.aggregate_filename,
                    "cases_path": run_dir / task.long_form_filename,
                }
                final = evaluate_checkpoints(
                    evaluation_checkpoints,
                    model,
                    evaluation_args,
                    test_loader,
                    **evaluation_kwargs,
                )
                logger.write(
                    f"Final evaluation {contract.dataset_name}/{task.list_key}: "
                    f"{selection.included_count} labeled cases, "
                    f"{selection.excluded_count} excluded\n"
                )
                logger.write(f"Final aggregate CSV: {final.csv_paths.results}\n")
                logger.write(f"Final per-case CSV: {final.csv_paths.cases}\n")
            return result
        finally:
            writer.close()


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse command-line arguments and run training.

    Args:
        argv (Sequence[str] | None): Arguments to parse; None uses sys.argv.

    Returns:
        int: Zero after training and evaluation complete.
    """

    run(TrainingConfig.from_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
