# Model weights

Set `CHECKPOINT` in an inference launcher to the weights you want to evaluate.
Training saves ranked weights and their `config.json` under `runs/`.

For weights without an adjacent training configuration, set `CONFIG` to the
matching JSON in `configs/training/`. Match the training crop, precision,
channels, and class count used to produce the weights. Checkpoint files are
excluded from version control.
