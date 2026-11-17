# Flumina inference models

Each attention implementation contains complete `Flumina-S`, `Flumina-B`,
`Flumina-M`, and `Flumina-L` packages. Each model directory includes its own
encoder, decoder, attention adapter, and Triton resize kernel. It can be copied
and imported independently, with no imports from other project directories.

| Directory | Attention implementation | Additional dependency |
| --- | --- | --- |
| `Original` | PyTorch SDPA with the Math backend forced | None |
| `SDPA-auto` | PyTorch automatic SDPA dispatch | None |
| `SDPA-Flash` | PyTorch SDPA with the Flash backend forced | None |
| `FlashAttention-4` | CuTe BF16 FlashAttention, using the original head-dimension scale | `flash-attn-4` and its CuTe dependencies |
| `SageAttention2++` | INT8 Q/K and FP8 P/V with BF16 interfaces | `sageattention` CUDA extensions |
| `SageAttention3` | FP4 attention with the original scale and private key storage | `sageattn3` CUDA extensions |

All packages require PyTorch and Triton. Attention runs under BF16 autocast
with gradients disabled and the model in evaluation mode. FA4 and Sage backends
require their supported CUDA hardware and installed extensions; their adapters
do not silently substitute a different attention implementation.

## Loading a model

From the repository root:

```python
from importlib import import_module

import torch

package = import_module("models.Flumina-Inference.SDPA-auto.Flumina-S")
model = package.build_model(
    in_channels=4,
    num_classes=4,
    crop_size=(128, 128, 128),
).float().cuda().eval()
model.load_state_dict(state_dict, strict=True)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(images)
```

Here `state_dict` contains the trained model's weights, and `images` has shape
`(B, 4, 128, 128, 128)`. Select the original training crop, input channels, output
classes, model size, and `attention_post_norm` setting when constructing a model.
The decoder option introduces no parameters or persistent buffers.

To import a copied model directory from any working directory:

```python
import importlib.util
from pathlib import Path
import sys

directory = Path("/path/to/Flumina-S")
spec = importlib.util.spec_from_file_location(
    "flumina_inference_model",
    directory / "__init__.py",
    submodule_search_locations=[str(directory)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
model = package.build_model(in_channels=4, num_classes=4, crop_size=(128, 128, 128))
```

## Decoder execution

The default `decoder_backend="triton"` fuses exact 2x trilinear interpolation
with the input cast of the immediately following BF16-autocast convolution.
Interpolation arithmetic remains float32. The fast path requires a CUDA
float32 input, BF16 autocast, positive strides, and no required backward pass.
The first decoder stage receives BF16 features and retains PyTorch interpolation.
Other unsupported inputs, including CPU inputs, also use PyTorch interpolation.
Difference-based interpolation retains the source kernel's overflow limitation
for opposite-sign extreme float32 inputs.

Pass `decoder_backend="torch"` to select PyTorch interpolation, or call
`model.set_decoder_backend("torch")` before compilation. If changing the option
after compiling a model, create a fresh compiled instance. Compilation is
optional; the source setup uses
`torch.compile(model, mode="default", fullgraph=True, dynamic=False)`.

Attention adapters retain their global PyTorch custom-operation names. Use one
model package per worker process when running these adapters. Importing multiple
copies of a backend can replace its dispatcher registrations.

FA4 uses its public default when `FLUMINA_FA4_TUNING_FILE` is unset. An optional
tuning file must contain completed measurements for the current source,
environment, device, and attention shapes. Source fingerprints include the local
adapter and tuning helper, so tuning files made before source or formatting
changes must be regenerated.
