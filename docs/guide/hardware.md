# Apple silicon and AMD GPUs

LAZY is tested on NVIDIA GPUs, on CPUs and on the GPU of Apple silicon Macs,
which PyTorch reaches through its Metal backend (also called MPS). AMD GPUs,
which PyTorch reaches through its ROCm builds, have not been tested: the AMD
half of this page is read from the code of LAZY and of each model's package,
and we would welcome a report (an issue on
[GitHub](https://github.com/biprateep/lazy-tfm/issues)) of what works there.

## How LAZY chooses a device

With `device="auto"`, LAZY takes a CUDA GPU when `torch.cuda.is_available()`
is true, then an Apple GPU when `torch.backends.mps.is_available()` is true,
and the CPU otherwise. An AMD GPU is chosen by `"auto"` too:
PyTorch's ROCm builds expose AMD GPUs through the same `torch.cuda` interface,
as devices named `"cuda"` (see PyTorch's
[HIP notes](https://docs.pytorch.org/docs/stable/notes/hip.html)), so LAZY
treats them exactly as NVIDIA GPUs.

A few settings cannot run on a GPU that `"auto"` finds (see the limits
below). Then `"auto"` runs that model on the CPU and warns with the reason,
and naming the GPU outright, such as `device="mps"`, raises a `ValueError`.

Mixed precision (`mixed_precision=True`, the default) applies only to
`"cuda"` devices, so on an Apple GPU every model runs in single precision,
while on an AMD GPU each model uses the same reduced precision as on an NVIDIA
GPU (float16 for most of them and bfloat16 for TabFM's weights).

## Apple silicon

Every model runs on an Apple GPU, in single precision, with the limits below.
Each model was fitted afresh on the CPU and on the GPU of an Apple M3 Ultra
(28 CPU cores, 256 GB of memory, macOS 26.0.1, Python 3.12.16, torch 2.14.0
at its default 20 threads, tabpfn 9.0.0, tabicl 2.2.0, tabfm 1.0.1,
structured-data-models 0.2.1), at its defaults with `device="auto"` and
`random_state=0`, on the first 300 `california_housing` rows, and predicted
the next 50:

| Model | Ran on | CPU against `"auto"` | Seed 0 against seed 1 |
|---|---|---:|---:|
| TabPFN | GPU | $2.0 \times 10^{-5}$ | 0.06 |
| TabICL | GPU | $4.8 \times 10^{-6}$ | 0.12 |
| TabFM | GPU | $1.4 \times 10^{-6}$ | 0.11 |
| LimiX-2 | GPU | $1.3 \times 10^{-4}$ | 0.13 |
| Kumo Tabular | CPU (see below) | 0 | 0.11 |

The figures are the largest change of the 5%, 50% and 95% quantiles, as a
fraction of a row's 5% to 95% range, as in the
[Across devices](reproducibility.md#across-devices) table. Each device
repeats its own answer to the bit, and the models agree across devices to
float32 rounding. On an Apple M5 (10 CPU cores, 24 GB, macOS 26.6.2, torch
2.15.0), the GPU predicted 1.7 to 3.1 times faster than the CPU at each
model's defaults, and ran TabPFN contexts larger than TabPFN allows on a CPU
([#311](https://github.com/biprateep/lazy-tfm/issues/311)).

The limits are these:

- **Kumo Tabular.** NVIDIA's network writes matrix products into strided
  views of a buffer, which Metal gets wrong on some Macs, silently: on the M3
  Ultra above its predictions ignored the features. LAZY checks for this once
  per process, in about a millisecond. Where the check fails, `"auto"` runs
  Kumo Tabular on the CPU with a warning, and `device="mps"` raises; where it
  passes, Kumo Tabular runs on the GPU. On the M5 above, before LAZY had the
  check, Kumo Tabular on its GPU agreed with its CPU to
  $4 \times 10^{-5}$ of the range.
- **TabPFN's `kv_cache="fp8"`.** PyTorch has no float8 on Metal. Under
  `"auto"` such a model runs on the CPU with a warning; `kv_cache="int8"` and
  the default `kv_cache=True` run on the GPU.
- **Memory.** The GPU shares memory with the system. LimiX-2 sizes its
  key/value caches within the share Metal recommends for the GPU
  (`torch.mps.recommended_max_memory()`), and `lazy.clear_model_cache()`
  returns the memory of the networks it drops. PyTorch's
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (see its
  [MPS environment variables](https://docs.pytorch.org/docs/main/mps_environment_variables.html))
  and TabPFN's own `TABPFN_MPS_MEMORY_FRACTION` limit how much the GPU may
  take. No model needed `PYTORCH_ENABLE_MPS_FALLBACK`.

## AMD GPUs

The table summarizes, for each model, what its own package says about AMD
GPUs and what we expect from the code. "Expected" means read from the code
and not run.

| Model | AMD GPU (ROCm) |
| ----- | -------------- |
| TabPFN | Documented by TabPFN (install PyTorch's ROCm build first); expected to work |
| TabICL | Not mentioned by TabICL; it uses PyTorch's standard attention, so expected to work |
| TabFM | Not documented by TabFM; needs a GPU with bfloat16 support under mixed precision |
| LimiX-2 | Possible in principle through the Triton build that ships with ROCm's PyTorch, which then runs LimiX's Triton kernels on the AMD GPU |
| Kumo Tabular | Possible in principle: its optional Triton kernels would run through ROCm's Triton, and fall back to PyTorch where Triton is missing |

On a CPU, the models run on Linux, macOS and Windows. On an
NVIDIA GPU under Windows, LimiX-2 needs `triton`, which PyTorch installs on
Linux only, so LAZY refuses `device="cuda"` there unless `triton` is
installed. The table of {doc}`../models/index` shows which
of them are fast enough on a CPU, and {doc}`scaling` how fast.

On an AMD GPU, we first install PyTorch's ROCm build from the PyTorch ROCm
index, which according to
[AMD's instructions](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-triton.html)
also brings a matching Triton, and then use `device="auto"` or `"cuda"` as on
an NVIDIA GPU. If a model gives errors, or answers that differ from a CPU run
by more than rounding, `mixed_precision=False` turns off reduced precision in
every model, including TabFM's bfloat16 weights and the Triton path of Kumo
Tabular's normalization, although LimiX-2's and Kumo Tabular's other Triton
kernels still run on the GPU. LimiX-2 sizes its key/value cache from the free
memory PyTorch reports, which ROCm also provides, and `kv_cache=False` removes
that step if it misbehaves.

On either kind of hardware, a quick check that a model is usable is to fit it
on the same small context on the GPU and on the CPU (`device="cpu"`), and
compare the scores of the two (for example, `evaluate` on a held-out set).
They should be close but not identical, since some models draw their ensemble
members with a random generator that differs between devices, and reduced
precision changes the answers by its rounding (see {doc}`reproducibility`).
