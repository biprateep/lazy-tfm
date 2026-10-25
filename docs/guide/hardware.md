# Apple silicon and AMD GPUs

We have tested LAZY only on NVIDIA GPUs and on CPUs. PyTorch can also run on
the GPU of an Apple silicon Mac, through its Metal backend (also called MPS),
and on AMD GPUs, through its ROCm builds, but none of the five models has been
run on either in LAZY. This page therefore describes what to expect from
reading the code of LAZY and of each model's package, which limitations are
already known, and what to try. Everything here is untested, and we would
welcome a report (an issue on
[GitHub](https://github.com/biprateep/lazy-tfm/issues)) of what works and what
does not.

## How LAZY chooses a device

With `device="auto"`, LAZY takes the GPU when `torch.cuda.is_available()` is
true and the CPU otherwise, so it never chooses an Apple GPU on its own. On a
Mac, a model therefore runs on the CPU unless `device="mps"` is passed, and a
model that is slow on a CPU warns that PyTorch sees no GPU, even when the
Metal backend is available. An AMD GPU, in contrast, is chosen by `"auto"`:
PyTorch's ROCm builds expose AMD GPUs through the same `torch.cuda` interface,
as devices named `"cuda"` (see PyTorch's
[HIP notes](https://docs.pytorch.org/docs/stable/notes/hip.html)), so LAZY
treats them exactly as NVIDIA GPUs. Mixed precision (`mixed_precision=True`,
the default) applies only to `"cuda"` devices, so on an Apple GPU every model
runs in single precision, while on an AMD GPU each model uses the same reduced
precision as on an NVIDIA GPU (float16 for most of them and bfloat16 for
TabFM's weights).

## Each model

The table summarizes, for each model, what its own package says about the
hardware and what we expect from the code. "Expected" means read from the code
and not run.

| Model | Apple silicon (`device="mps"`) | AMD GPU (ROCm) |
| ----- | ------------------------------ | -------------- |
| TabPFN | Supported by TabPFN itself, which needs PyTorch 2.6 or newer for it; expected to work, in single precision | Documented by TabPFN (install PyTorch's ROCm build first); expected to work |
| TabICL | Supported by TabICL itself; expected to work, in single precision | Not mentioned by TabICL; it uses PyTorch's standard attention, so expected to work |
| TabFM | Not documented by TabFM; may fail when double-precision arrays reach the GPU, which Metal does not support | Not documented by TabFM; needs a GPU with bfloat16 support under mixed precision |
| LimiX-2 | Not possible: LimiX needs `triton`, which is published for Linux only | Possible in principle through the Triton build that ships with ROCm's PyTorch, which then runs LimiX's Triton kernels on the AMD GPU |
| Kumo Tabular | Refused by LAZY, since its preprocessing runs in double precision, which Metal does not support | Possible in principle: its optional Triton kernels would run through ROCm's Triton, and fall back to PyTorch where Triton is missing |

On a CPU, the models are expected to run on Linux, macOS and Windows, except
LimiX-2, which needs Linux. The table of {doc}`../models/index` shows which
of them are fast enough on a CPU, and {doc}`scaling` how fast.

Two of these limitations are choices of LAZY rather than of the models. Kumo
Tabular's own package runs on Metal by converting its double-precision steps
to single precision, which LAZY refuses so that its preprocessing gives the
same numbers on every device. Since `device="auto"` was written for CUDA, it
also never selects Metal, whereas TabPFN and TabICL, used on their own, prefer
it to the CPU.

## Making a model run

On an Apple silicon Mac, we suggest the following. We first pass
`device="mps"` explicitly, with a recent PyTorch (TabPFN recommends 2.13 or
newer). If PyTorch then reports an operation that Metal does not implement,
setting the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` before
importing PyTorch runs those operations on the CPU instead (see PyTorch's
[MPS environment variables](https://docs.pytorch.org/docs/main/mps_environment_variables.html)),
although it does not help with double precision, which Metal does not support
at all. The same page describes `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, which
limits how much of the shared memory PyTorch may take, and TabPFN has its own
`TABPFN_MPS_MEMORY_FRACTION`. TabPFN's `kv_cache="fp8"` is not available on
Metal, so we keep its default `kv_cache=True`. If TabFM fails with a
double-precision error, `kv_cache=False` sends its predictions through TabFM's
own path, which converts the features before moving them to the GPU, although
the targets may still fail in the same way. Kumo Tabular and LimiX-2 run on
the CPU (`device="cpu"`), LimiX-2 only on Linux.

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
