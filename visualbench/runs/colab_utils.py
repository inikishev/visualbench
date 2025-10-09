import shutil
import zipfile


def load_movie_lens():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    output_filename = "ml-100k.zip"

    import requests
    response = requests.get(url, stream=True, timeout=100)
    response.raise_for_status()

    with open(output_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile("ml-100k.zip","r") as zip_ref:
        zip_ref.extractall("MovieLens-100k")

    return "MovieLens-100k/ml-100k"

def colab_pack():
    from google.colab import files # pylint:disable=import-error,no-name-in-module # pyright:ignore[reportMissingImports]

    shutil.make_archive("MLBench", 'zip', "MLBench")

    # Download the file
    files.download('MLBench.zip')

def kaggle_pack():
    shutil.make_archive("MLBench", 'zip', "MLBench")

def performance_tweaks(
    cudnn_bench: bool | None,
    onednn_fusion: bool | None=True,
    detect_anomaly: bool | None=False,
    checknan: bool | None=False,
    autograd_profiler: bool | None=False,
    emit_nvtx: bool | None=False,
    deterministic: bool | None=None,
    float32_matmul_precision: str | None = 'high',
    opt_einsum: bool | None = True,
    opt_einsum_strategy: str | None = 'auto-hq',
    gradcheck: bool | None=False,
    gradgradcheck: bool | None=False,
):
    """note that deterministic is none only for this one"""
    import torch
    # causes cuDNN to benchmark multiple convolution algorithms and select the fastest
    if cudnn_bench is not None: torch.backends.cudnn.benchmark = cudnn_bench

    # deterministic operations tend to have worse performance than nondeterministic operations
    if deterministic is not None:
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic)

    # Running float32 matrix multiplications in lower precision may significantly increase performance,
    # and in some programs the loss of precision has a negligible impact.
    # (default is "highest")
    if float32_matmul_precision is not None: torch.set_float32_matmul_precision(float32_matmul_precision)

    # operator-fusion (only for TorchScript inference)
    if onednn_fusion is not None: torch.jit.enable_onednn_fusion(onednn_fusion)

    # disables anomaly detection
    if detect_anomaly is not None: torch.autograd.set_detect_anomaly(detect_anomaly, checknan) # type:ignore

    # disable autograd profiler
    if autograd_profiler is not None: torch.autograd.profiler.profile(autograd_profiler) # type:ignore

    # disable nvtx emiting (which I am pretty sure is only for nvprof)
    if emit_nvtx is not None: torch.autograd.profiler.emit_nvtx(emit_nvtx) # type:ignore

    # optimizes contraction order for einsum operation
    if hasattr(torch.backends, "opt_einsum"):
        if opt_einsum is not None:
            torch.backends.opt_einsum.enabled = opt_einsum # pyright:ignore[reportAttributeAccessIssue]

        # larger search time (1 s. on 1st call) but very fast einsum
        if opt_einsum_strategy is not None: torch.backends.opt_einsum.strategy = opt_einsum_strategy # pyright:ignore[reportAttributeAccessIssue]
