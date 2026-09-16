"""Build script for the row2col_grouped CUDA extension.

Usage
-----
    cd llmfoundry/models/ops/fp8/triton_kernels/cuda_row2col
    pip install -e .          # editable install
    # or:
    python setup.py build_ext --inplace
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_DIR = os.path.dirname(os.path.abspath(__file__))

_arch_flags = []
_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
if _arch_list:
    for _spec in _arch_list.replace(",", " ").split():
        _sm = _spec.replace(".", "")
        _arch_flags += [f"-gencode=arch=compute_{_sm},code=sm_{_sm}"]
else:
    _arch_flags = ["-arch=native"]

setup(
    name="row2col_cuda",
    ext_modules=[
        CUDAExtension(
            name="row2col_cuda_ext",
            sources=[os.path.join(_DIR, "row2col_kernel.cu")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    *_arch_flags,
                    "--use_fast_math",
                    "-lineinfo",
                    "--ptxas-options=-v",
                    "-std=c++17",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
