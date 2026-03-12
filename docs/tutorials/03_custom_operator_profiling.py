"""
Profiling Custom C++ and CUDA Operators
=======================================

**Author:** Manus AI

This tutorial explains how to add profiling support to your custom C++ and
CUDA operators in PyTorch. By instrumenting your custom code with the
`record_function` API, you can gain visibility into its performance and see
how it interacts with the rest of your model.

Introduction
------------

While PyTorch provides a rich library of optimized operators, you may
sometimes need to implement your own custom operators in C++ or CUDA to achieve
the performance you need. When you do this, it's important to be able to
profile your custom code alongside the rest of your PyTorch model.

The PyTorch Profiler can be extended to recognize and record events from your
custom operators. This is done using the `torch::autograd::profiler::record_function`
API in C++.

"""

# %%
# 1. Creating a Custom Operator
# -----------------------------
#
# First, let's create a simple custom operator in C++. We will create a simple
# `add` operator that adds two tensors. We will also create a CUDA version of
# this operator.
#
# We will use PyTorch's C++ extension mechanism to build this operator. First,
# let's write the C++ and CUDA code.

cpp_code = """
#include <torch/extension.h>

// Custom C++ operator
torch::Tensor custom_add_cpu(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

// Custom CUDA operator forward declaration
torch::Tensor custom_add_cuda(torch::Tensor a, torch::Tensor b);

// Binding the operators to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_add_cpu", &custom_add_cpu, "Custom Add (CPU)");
    m.def("custom_add_cuda", &custom_add_cuda, "Custom Add (CUDA)");
}
"""

cuda_code = """
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Custom CUDA kernel
__global__ void custom_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Custom CUDA operator
torch::Tensor custom_add_cuda(torch::Tensor a, torch::Tensor b) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    torch::Tensor out = torch::empty_like(a);
    int size = a.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    custom_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

# %%
# 2. Instrumenting the Custom Operator with `record_function`
# -----------------------------------------------------------
#
# Now, let's modify our custom operator to include profiling information. We
# will use the `torch::autograd::profiler::record_function` class to create a
# RAII-style guard that records the start and end of our operator's execution.

cpp_code_instrumented = """
#include <torch/extension.h>
#include <torch/csrc/autograd/profiler.h>

torch::Tensor custom_add_cpu_instrumented(torch::Tensor a, torch::Tensor b) {
    torch::autograd::profiler::RecordFunction record("custom_add_cpu_instrumented");
    return a + b;
}

torch::Tensor custom_add_cuda_instrumented(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_add_cpu_instrumented", &custom_add_cpu_instrumented, "Custom Add Instrumented (CPU)");
    m.def("custom_add_cuda_instrumented", &custom_add_cuda_instrumented, "Custom Add Instrumented (CUDA)");
}
"""

cuda_code_instrumented = """
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/profiler.h>

__global__ void custom_add_kernel_instrumented(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor custom_add_cuda_instrumented(torch::Tensor a, torch::Tensor b) {
    torch::autograd::profiler::RecordFunction record("custom_add_cuda_instrumented");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
    torch::Tensor out = torch::empty_like(a);
    int size = a.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    custom_add_kernel_instrumented<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

# %%
# 3. Building the Custom Operator
# -------------------------------
#
# Now, let's build our custom operator using `torch.utils.cpp_extension.load`.

from torch.utils.cpp_extension import load

# Create directories for the source files
import os
if not os.path.exists('custom_op_src'):
    os.makedirs('custom_op_src')

with open('custom_op_src/custom_op.cpp', 'w') as f:
    f.write(cpp_code_instrumented)

with open('custom_op_src/custom_op.cu', 'w') as f:
    f.write(cuda_code_instrumented)

custom_op = load(
    name='custom_op',
    sources=['custom_op_src/custom_op.cpp', 'custom_op_src/custom_op.cu'],
    verbose=True
)

# %%
# 4. Profiling the Custom Operator
# --------------------------------
#
# Now that we have our instrumented custom operator, let's use it in a model
# and profile it.

import torch.profiler

device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(1000, 1000).to(device)
b = torch.randn(1000, 1000).to(device)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/custom_op')
) as prof:
    if device == 'cuda':
        custom_op.custom_add_cuda_instrumented(a, b)
    else:
        custom_op.custom_add_cpu_instrumented(a, b)
    prof.step()

# %%
# 5. Analyzing the Trace
# ----------------------
#
# If you now load the trace from the `./log/custom_op` directory into
# Perfetto, you will see our custom operator's label in the timeline.
#
# .. image:: https://i.imgur.com/your_image_here.png
#    :alt: Perfetto trace showing the custom operator
#
# This allows you to see exactly how much time is being spent in your custom
# code and how it is interacting with the rest of your model.
#
# Conclusion
# ----------
#
# This tutorial has shown how to add profiling support to your custom C++ and
# CUDA operators. By using the `torch::autograd::profiler::record_function`
# API, you can seamlessly integrate your custom code with the PyTorch
# Profiler and gain valuable insights into its performance.
