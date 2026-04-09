"""
Profiling Model Execution with PyTorch Profiler
===============================================

In this tutorial, we will demonstrate how to use the PyTorch Profiler to analyze
the execution of a model, identify performance bottlenecks, and verify
optimizations.

We will focus on the most common use case: understanding how your model executes
on the CPU and GPU, rather than profiling data loading. We will use the
`torch.profiler.profile` context manager and visualize the resulting trace in
Perfetto.

1. Setup and Model Definition
-----------------------------

First, let's define a simple model. To demonstrate the profiler's capabilities,
we will intentionally introduce a performance bottleneck: an inefficient custom
linear layer that performs operations sequentially instead of in parallel.
"""

import torch
import torch.nn as nn
import torch.profiler

class InefficientLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # INTENTIONAL BOTTLENECK:
        # Instead of a single matrix multiplication, we do a slow loop.
        # This will show up clearly in the profiler trace.
        out = torch.zeros(x.size(0), self.weight.size(0), device=x.device)
        for i in range(x.size(0)):
            for j in range(self.weight.size(0)):
                out[i, j] = torch.dot(x[i], self.weight[j]) + self.bias[j]
        return out

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            InefficientLinear(256, 128), # Our bottleneck
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.features(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device)
inputs = torch.randn(32, 128).to(device)

######################################################################
# 2. Profiling the Execution
# --------------------------
#
# We use `torch.profiler.profile` to record the execution. We will configure
# it to record both CPU and CUDA activities, capture input shapes, and record
# the call stack to help us pinpoint the exact line of code causing the issue.
#
# We also use `torch.profiler.schedule` to skip the first step (warmup) and
# record the subsequent steps.

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(5):
        # We use record_function to add a label to the trace
        with torch.profiler.record_function("model_forward"):
            outputs = model(inputs)
        prof.step()

######################################################################
# 3. Analyzing the Results with `key_averages`
# --------------------------------------------
#
# We can print a summary table to get an immediate overview of where the time
# is being spent.

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

######################################################################
# The output will clearly show that `aten::dot` and the Python loop inside
# `InefficientLinear` are taking up the vast majority of the execution time,
# confirming our intentional bottleneck.
#
# 4. Visualizing the Trace in Perfetto
# ------------------------------------
#
# For a deeper understanding, we export the trace and view it in Perfetto.
# Perfetto is a web-based trace viewer available at https://ui.perfetto.dev.

prof.export_chrome_trace("simple_model_trace.json")

######################################################################
# You can drag and drop `simple_model_trace.json` into the Perfetto UI.
# In the trace, you will see:
#
# 1.  A massive block of CPU time dedicated to thousands of tiny `aten::dot`
#     operations.
# 2.  The GPU timeline (if running on CUDA) will show many small kernel
#     launches with large gaps between them, indicating that the GPU is
#     starved for work while waiting for the CPU loop.
#
# 5. Optimizing the Model
# -----------------------
#
# Now that we have identified the bottleneck, let's fix it by replacing
# the inefficient loop with a standard PyTorch matrix multiplication
# (`nn.Linear`).

class OptimizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128), # Fixed: Standard linear layer
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.features(x)

optimized_model = OptimizedModel().to(device)

######################################################################
# 6. Verifying the Optimization
# -----------------------------
#
# Let's profile the optimized model to verify the improvement.

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof_opt:
    for step in range(5):
        with torch.profiler.record_function("optimized_model_forward"):
            outputs = optimized_model(inputs)
        prof_opt.step()

print(prof_opt.key_averages().table(sort_by="cpu_time_total", row_limit=10))
prof_opt.export_chrome_trace("optimized_model_trace.json")

######################################################################
# The new summary table will show a dramatic reduction in execution time.
# If you load `optimized_model_trace.json` into Perfetto, you will see
# a much denser GPU timeline with large, efficient kernel executions
# (`aten::addmm`) and very little CPU overhead.
