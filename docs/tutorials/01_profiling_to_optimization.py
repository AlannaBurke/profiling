"""
From Profiling to Optimization: A Case Study
============================================

**Author:** Manus AI

This tutorial provides a practical, end-to-end example of how to use the
PyTorch Profiler to identify and fix a common performance bottleneck. We will
start with a model that has a data loading issue, use the profiler to diagnose
the problem, apply a fix, and then re-profile to verify the improvement.

Introduction
------------

The PyTorch Profiler is a powerful tool, but its output can be overwhelming
at first. The best way to learn how to use it is to see it in action. In this
tutorial, we will simulate a common real-world scenario: a training loop that
is bottlenecked by data loading. We will use the profiler to pinpoint the
issue and then demonstrate how a simple change to the `DataLoader` can lead
to a significant performance improvement.

"""

# %%
# 1. The Problem: A Slow Data Loading Pipeline
# --------------------------------------------
#
# Let's start by creating a synthetic dataset and a simple model. We will
# intentionally create a data loading bottleneck by using a small number of
# workers in our `DataLoader`.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

# Create a synthetic dataset
inputs = torch.randn(1000, 3, 224, 224)
labels = torch.randint(0, 1000, (1000,))
dataset = TensorDataset(inputs, labels)

# Create a DataLoader with a small number of workers (to simulate a bottleneck)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)

# Create a simple model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# %%
# 2. Profiling the Initial Code
# -----------------------------
#
# Now, let's profile our training loop to see if we can identify the
# bottleneck. We will use the `torch.profiler.profile` context manager to
# collect performance data.

import torch.profiler

def train(loader):
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18_slow'),
    record_shapes=True,
    with_stack=True
) as prof:
    for i, batch in enumerate(data_loader):
        train([batch])
        prof.step()
        if i >= 6: # Run for a few steps
            break

# %%
# 3. Analyzing the Trace
# ----------------------
#
# After running the code above, a trace file will be generated in the
# `./log/resnet18_slow` directory. We can now load this trace into Perfetto
# to visualize the timeline.
#
# When you open the trace, you will likely see a pattern like this:
#
# .. image:: https://i.imgur.com/2j2k3yD.png
#    :alt: Perfetto trace showing a data loading bottleneck
#
# Notice the large gaps in the GPU track. During these gaps, the GPU is idle.
# If you look at the CPU track during these times, you will see activity
# related to the `DataLoader`. This is a classic sign of a data loading
# bottleneck: the GPU is waiting for the CPU to load and preprocess the data.

# %%
# 4. The Fix: Increasing the Number of Workers
# ---------------------------------------------
#
# The fix for this particular bottleneck is straightforward: we need to
# increase the number of workers in our `DataLoader`. This will allow the
# data to be loaded in parallel, which should keep the GPU fed with data.

# Create a new DataLoader with more workers
fast_data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# %%
# 5. Profiling the Optimized Code
# -------------------------------
#
# Now, let's profile the training loop again with our new `DataLoader`.

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18_fast'),
    record_shapes=True,
    with_stack=True
) as prof:
    for i, batch in enumerate(fast_data_loader):
        train([batch])
        prof.step()
        if i >= 6:
            break

# %%
# 6. Verifying the Improvement
# ----------------------------
#
# If you now load the new trace from the `./log/resnet18_fast` directory into
# Perfetto, you should see a significant improvement. The gaps in the GPU
# track should be much smaller, indicating that the GPU is being utilized much
# more effectively.
#
# .. image:: https://i.imgur.com/v8b0g2Y.png
#    :alt: Perfetto trace showing improved GPU utilization
#
# Conclusion
# ----------
#
# This tutorial has demonstrated how to use the PyTorch Profiler to identify
# and fix a common performance bottleneck. By visualizing the execution
# timeline, we were able to quickly see that our GPU was being underutilized
# and that the data loading pipeline was the culprit. A simple change to the
# `DataLoader` was all it took to fix the issue and significantly improve the
# performance of our training loop.
#
# This is just one example of how the PyTorch Profiler can be used to
# optimize your models. By applying the same principles, you can diagnose and
# fix a wide range of performance issues in your own code.
