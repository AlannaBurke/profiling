"""
Profiling Distributed Training with Holistic Trace Analysis (HTA)
===============================================================

**Author:** Manus AI

This tutorial provides a guide to profiling distributed training jobs using the
PyTorch Profiler and analyzing the results with Holistic Trace Analysis (HTA).
We will cover how to collect traces from multiple ranks and how to use HTA to
identify common distributed training bottlenecks.

Introduction
------------

Profiling distributed training jobs presents a unique set of challenges.
It's not enough to look at the performance of a single GPU; you need to
understand how all the GPUs in your system are interacting. Are they working
in parallel, or are they waiting on each other? Is the communication between
GPUs efficient? These are the kinds of questions that HTA is designed to
answer.

HTA is a powerful open-source tool developed by Meta for analyzing PyTorch
Profiler traces from distributed workloads. It takes the individual traces
from each rank and stitches them together, providing a holistic view of your
training job's performance.

"""

# %%
# 1. Setting up a Distributed Training Job
# ----------------------------------------
#
# First, let's set up a simple distributed training job. We will use PyTorch's
# `DistributedDataParallel` (DDP) to train a ResNet18 model on a synthetic
# dataset. This code is a simplified version of a typical distributed training
# script.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torchvision.models as models
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)

    # Create a synthetic dataset
    inputs = torch.randn(1000, 3, 224, 224)
    labels = torch.randint(0, 1000, (1000,))
    dataset = TensorDataset(inputs, labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Create a simple model
    model = models.resnet18().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    # The training loop
    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i >= 9: # Run for a few steps
            break

    cleanup()

# %%
# 2. Collecting Traces from Multiple Ranks
# ----------------------------------------
#
# To profile a distributed job, you need to wrap the training loop on each
# rank with the PyTorch Profiler. It is crucial to use the
# `tensorboard_trace_handler` and to specify a different directory for each
# rank's trace.

import torch.profiler

def profiled_train_ddp(rank, world_size):
    setup(rank, world_size)

    # ... (dataset and model setup as before) ...

    trace_dir = f"./log/ddp_rank_{rank}"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for i, (data, target) in enumerate(data_loader):
            # ... (training step as before) ...
            prof.step()
            if i >= 6:
                break

    cleanup()

# %%
# 3. Launching the Distributed Profiling Run
# -------------------------------------------
#
# Now, you need to launch the `profiled_train_ddp` function on multiple
# processes. You can do this using `torch.multiprocessing.spawn`.

import torch.multiprocessing as mp

if __name__ == "__main__":
    world_size = 2 # Use 2 GPUs for this example
    mp.spawn(profiled_train_ddp, args=(world_size,), nprocs=world_size, join=True)

# %%
# 4. Analyzing the Traces with HTA
# --------------------------------
#
# After running the code above, you will have two trace directories:
# `./log/ddp_rank_0` and `./log/ddp_rank_1`. Now, you can use HTA to analyze
# these traces together.
#
# First, you need to install HTA:
#
# .. code-block:: sh
#
#    pip install HolisticTraceAnalysis
#
# Then, you can use the HTA library to load and analyze the traces.

from hta.trace_analysis import TraceAnalysis

# The directory containing the traces from all ranks
trace_dir = "./log/"

analyzer = TraceAnalysis(trace_dir=trace_dir)

# %%
# 5. HTA Analysis: Communication vs. Computation
# ----------------------------------------------
#
# One of the most useful features of HTA is its ability to break down the
# time spent in communication versus computation. The `get_comm_comp_overlap`
# function provides a detailed analysis of this.

comm_comp_overlap = analyzer.get_comm_comp_overlap()
print(comm_comp_overlap)

# %%
# The output of this function will be a table that shows, for each rank, the
# percentage of time spent in communication, computation, and the overlap
# between the two. A high overlap is desirable, as it means the GPU is not
# sitting idle waiting for communication to complete.
#
# HTA provides many other useful analysis functions, such as:
#
# * `get_temporal_breakdown()`: Shows the time spent in different categories
#   (e.g., idle, compute, memory).
# * `get_idle_time_breakdown()`: Breaks down the idle time into different
#   categories (e.g., host wait, kernel wait).
# * `get_gpu_kernel_breakdown()`: Provides a detailed breakdown of the time
#   spent in different GPU kernels.
#
# Conclusion
# ----------
#
# This tutorial has provided a brief introduction to profiling distributed
# training jobs with the PyTorch Profiler and HTA. By collecting traces from
# all ranks and using HTA to analyze them, you can gain a deep understanding
# of your distributed training performance and identify opportunities for
# optimization. For more information on HTA, please refer to the official
# [HTA documentation](https://hta.readthedocs.io/).
