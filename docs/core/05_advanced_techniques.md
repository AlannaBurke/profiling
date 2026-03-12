# Advanced Profiling APIs and Techniques

Beyond the basics of creating and visualizing traces, the PyTorch Profiler offers a suite of advanced features that provide finer control and deeper insights into your model's performance. This section covers some of the most important advanced APIs and techniques that can help you tackle more complex profiling scenarios.

## Scheduling Profiling Runs

In many cases, you don't need to profile every single iteration of your training loop. Profiling adds overhead, and often, the performance characteristics of your model are similar from one iteration to the next. The `torch.profiler.schedule` function allows you to specify exactly which iterations to profile, minimizing overhead while still capturing representative data.

A common pattern is to use a schedule that includes a `wait` period, a `warmup` period, and an `active` period.

*   **`wait`**: The profiler is disabled.
*   **`warmup`**: The profiler is running and collecting data, but the results are discarded. This is to account for the initial overhead of the profiler.
*   **`active`**: The profiler is running and recording data.

Here is an example of how to use a schedule to profile a training loop:

```python
my_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)

with torch.profiler.profile(schedule=my_schedule) as prof:
    for step in range(10):
        # ... your training code ...
        prof.step()  # Notify the profiler that a step has completed
```

In this example, the profiler will:

1.  **Wait** for the first step.
2.  **Warm up** during the second step.
3.  **Actively record** steps 3, 4, and 5.
4.  **Repeat** this cycle once more for steps 6, 7, and 8.

The `prof.step()` call is crucial for signaling the profiler to advance to the next state in its schedule.

## Customizing Traces with `record_function`

We have already seen how `record_function` can be used to add high-level labels to your code. However, it can also be used to add more granular labels to specific parts of your model or data loading pipeline. This is particularly useful for isolating the performance of custom components or for breaking down a complex operation into smaller, more understandable parts.

```python
with torch.profiler.profile(...) as prof:
    with torch.profiler.record_function("data_loading"):
        # ... your data loading code ...

    with torch.profiler.record_function("model_forward"):
        # ... your model's forward pass ...

    with torch.profiler.record_function("loss_computation"):
        # ... your loss computation ...
```

These custom labels will appear in your Perfetto trace, making it much easier to correlate the timeline with your source code.

## Memory Profiling

Understanding your model's memory usage is just as important as understanding its execution time. The PyTorch Profiler provides tools for tracking memory allocations and deallocations, helping you identify memory leaks or opportunities to reduce your model's memory footprint.

To enable memory profiling, set `profile_memory=True` in the `profile` context manager:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,
) as prof:
    # ... your code ...
```

While the `export_memory_timeline` function is now deprecated, the memory information is still captured in the trace and can be analyzed. The recommended way to analyze memory usage is to use the memory snapshot APIs, which provide a more detailed view of the memory landscape. A full guide on these new APIs will be provided in a dedicated tutorial.

## Profiling `torch.compile`

`torch.compile` is a powerful feature that can significantly speed up your models by JIT-compiling them. However, it also introduces a new layer of abstraction that can make profiling more challenging. When you profile a compiled model, you will see a new type of event in your trace: `Torch-Compiled Region`.

This event represents the execution of the compiled graph. To understand what is happening inside this region, you need to look at the detailed output of the compiler. The profiler can provide some insights, but a full analysis often requires a combination of profiling and compiler-specific debugging techniques. A dedicated tutorial will cover this topic in more detail.

## Estimating Operator FLOPS

The profiler can also estimate the number of floating-point operations (FLOPS) for certain operators, such as matrix multiplications and convolutions. This can be useful for understanding the computational complexity of your model and for comparing the efficiency of different implementations.

To enable FLOPS estimation, set `with_flops=True` in the `profile` context manager:

```python
with torch.profiler.profile(with_flops=True) as prof:
    # ... your code ...
```

The estimated FLOPS will be included in the `key_averages()` table.

By mastering these advanced features, you can take your performance analysis skills to the next level and tackle even the most challenging optimization tasks.
