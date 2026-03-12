# Getting Started: Your First Profiling Session

Now that we have a conceptual understanding of the PyTorch Profiler, let's walk through a practical example of how to use it. This section will guide you through the process of profiling a simple model, from setting up the code to interpreting the results.

## 1. Setting Up the Model and Data

First, let's create a simple model and some dummy data. We will use a ResNet18 model from `torchvision` and a random tensor as input.

```python
import torch
import torchvision.models as models

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)
```

## 2. Using the `torch.profiler.profile` Context Manager

The easiest way to use the profiler is with the `torch.profiler.profile` context manager. This context manager takes several arguments to configure the profiling session. For this example, we will profile both CPU and GPU activities and record the shapes of the operator inputs.

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True
) as prof:
    with torch.profiler.record_function("model_inference"):
        model(inputs)
```

Let's break down what's happening here:

*   We create a `profile` context and specify the `activities` we want to record. Since we have a CUDA-enabled device, we include both `ProfilerActivity.CPU` and `ProfilerActivity.CUDA`.
*   We set `record_shapes=True` to capture the input shapes of the operators. This can be very useful for debugging and performance analysis.
*   Inside the `profile` context, we use `record_function` to add a custom label to our model inference code. This will make it easier to find this specific block of code in the profiling results.

## 3. Analyzing the Results with `key_averages()`

Once the `profile` context is exited, the `prof` object contains all the collected performance data. One of the most useful methods for analyzing this data is `key_averages()`. This method aggregates the results by operator name and provides a summary table.

```python
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

This will print a table similar to the following (the exact numbers will vary):

```
---------------------------------  ------------  ------------  ------------  ------------
                             Name      Self CPU     CPU total  CPU time avg    # of Calls
---------------------------------  ------------  ------------  ------------  ------------
                  model_inference       5.509ms      57.503ms      57.503ms             1
                     aten::conv2d     231.000us      31.931ms       1.597ms            20
                aten::convolution     250.000us      31.700ms       1.585ms            20
               aten::_convolution     336.000us      31.450ms       1.573ms            20
         aten::mkldnn_convolution      30.838ms      31.114ms       1.556ms            20
                 aten::batch_norm     211.000us      14.693ms     734.650us            20
     aten::_batch_norm_impl_index     319.000us      14.482ms     724.100us            20
          aten::native_batch_norm       9.229ms      14.109ms     705.450us            20
                       aten::mean     332.000us       2.631ms     125.286us            21
                     aten::select       1.668ms       2.292ms       8.988us           255
---------------------------------  ------------  ------------  ------------  ------------
```

Here's how to interpret the columns:

*   **Name:** The name of the operator or `record_function` label.
*   **Self CPU:** The total time spent in the operator itself, not including the time spent in any operators it calls.
*   **CPU total:** The total time spent in the operator, including the time spent in any operators it calls.
*   **CPU time avg:** The average time spent in the operator per call.
*   **# of Calls:** The number of times the operator was called.

From this table, we can see that the `aten::conv2d` operator is the most time-consuming, which is expected for a convolutional neural network.

## 4. Exporting a Trace for Visualization

While the `key_averages()` table is useful for a quick overview, a timeline visualization can provide a much more detailed picture of your model's execution. The profiler can export the collected data into a Chrome-compatible trace file.

```python
prof.export_chrome_trace("trace.json")
```

This will create a file named `trace.json` in your current directory. In the next section, we will learn how to use Perfetto to visualize this file and analyze this file.
