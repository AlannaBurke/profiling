# Visualizing Traces with Perfetto

While the summary table from `key_averages()` provides a good starting point, a timeline visualization is often necessary for a deep and intuitive understanding of your model's performance. The PyTorch Profiler can export a detailed trace of your model's execution, which can then be loaded into a specialized trace viewer. The recommended tool for this is **Perfetto**.

Perfetto is a powerful, open-source trace analysis tool developed by Google. It provides a web-based UI for visualizing and analyzing large traces, making it an ideal choice for deep learning performance analysis. This section will guide you through the process of using Perfetto to visualize your PyTorch Profiler traces.

## 1. Loading a Trace into Perfetto

First, you need to generate a trace file from the PyTorch Profiler, as we saw in the previous section:

```python
prof.export_chrome_trace("trace.json")
```

Once you have your `trace.json` file, open the Perfetto UI in your web browser by navigating to [ui.perfetto.dev](https://ui.perfetto.dev).

To load your trace, you can either click the "Open trace file" button or simply drag and drop your `trace.json` file onto the Perfetto UI.

## 2. Navigating the Perfetto UI

After loading your trace, you will be presented with the main Perfetto UI. The UI can be broken down into three main components:

*   **The Timeline:** This is the main area of the UI, where you can see a chronological view of the events that occurred during your profiling session.
*   **The Track Pane:** On the left side of the UI, you will see a list of tracks. Each track represents a different source of events, such as a CPU core or a GPU stream.
*   **The Details Pane:** When you select an event in the timeline, the details pane at the bottom of the UI will show you more information about that event.

Here is a conceptual diagram of the Perfetto UI:

![Perfetto UI Diagram](https://i.imgur.com/8f4p9gN.png)

Navigating the timeline is straightforward:

*   **Zoom:** Use the `W` and `S` keys to zoom in and out, or use the scroll wheel on your mouse.
*   **Pan:** Click and drag the timeline to pan left and right.
*   **Select:** Click on an event to select it and view its details.

## 3. Interpreting the Tracks

The real power of Perfetto lies in its ability to display multiple tracks of data in a synchronized view. This allows you to see how different parts of your system are interacting. Here are some of the key tracks you will see when you load a PyTorch Profiler trace:

| Track Name              | Description                                                                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`Tracer`**            | This track shows the overall duration of the profiling session.                                                                                                         |
| **`Processes`**         | This track shows the different processes that were running during the profiling session. You will typically see one process for your Python script.                       |
| **`GPU`**               | This track shows the activity on the GPU. You will see a separate sub-track for each CUDA stream, as well as a track for GPU memory copies.                               |
| **`CPU`**               | This track shows the activity on each CPU core. You can see which operators were running on which core and for how long.                                                  |
| **`PyTorch Profiler`**  | This track contains metadata about the profiling session, such as the labels you added with `record_function`.                                                          |

## 4. Identifying a Bottleneck

By examining the different tracks in the Perfetto UI, you can start to identify performance bottlenecks. For example, if you see a large gap in the GPU track, it means the GPU was idle during that time. You can then look at the CPU track to see what was happening on the CPU during that time. If you see a lot of activity in the data loading part of your code, it could indicate a data loading bottleneck.

In the example below, we can see a gap in the GPU track. By zooming in, we can see that the CPU is busy with data loading operations during this time, which is preventing the GPU from being fully utilized.

![Perfetto Bottleneck Example](https://i.imgur.com/2j2k3yD.png)

By using Perfetto to visualize your PyTorch Profiler traces, you can gain a deep understanding of your model's performance and identify opportunities for optimization. In the next section, we will explore some of the more advanced features of the PyTorch Profiler.
