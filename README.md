# PyTorch Profiler & Kineto Documentation

This repository contains a complete set of open-source documentation and tutorials for the PyTorch Profiler and its underlying Kineto library. This content was created to fill critical gaps in the official PyTorch documentation and is intended for contribution to the `pytorch/tutorials` and `pytorch/pytorch` repositories.

## Repository Structure

The repository is organized as follows:

```
.
├── README.md
├── pytorch_profiler_kineto_docs.md
├── pdf/
│   └── pytorch_profiler_kineto_docs.pdf
└── docs/
    ├── core/
    │   ├── 01_introduction.md
    │   ├── 02_architecture.md
    │   ├── 03_getting_started.md
    │   ├── 04_visualizing_traces.md
    │   └── 05_advanced_techniques.md
    ├── tutorials/
    │   ├── 01_profiling_to_optimization.py
    │   ├── 02_distributed_profiling_hta.py
    │   └── 03_custom_operator_profiling.py
    └── advanced/
        ├── 01_memory_profiling.md
        └── 02_torch_compile_profiling.md
```

### Root Directory

*   `README.md`: This file, providing an overview of the repository.
*   `pytorch_profiler_kineto_docs.md`: A comprehensive, single-file document containing all the documentation and tutorials.

### `pdf/` Directory

*   `pytorch_profiler_kineto_docs.pdf`: A PDF version of the main documentation file.

### `docs/` Directory

This directory contains the modular documentation files, organized by topic.

*   **`core/`**: Contains the core documentation for the PyTorch Profiler. These files are intended to be contributed to the main PyTorch documentation.
*   **`tutorials/`**: Contains in-depth, executable tutorials in the form of Python scripts. These are formatted for use with sphinx-gallery and are intended for the `pytorch/tutorials` repository.
*   **`advanced/`**: Contains deep-dive guides on advanced profiling topics.

## File Descriptions

A brief description of each file is provided in the main `pytorch_profiler_kineto_docs.md` file.

# PyTorch Profiler Documentation Audit Report

## 1. Introduction

This report provides a comprehensive audit of the official PyTorch documentation for its performance profiling tools, including the `torch.profiler` module and the underlying Kineto library. The objective of this audit was to identify the scope of existing documentation, pinpoint critical gaps and outdated information, and detail the new, comprehensive documentation suite created to address these deficiencies.

The new documentation was developed based on the findings of this audit. It aims to provide the PyTorch community with a complete, accurate, and high-quality resource for performance analysis and optimization.

---

## 2. Official Sources Audited

The following official PyTorch and related source materials were reviewed to assess the state of the existing documentation as of March 2026.

| # | Source Title | URL |
| - | ------------ | --- |
| 1 | `torch.profiler` API Reference | `https://docs.pytorch.org/docs/stable/profiler.html` |
| 2 | PyTorch Profiler Recipe | `https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html` |
| 3 | PyTorch Profiler With TensorBoard Tutorial | `https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html` |
| 4 | Introduction to Holistic Trace Analysis (HTA) | `https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html` |
| 5 | Legacy Profiler Tutorial | `https://docs.pytorch.org/tutorials/beginner/profiler.html` |
| 6 | Inductor CPU Backend Debugging and Profiling | `https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html` |
| 7 | Kineto GitHub Repository | `https://github.com/pytorch/kineto` |

---

## 3. Analysis of Existing Documentation

This section details the content covered by each audited source and identifies key gaps.

| Source Audited | Scope and Content | Completeness and Gaps |
| :--- | :--- | :--- |
| **`torch.profiler` API Reference** [1] | Provides a reference for the main classes and functions: `profile`, `schedule`, `ProfilerActivity`, `record_function`, and methods like `key_averages`, `export_chrome_trace`, and `export_stacks`. | **Gap:** Lacks a conceptual overview of the profiler architecture (Kineto, CUPTI). **Outdated:** The `export_memory_timeline` function is marked as deprecated, but no clear replacement is documented on this page. The TensorBoard handler is not marked as deprecated here. |
| **PyTorch Profiler Recipe** [2] | A basic tutorial covering `profile` usage, analyzing execution time with `key_averages`, basic memory profiling, Chrome trace export, and using `schedule`. | **Gap:** Provides only a surface-level overview. It does not cover how to interpret the Chrome trace, nor does it address advanced topics like distributed profiling or `torch.compile`. It serves as a good "hello world" but lacks depth. |
| **Profiler w/ TensorBoard Tutorial** [3] | A tutorial on using the (now deprecated) TensorBoard plugin to visualize profiler output. | **Outdated:** The entire tutorial is obsolete. A prominent warning box states that the TensorBoard integration is deprecated and that users should use Perfetto or `chrome://tracing` instead. **Critical Gap:** No replacement tutorial exists to guide users on the recommended Perfetto workflow. |
| **HTA Intro Tutorial** [4] | An introductory guide to the Holistic Trace Analysis (HTA) tool for analyzing distributed training traces. It covers temporal breakdown, idle time analysis, and communication/computation overlap. | **Good Coverage:** Provides a solid introduction to HTA. **Gap:** It does not cover the process of *generating* the traces for a distributed job within a PyTorch script, only the analysis part. It assumes the user already has the trace files. |
| **Legacy Profiler Tutorial** [5] | A tutorial for the old `torch.autograd.profiler` API. | **Outdated:** This entire tutorial is based on a legacy API that has been superseded by `torch.profiler`. A note at the top directs users to the new API, but the content itself is no longer relevant best practice. |
| **Inductor CPU Profiling Tutorial** [6] | A tutorial on debugging and profiling `torch.compile` with the Inductor backend. It mentions using the profiler to see compiled regions. | **Thin Coverage:** The tutorial focuses more on debugging correctness with `TORCH_COMPILE_DEBUG` and `explain()`. The profiling section is very brief and doesn't provide a deep dive into how to interpret the `Torch-Compiled Region` or diagnose compiler-related performance issues. |
| **Kineto GitHub Repo** [7] | The README provides a high-level mission statement for Kineto as a CPU+GPU profiling library. It links to the `tb_plugin` and HTA. | **Critical Gap:** There is no architectural documentation. The README does not explain what Kineto is, how it works, or its relationship to the PyTorch Profiler and low-level libraries like CUPTI. It serves as a code repository, not a source of documentation. |

---

## 4. New Documentation Coverage

The new documentation directly addresses the gaps identified above. The following table maps each major gap to the new content that fills it.

| Identified Gap | Covered By (New Document) | Key Topics Covered in New Document |
| :--- | :--- | :--- |
| **No Kineto/CUPTI architecture explanation** | `docs/core/02_architecture.md` | - The three-layer architecture: `torch.profiler`, Kineto, and CUPTI.<br>- A clear data flow diagram from user code to trace file.<br>- Explanation of each component's role. |
| **Deprecated TensorBoard tutorial with no Perfetto replacement** | `docs/core/04_visualizing_traces.md` | - Step-by-step guide to loading traces into `ui.perfetto.dev`.<br>- How to navigate the Perfetto UI (zoom, pan, select).<br>- How to interpret key tracks (CPU, GPU, Processes) to find bottlenecks. |
| **Lack of end-to-end optimization example** | `docs/tutorials/01_profiling_to_optimization.py` | - A full case study: starting with a slow model, using the profiler to diagnose a data loading bottleneck, applying a fix (`num_workers`), and verifying the improvement. |
| **No guide on profiling distributed (DDP) jobs** | `docs/tutorials/02_distributed_profiling_hta.py` | - How to correctly set up the profiler in a DDP script to capture traces from all ranks.<br>- How to use the Holistic Trace Analysis (HTA) tool to analyze the multi-rank traces for communication vs. computation overlap. |
| **No guide on profiling custom C++/CUDA operators** | `docs/tutorials/03_custom_operator_profiling.py` | - How to use `torch::autograd::profiler::RecordFunction` in C++ and CUDA code.<br>- A complete example of building and profiling a custom operator extension. |
| **Outdated memory profiling info (`export_memory_timeline`)** | `docs/advanced/01_memory_profiling.md` | - Explanation of why `export_memory_timeline` is deprecated.<br>- A complete guide to the modern workflow using `torch.cuda.memory._snapshot()` and the online memory visualizer. |
| **Thin coverage on profiling `torch.compile`** | `docs/advanced/02_torch_compile_profiling.md` | - How to interpret the `Torch-Compiled Region` event.<br>- A practical workflow combining the profiler with `torch._dynamo.explain()` to diagnose graph breaks and compiler performance. |

---

## 5. Conclusion

The audit reveals that the official PyTorch documentation for performance profiling, while containing some useful introductory material, is fragmented, partially outdated, and suffers from significant gaps in critical areas. Key topics such as the profiler's architecture, the modern trace visualization workflow (Perfetto), distributed profiling, and memory profiling are either undocumented or poorly covered.

The new documentation suite provides a comprehensive and modern resource that rectifies these issues. It is structured for direct contribution to the official PyTorch documentation and tutorials, ensuring that the community has access to the high-quality information needed to effectively diagnose and optimize model performance.

---

## 6. References

[1] PyTorch. (2026). *torch.profiler — PyTorch 2.11 documentation*. Retrieved from https://docs.pytorch.org/docs/stable/profiler.html

[2] PyTorch. (2025). *PyTorch Profiler — PyTorch Tutorials 2.11.0+cu130 documentation*. Retrieved from https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

[3] PyTorch. (2024). *PyTorch Profiler With TensorBoard — PyTorch Tutorials 2.11.0+cu130 documentation*. Retrieved from https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

[4] PyTorch. (2024). *Introduction to Holistic Trace Analysis — PyTorch Tutorials 2.11.0+cu130 documentation*. Retrieved from https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html

[5] PyTorch. (2025). *Profiling your PyTorch Module — PyTorch Tutorials 2.11.0+cu130 documentation*. Retrieved from https://docs.pytorch.org/tutorials/beginner/profiler.html

[6] PyTorch. (2025). *Inductor CPU backend debugging and profiling — PyTorch Tutorials 2.11.0+cu130 documentation*. Retrieved from https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html

[7] PyTorch. (n.d.). *kineto*. GitHub. Retrieved March 25, 2026, from https://github.com/pytorch/kineto

