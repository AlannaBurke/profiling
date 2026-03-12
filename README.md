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
