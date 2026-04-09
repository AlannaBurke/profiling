# PyTorch Profiler Documentation

This repository contains a complete set of open-source documentation and tutorials for the PyTorch Profiler. This content was created to fill critical gaps in the official PyTorch documentation and is intended for contribution to the `pytorch/tutorials` and `pytorch/pytorch` repositories.

## Repository Structure

The repository is organized as follows:

```
.
├── README.md
├── documentation_placement_analysis.md
├── documentation_audit_report.md
└── docs/
    ├── core/
    │   ├── 01_introduction.md
    │   ├── 03_getting_started.md
    │   ├── 04_visualizing_traces.md
    │   └── 05_advanced_techniques.md
    └── tutorials/
        └── 01_profiling_to_optimization.py
```

### Root Directory

*   `README.md`: This file, providing an overview of the repository.
*   `documentation_placement_analysis.md`: Analysis of where these files should live in the PyTorch repos.
*   `documentation_audit_report.md`: Audit of the existing documentation.

### `docs/` Directory

This directory contains the modular documentation files, organized by topic.

*   **`core/`**: Contains the core documentation for the PyTorch Profiler. These files are intended to be contributed to the main PyTorch documentation.
*   **`tutorials/`**: Contains in-depth, executable tutorials in the form of Python scripts. These are formatted for use with sphinx-gallery and are intended for the `pytorch/tutorials` repository.

## Content Overview

The documentation has been specifically tailored to focus on the current public API (`torch.profiler.profile`), utilizing Perfetto for visualization, and addressing common model execution bottlenecks. It explicitly avoids deprecated tools like TensorBoard and legacy profilers.
