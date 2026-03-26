# PyTorch Documentation Placement Analysis

**Date:** March 26, 2026
**Author:** Manus AI

## 1. Introduction

Following the creation of a comprehensive documentation suite for the PyTorch Profiler and Kineto, this report provides a detailed analysis of the official PyTorch documentation ecosystem and proposes a strategic placement plan for each new piece of content. The goal is to integrate the new documentation seamlessly and logically into the existing structure of `pytorch.org` and `pytorch.org/tutorials`, ensuring maximum visibility and utility for the community.

This analysis is based on a thorough audit of the `pytorch/pytorch` and `pytorch/tutorials` repositories to understand their structure, conventions, and content categories.

---

## 2. Analysis of the PyTorch Documentation Ecosystem

The PyTorch documentation is primarily split across two different sites and repositories, each with a distinct purpose.

### 2.1. The Main API Documentation (`pytorch.org/docs`)

*   **Repository:** `pytorch/pytorch`
*   **Source Directory:** `docs/source/`
*   **Content:** This site hosts the core API reference documentation. It is built from a combination of reStructuredText (`.rst`) and Markdown (`.md`) files. Docstrings from the Python source code are automatically pulled into these files to generate the final API pages.
*   **Key Subdirectories:**
    *   `docs/source/`: Contains the `.rst` and `.md` files for each module (e.g., `profiler.md`, `autograd.md`). These files define the structure of the page and use Sphinx directives (`.. autoclass::`, `.. automodule::`) to pull in content.
    *   `docs/source/notes/`: A dedicated section for deep-dive, conceptual explanations of PyTorch internals and advanced features. These are aimed at developers who want to understand *how* something works, not just *what* it does. Examples include `autograd.rst` and `cuda.rst`.

### 2.2. The Tutorials Site (`pytorch.org/tutorials`)

*   **Repository:** `pytorch/tutorials`
*   **Source Directories:** `beginner_source/`, `intermediate_source/`, `advanced_source/`, `recipes_source/`
*   **Content:** This site hosts hands-on, example-driven tutorials and recipes. Tutorials are written as Python scripts (`.py`) that are automatically converted into HTML pages and runnable Jupyter notebooks by Sphinx-Gallery.
*   **Key Subdirectories:**
    *   `{level}_source/`: Tutorials are categorized by difficulty. They are expected to be longer, narrative-style guides that walk a user through a complete example.
    *   `recipes_source/`: Contains 
shorter, more focused, and task-oriented guides. They provide a quick answer to a specific "how-to" question (e.g., "How do I use the profiler?").

---

## 3. Content Placement Strategy

Based on this ecosystem structure, a two-pronged strategy is recommended. The new content should be split between the `pytorch/pytorch` and `pytorch/tutorials` repositories to align with their respective content models.

1.  **Core Concepts & API Reference** will go into `pytorch/pytorch`.
2.  **Hands-On Tutorials & Recipes** will go into `pytorch/tutorials`.

This ensures that users find information where they expect it: API details and deep-dive explanations on the main docs site, and practical, runnable examples on the tutorials site.

### 3.1. Detailed Placement Map

The following table provides a precise file-by-file placement plan for every new piece of documentation.

| New Content File (from `AlannaBurke/profiling`) | Target Repository | Proposed Location & Filename | Content Type | Justification & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `docs/core/01_introduction.md` | `pytorch/pytorch` | `docs/source/profiler.md` (Prepend) | API Docs | This content should be prepended to the existing `profiler.md` to serve as the high-level **Overview** section, which is currently missing. |
| `docs/core/02_architecture.md` | `pytorch/pytorch` | `docs/source/notes/profiler_architecture.rst` | Developer Note | This is a deep-dive, conceptual explanation of the profiler's internal workings (Kineto, CUPTI). The `notes/` directory is the perfect home for this, alongside other architectural explainers. |
| `docs/core/03_getting_started.md` | `pytorch/tutorials` | `recipes_source/recipes/profiler_recipe.py` | Recipe | This content should **replace** the existing `profiler_recipe.py`. It serves the same purpose (a quick start) but is more comprehensive and up-to-date. |
| `docs/core/04_visualizing_traces.md` | `pytorch/tutorials` | `intermediate_source/perfetto_trace_viz_tutorial.py` | Tutorial | This is a brand-new, step-by-step tutorial that fills the critical gap left by the deprecated TensorBoard tutorial. It's a narrative guide, making it a perfect fit for the `intermediate_source` category. |
| `docs/core/05_advanced_techniques.md` | `pytorch/pytorch` | `docs/source/profiler.md` (Append) | API Docs | This content details advanced API features like scheduling and memory profiling. It should be appended to the main `profiler.md` page to enrich the API reference documentation. |
| `docs/tutorials/01_profiling_to_optimization.py` | `pytorch/tutorials` | `intermediate_source/profiling_optimization_case_study.py` | Tutorial | This is a classic, end-to-end story of diagnosing and fixing a bottleneck. Its narrative structure makes it an ideal intermediate tutorial. |
| `docs/tutorials/02_distributed_profiling_hta.py` | `pytorch/tutorials` | `advanced_source/distributed_profiling_hta_tutorial.py` | Tutorial | Profiling distributed jobs is an advanced topic. This tutorial belongs in `advanced_source` and complements the existing HTA tutorial by showing how to *generate* the traces. |
| `docs/tutorials/03_custom_operator_profiling.py` | `pytorch/tutorials` | `advanced_source/custom_operator_profiling_tutorial.py` | Tutorial | Extending PyTorch with custom C++/CUDA operators and profiling them is an advanced developer task. This fits perfectly in the `advanced_source` directory. |
| `docs/advanced/01_memory_profiling.md` | `pytorch/pytorch` | `docs/source/notes/memory_profiling.rst` | Developer Note | This provides a deep dive into the modern memory snapshot APIs, replacing the deprecated `export_memory_timeline`. Its conceptual depth makes it a great fit for the `notes/` section. |
| `docs/advanced/02_torch_compile_profiling.md` | `pytorch/pytorch` | `docs/source/notes/torch_compile_profiling.rst` | Developer Note | This expands significantly on the brief mention in the existing Inductor tutorial. It provides a conceptual guide to interpreting compiled regions and graph breaks, making it an ideal developer note. |

---

## 4. Conclusion

By strategically placing the new documentation across the `pytorch/pytorch` and `pytorch/tutorials` repositories, the content will be discoverable, consistent with the existing information architecture, and maximally beneficial to the PyTorch community. The core API documentation will be enriched with architectural details and advanced usage, while the tutorials site will gain modern, in-depth, and practical examples that cover critical, previously undocumented workflows.

