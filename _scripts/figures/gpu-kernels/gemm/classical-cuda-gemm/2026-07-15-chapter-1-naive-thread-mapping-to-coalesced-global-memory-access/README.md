# Figure generators

This directory contains deterministic figure generators for
`_posts/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access.md`.

Generate the figures from the repository root:

```bash
python3 _scripts/figures/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/generate_figure_1_gemm_output_element.py
python3 _scripts/figures/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/generate_figure_2_cta_output_tiling.py
python3 _scripts/figures/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/generate_figure_3_kernel13_warp_partition.py
python3 _scripts/figures/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/generate_figure_4_kernel13_warp_accesses.py
python3 _scripts/figures/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/generate_figure_5_global_memory_sectors.py
python3 _scripts/figures/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/generate_figure_6_thread_mapping_comparison.py
python3 _scripts/figures/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/generate_figure_7_kernel14_warp_accesses.py
```

The scripts use only the Python standard library and write the SVGs to the matching post directory under `assets/img/`.
