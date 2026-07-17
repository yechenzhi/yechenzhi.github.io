---
layout: series-post
title: "Chapter 1 — From Naive Thread Mapping to Coalesced Global Memory Access"
date: 2026-07-15 12:21:29 +0800
permalink: /gpu-kernels/gemm/classical-cuda-gemm/chapter-1/
series: classical-cuda-gemm
chapter: 1
related_posts: false
---


## Overview
This chapter studies the simplest parallel GEMM algorithm: each CUDA thread computes one output element with a length-$$K$$ dot product.

We compare two kernels that perform exactly the same arithmetic. Their only difference is how CUDA threads are mapped to the rows and columns of the output matrix.

The chapter centers on one question:

> **Why can simply swapping the thread-to-output mapping significantly change performance?**

Because threads execute in warps, this mapping determines the global-memory addresses accessed together. By tracing those addresses, we will see how memory coalescing produces the performance difference.

The two implementations are
[Kernel 13 — Naive Thread Mapping](https://github.com/yechenzhi/SGEMM_CUDA/blob/8a81f85bef33f4592a53376398aa0f37ece148d9/src/kernels/13_my_naive.cuh)
and
[Kernel 14 — Coalesced Global Memory Access](https://github.com/yechenzhi/SGEMM_CUDA/blob/8a81f85bef33f4592a53376398aa0f37ece148d9/src/kernels/14_my_global_mem_coalesce.cuh)
from the [SGEMM_CUDA repository](https://github.com/yechenzhi/SGEMM_CUDA).

---

## GEMM and the Naive CUDA Kernel

General matrix multiplication (GEMM) computes

$$
D = \alpha AB + \beta C,
$$

where

$$
A \in \mathbb{R}^{M \times K},
\qquad
B \in \mathbb{R}^{K \times N},
\qquad
C,D \in \mathbb{R}^{M \times N}.
$$

The scaling factors and the $$C$$ term are not important to this chapter, so we focus on

$$
D=AB.
$$

Each output element is the dot product of one row of $$A$$ and one column of $$B$$:

$$
D_{m,n} = \sum_{k=0}^{K-1} A_{m,k}B_{k,n}.
$$

{% include figure.liquid
  loading="eager"
  path="assets/img/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/figure-1-gemm-output-element.svg"
  class="img-fluid rounded z-depth-1"
  zoomable=true
%}

*Figure 1. Computing one output element $$D_{m,n}$$: row $$m$$ of $$A$$ and column $$n$$ of $$B$$ form a length-$$K$$ dot product.*

The equation defines one independent task for every output coordinate $$(m,n)$$. Since $$D$$ contains $$M\times N$$ elements, the simplest CUDA parallelization launches enough threads to cover the output and assigns one thread to each valid output element.

The baseline implementation used in this chapter is
[Kernel 13](https://github.com/yechenzhi/SGEMM_CUDA/blob/8a81f85bef33f4592a53376398aa0f37ece148d9/src/kernels/13_my_naive.cuh).
It launches a two-dimensional grid of CUDA thread blocks, also called CTAs. Each CTA covers a rectangular region of $$D$$, and each thread within the CTA is responsible for one output coordinate in that region.

The [host-side launch geometry](https://github.com/yechenzhi/SGEMM_CUDA/blob/8a81f85bef33f4592a53376398aa0f37ece148d9/src/runner.cu#L175-L179) is

```cpp
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
dim3 blockDim(32, 32);
```

The block dimensions make each interior CTA cover a $$32\times32$$ tile of $$D$$. The ceiling divisions launch enough CTAs to cover all $$M$$ rows and $$N$$ columns, including partial tiles at the matrix boundaries.

For Kernel 13, the block and thread coordinates are mapped to an output coordinate as

$$
m =
\texttt{blockIdx.x}\cdot\texttt{blockDim.x}
+
\texttt{threadIdx.x},
\qquad
n =
\texttt{blockIdx.y}\cdot\texttt{blockDim.y}
+
\texttt{threadIdx.y}.
$$

The block indices select an output tile of $$D$$, while the thread indices select one element within that tile. Therefore, every valid output element $$D_{m,n}$$ is computed by exactly one thread, and every in-bounds thread computes exactly one output element.

The actual kernel uses a $$32\times32$$ CTA. For clarity, Figure 2 shrinks it to an illustrative $$4\times4$$ CTA while preserving the same indexing structure.

{% include figure.liquid
  loading="lazy"
  path="assets/img/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/figure-2-cta-output-tiling.svg"
  class="img-fluid rounded z-depth-1"
  zoomable=true
%}

*Figure 2. In Kernel 13, blockIdx.x and threadIdx.x select output rows, while blockIdx.y and threadIdx.y select output columns. The enlarged $$4\times4$$ CTA is illustrative.*

If $$M$$ or $$N$$ is not divisible by the block dimensions, an edge CTA may extend beyond $$D$$. Threads for which $$m\ge M$$ or $$n\ge N$$ skip the dot product and the output store.

All matrices are stored in row-major order, so their element offsets are

$$
\operatorname{offset}(A_{m,k}) = mK+k,
\qquad
\operatorname{offset}(B_{k,n}) = kN+n,
\qquad
\operatorname{offset}(D_{m,n}) = mN+n.
$$

The [Kernel 13 device function](https://github.com/yechenzhi/SGEMM_CUDA/blob/8a81f85bef33f4592a53376398aa0f37ece148d9/src/kernels/13_my_naive.cuh#L15-L27) implements this decomposition directly:

```cpp
__global__ void my_sgemm_naive(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *D) {
  const uint row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[row * K + i] * B[i * N + col];
    }
    D[row * N + col] = alpha * tmp + beta * D[row * N + col];
  }
}
```

**The two coordinate calculations for `row` and `col` are the thread-to-output mapping.** The block coordinates select an output tile, and the thread coordinates select one element inside that tile. Together they produce one unique output coordinate for each in-bounds thread: `row` is $$m$$, `col` is $$n$$, and that thread computes $$D_{m,n}$$.

After this mapping, the thread evaluates the entire length-$$K$$ dot product directly from global memory. It performs $$K$$ loads from each input matrix and $$K$$ multiply-add operations, then updates one output element. The kernel does not stage tiles of $$A$$ or $$B$$ in shared memory, so input values are not explicitly reused across threads.

One epilogue detail also matters in the complete access count: the repository uses the old contents of $$D$$ as $$C$$, so with $$\beta=3$$ the expression `beta * D[...]` loads the same element before the final store; this load has the same lane-address pattern as the store and does not change the Kernel 13-versus-14 comparison.

This fully specifies the naive work decomposition: each CTA covers an output tile, and each in-bounds thread computes the length-$$K$$ dot product for one element of that tile.

---

## Warp Formation in Kernel 13

To determine which threads belong to the same warp, CUDA uses a fixed [thread-index linearization rule](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#thread-hierarchy): $$x$$ changes first, then $$y$$, then $$z$$. Here, $$(x,y,z)$$ denotes `threadIdx`, $$(D_x,D_y,D_z)$$ denotes `blockDim`, and $$t$$ denotes the resulting linear thread ID:

$$
t=x+yD_x+zD_xD_y.
$$

The resulting order begins with

$$
(0,0,0),(1,0,0),\ldots,(D_x-1,0,0),
(0,1,0),\ldots,(D_x-1,D_y-1,0),
(0,0,1),\ldots
$$

CUDA [partitions consecutive linear thread IDs into 32-thread warps](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading): IDs 0–31 form the first warp, IDs 32–63 form the second warp, and so on.

Kernel 13 uses $$\texttt{blockDim}=(32,32,1)$$, so $$z=0$$ and

$$
t=x+32y.
$$

For any fixed $$y$$, the 32 threads with $$x=0,\ldots,31$$ have consecutive linear IDs $$32y,\ldots,32y+31$$ and therefore form one warp. Since $$y=0,\ldots,31$$, the $$32\times32$$ CTA contains 32 warps.

{% include figure.liquid
  loading="lazy"
  path="assets/img/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/figure-3-kernel13-warp-partition.svg"
  class="img-fluid rounded z-depth-1"
  zoomable=true
%}

*Figure 3. Warp formation in Kernel 13. Each column has a fixed $$\texttt{threadIdx.y}$$ and contains one 32-thread warp; the $$32\times32$$ thread block contains 32 such warps.*

---

## What Two Neighboring Threads Access

Figure 3 shows that the warp at $$\texttt{threadIdx.y}=w$$ contains the 32 threads with $$\texttt{threadIdx.x}=0,\ldots,31$$. Choose two consecutive threads with $$\texttt{threadIdx.x}=\ell$$ and $$\ell+1$$, where $$0\leq\ell<31$$. Suppose thread $$\ell$$ computes $$D_{m,n}$$. Because $$\texttt{threadIdx.x}$$ maps to the output row while $$\texttt{threadIdx.y}$$ remains fixed within the warp, thread $$\ell+1$$ computes $$D_{m+1,n}$$.

During one loop iteration at a fixed $$k$$:

- thread $$\ell$$ loads $$A_{m,k}$$ and $$B_{k,n}$$;
- thread $$\ell+1$$ loads $$A_{m+1,k}$$ and the same $$B_{k,n}$$.

After all $$K$$ iterations, the two threads load the old values and store the updated values at $$D_{m,n}$$ and $$D_{m+1,n}$$. The same pattern extends across the entire warp: at fixed $$k$$, its 32 threads load 32 elements from column $$k$$ of $$A$$ and all load the same element $$B_{k,n}$$; in the epilogue, they load and store 32 elements in column $$n$$ of $$D$$.

{% include figure.liquid
  loading="lazy"
  path="assets/img/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/figure-4-kernel13-warp-accesses.svg"
  class="img-fluid rounded z-depth-1"
  zoomable=true
%}

*Figure 4. Elements accessed by one Kernel 13 warp. The $$A$$ and $$B$$ highlights show one fixed $$k$$ iteration; the $$D$$ highlights show the epilogue load and store performed after all $$K$$ iterations.*

Because $$A$$, $$B$$, and $$D$$ are stored in row-major order, these matrix positions determine the corresponding address patterns. The highlighted elements in consecutive rows of $$A$$ and $$D$$ are $$K$$ and $$N$$ FP32 values apart, respectively, while all 32 threads request the same element of $$B$$. The next section translates these patterns into 32-byte sector requests.

---

## How the Memory System Serves Those Addresses

Consider an FP32 global-memory load such as `A[row * K + k]`. Although this is one warp instruction, each participating thread computes an address and requests the 4 bytes at that address. The result is a set of per-thread accesses—one for each participating thread—and their addresses may be consecutive, identical, or far apart. A store produces the same kind of set, but with destination addresses. **Coalescing** determines how many aligned memory sectors this warp request must access.

For devices with compute capability 6.0 or later, the [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory) gives a simple counting model: view global memory as a sequence of fixed, aligned 32-byte segments and determine which segments contain the bytes requested by one warp instruction. [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#quantities) calls each aligned 32-byte segment a **sector**. One sector holds eight FP32 values.

For **one** warp load or store, mark every sector touched by at least one participating thread, then count the marked sectors:

> **Number of sectors requested = number of distinct aligned 32-byte sectors touched by one warp instruction.**

This is an instruction-level coalescing model, not a count of final DRAM transactions. A requested sector may hit in L1 or L2, and cache reuse can prevent it from reaching device memory. The sector count therefore explains the efficiency of an address pattern, but it does not by itself predict the exact runtime speedup.

Applying the counting rule to the three address patterns in Figure 5 gives:

- **Consecutive addresses:** 32 aligned consecutive FP32 values span four sectors, so the warp requests four sectors.
- **The same address:** if all 32 threads load one FP32 value, they request one sector. The returned value is supplied to every thread, giving a broadcast-like effect.
- **A large stride:** if every thread's address falls in a different sector, the warp requests 32 sectors.

{% include figure.liquid
  loading="lazy"
  path="assets/img/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/figure-5-global-memory-sectors.svg"
  class="img-fluid rounded z-depth-1"
  zoomable=true
%}

*Figure 5. Three address patterns for one warp memory instruction and the aligned 32-byte sectors they touch.*

Now apply this rule to the full, in-bounds Kernel 13 warp from Figure 4. Let thread $$\ell$$ compute $$D_{m+\ell,n}$$, and assume $$K,N\geq8$$:

| Operation | Address requested by thread $$\ell$$ | Distance between threads | 32-byte sectors requested |
|---|---|---:|---:|
| Load $$A$$ at fixed $$k$$ | $$A_{m+\ell,k}$$ | $$4K$$ bytes | 32 |
| Load $$B$$ at fixed $$k$$ | $$B_{k,n}$$ | 0 bytes | 1 |
| Load old $$D$$ in the epilogue | $$D_{m+\ell,n}$$ | $$4N$$ bytes | 32 |
| Store new $$D$$ in the epilogue | $$D_{m+\ell,n}$$ | $$4N$$ bytes | 32 |

The counts follow directly from row-major storage. Neighboring threads are $$4K$$ bytes apart in $$A$$ and $$4N$$ bytes apart in $$D$$, so for $$K,N\geq8$$ each thread touches a different sector. In contrast, all 32 threads request the same $$B_{k,n}$$ and therefore request only one sector. Because the $$A$$ and $$B$$ loads are separate instructions, one fixed $$k$$ iteration requests $$32+1=33$$ sectors. Across the full dot product and epilogue, the warp requests

$$
33K + 32 + 32 = 33K + 64
$$

sectors: $$33K$$ for the input loads, 32 for the old-$$D$$ load, and 32 for the new-$$D$$ store.

This gives us the principle that drives the rest of the chapter:

> **Thread mapping is memory mapping.**

Kernel 13 performs the correct arithmetic, but a warp varies the row index, producing stride-$$K$$ loads from $$A$$ and stride-$$N$$ accesses to $$D$$.

---

## Kernel 14: Put Adjacent Threads on Adjacent Columns

[Kernel 14](https://github.com/yechenzhi/SGEMM_CUDA/blob/8a81f85bef33f4592a53376398aa0f37ece148d9/src/kernels/14_my_global_mem_coalesce.cuh#L15-L27) keeps the same $$32\times32$$ thread block and the same dot-product loop. It changes only the two lines that map a thread to an output element:

```cpp
const uint col = blockIdx.x * blockDim.x + threadIdx.x;
const uint row = blockIdx.y * blockDim.y + threadIdx.y;
```

The [matching host-side launch geometry](https://github.com/yechenzhi/SGEMM_CUDA/blob/8a81f85bef33f4592a53376398aa0f37ece148d9/src/runner.cu#L190-L194) is

```cpp
dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
dim3 blockDim(32, 32);
```

Here the $$x$$ dimension covers output columns, so `gridDim.x` is determined by $$N$$; the $$y$$ dimension covers output rows, so `gridDim.y` is determined by $$M$$.

The warp formation from Figure 3 does not change: for a fixed `threadIdx.y`, the 32 threads with `threadIdx.x = 0, ..., 31` still form one warp. Only their output coordinates change. For each kernel, let $$D_{m,n}$$ denote the output element computed by the thread with `threadIdx.x = 0` in the selected warp. The thread whose `threadIdx.x` equals $$\ell$$ then computes

$$
\begin{aligned}
\text{Kernel 13:}\quad &D_{m+\ell,n},\\
\text{Kernel 14:}\quad &D_{m,n+\ell}.
\end{aligned}
$$

{% include figure.liquid
  loading="lazy"
  path="assets/img/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/figure-6-thread-mapping-comparison.svg"
  class="img-fluid rounded z-depth-1"
  zoomable=true
%}

*Figure 6. The same `threadIdx.x`-varying warp maps to one column of $$D$$ in Kernel 13 and one row of $$D$$ in Kernel 14. The warp itself has not changed; only the thread-to-output mapping has.*

For one fixed $$k$$ iteration, a Kernel 14 thread computing $$D_{m,n+\ell}$$ loads

$$
A_{m,k}
\qquad\text{and}\qquad
B_{k,n+\ell}.
$$

All 32 threads therefore load the same element of $$A$$ and 32 consecutive elements from one row of $$B$$. In the epilogue, they load the old values and store the updated values for 32 consecutive elements in one row of $$D$$.

{% include figure.liquid
  loading="lazy"
  path="assets/img/gpu-kernels/gemm/classical-cuda-gemm/2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access/figure-7-kernel14-warp-accesses.svg"
  class="img-fluid rounded z-depth-1"
  zoomable=true
%}

*Figure 7. Elements accessed by one Kernel 14 warp. The $$A$$ and $$B$$ highlights show one fixed $$k$$ iteration; the $$D$$ highlight shows the epilogue load and store performed after all $$K$$ iterations.*

---

## Comparing the Two Thread Mappings

Assume a full, in-bounds warp, $$K,N\geq8$$, and a 32-byte-aligned starting address for each consecutive access in Kernel 14. Applying the rule from Figure 5 gives:

| Warp memory instruction | Kernel 13 | Kernel 14 |
|---|---:|---:|
| Load $$A$$ at fixed $$k$$ | 32 sectors: stride $$4K$$ bytes | 1 sector: same address |
| Load $$B$$ at fixed $$k$$ | 1 sector: same address | 4 sectors: 32 consecutive FP32 values |
| Load old $$D$$ in the epilogue | 32 sectors: stride $$4N$$ bytes | 4 sectors: 32 consecutive FP32 values |
| Store new $$D$$ in the epilogue | 32 sectors: stride $$4N$$ bytes | 4 sectors: 32 consecutive FP32 values |

The $$A$$ and $$B$$ loads are separate instructions. At each $$k$$ iteration, their combined count changes from $$32+1=33$$ sectors in Kernel 13 to $$1+4=5$$ in Kernel 14. The epilogue load and store change from $$32+32=64$$ sectors to $$4+4=8$$. Across one complete warp-level dot product, the instruction-level totals are therefore

$$
\begin{aligned}
\text{Kernel 13:}\quad &33K+64,\\
\text{Kernel 14:}\quad &5K+8.
\end{aligned}
$$

Kernel 14 does not reduce every entry in the table: the $$B$$ load rises from one to four sectors because the warp now requests 32 distinct, consecutive $$B$$ values instead of one shared value. The overall pattern is nevertheless much better: the 32-sector patterns for $$A$$ and $$D$$ become one and four sectors, respectively.

### H800 PCIe Performance

The measurements use square matrices, $$\alpha=0.5$$, and $$\beta=3$$, with each result averaged over 50 launches by the [benchmark driver](https://github.com/yechenzhi/SGEMM_CUDA/blob/8a81f85bef33f4592a53376398aa0f37ece148d9/sgemm.cu#L120-L184).

Each of the $$MN$$ output elements performs $$K$$ fused multiply-adds. Counting one FMA as two floating-point operations, the conventional GEMM workload is $$2MNK$$ FLOPs. For an average elapsed time of $$t$$ seconds,

$$
\mathrm{GFLOP/s}
=
\frac{2MNK}{t\times10^9}.
$$

The lower-order $$\alpha/\beta$$ epilogue operations are not included in this conventional count.

| $$M=N=K$$ | Kernel 13 (GFLOP/s) | Kernel 14 (GFLOP/s) | Speedup |
|---:|---:|---:|---:|
| 256 | 211.3 | 1620.3 | 7.67× |
| 512 | 287.8 | 2623.9 | 9.12× |
| 1024 | 385.0 | 3681.7 | 9.56× |
| 2048 | 385.6 | 3733.1 | 9.68× |
| 4096 | 386.8 | 3582.3 | 9.26× |

Kernel 14 is $$7.67\times$$ to $$9.68\times$$ faster across the tested sizes, reaching roughly $$3.6$$–$$3.7$$ TFLOP/s for the three largest matrices.

---

## What This Optimization Does Not Solve

Kernel 14 changes neither the GEMM arithmetic nor the amount of work assigned to each thread. It introduces no shared memory and performs the same length-$$K$$ dot product as Kernel 13. The performance-relevant change is only that `threadIdx.x` now follows the contiguous column dimension of the row-major matrices.

The kernel is still naive because it does not explicitly reuse input data across threads. In particular, the 32 threads in a warp all need the same $$A_{m,k}$$ value, and different warps in the CTA request overlapping data from $$B$$. The next optimization will stage tiles of $$A$$ and $$B$$ in shared memory so that threads can reuse them.

---

## Next Chapter

The next chapter introduces shared-memory tiling:

> **Chapter 2 — Reusing Matrix Tiles with Shared Memory**

A thread block will cooperatively load tiles of $$A$$ and $$B$$ and reuse them across multiple multiply-add operations.
