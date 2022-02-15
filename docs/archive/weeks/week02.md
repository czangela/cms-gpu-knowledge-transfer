# Week 02 - 2021.12.13-17.

???+ tip "Resources"

    * [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

    * [Introduction to GPUs by New York University](https://nyu-cds.github.io/python-gpu/02-cuda/)

    * [NVidia Developer Blog: Using Shared Memory CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

In the previous material in [A Scalable Programming Model](./week01/#2-a-scalable-programming-model) we've been reading about the three key abstractions in the CUDA programming model:

!!! quote ""

    * a hierarchy of thread groups
    * shared memories
    * barrier synchronization

[Thread hierarchy](./week01/#4-thread-hierarchy) has been previously covered, and in this part **shared memory** and **barrier synchronization** follows.

## 1. Shared memory

### Memory hierarchy

CUDA threads may access data from multiple memory spaces during their execution.

Each thread has private **local memory**.

Each thread block has **shared memory** visible to all threads of the block and with the same lifetime as the block.

All threads have access to the same **global memory**.

![memory hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/memory-hierarchy.png){ width=800 }

There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces, which we won't cover in detail here. [For more information continue reading here.](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)

The global, constant, and texture memory spaces are persistent across kernel launches by the same application.

### `__shared__`

As detailed in Variable Memory Space Specifiers shared memory is allocated using the `__shared__`    memory space specifier.

Shared memory is expected to be much faster than global memory.

### Static and dynamic

Example from: [https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/shared-memory/shared-memory.cu](https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/shared-memory/shared-memory.cu)

!!! example "Usage of static and dynamic memory"

    === "copyright"

        ```cuda
        /* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
        *
        * Redistribution and use in source and binary forms, with or without
        * modification, are permitted provided that the following conditions
        * are met:
        *  * Redistributions of source code must retain the above copyright
        *    notice, this list of conditions and the following disclaimer.
        *  * Redistributions in binary form must reproduce the above copyright
        *    notice, this list of conditions and the following disclaimer in the
        *    documentation and/or other materials provided with the distribution.
        *  * Neither the name of NVIDIA CORPORATION nor the names of its
        *    contributors may be used to endorse or promote products derived
        *    from this software without specific prior written permission.
        *
        * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
        * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
        * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
        * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
        * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
        * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
        * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        */
        ```

    === "staticReverse"

        ```cuda
        #include <stdio.h>

        __global__ void staticReverse(int *d, int n)
        {
          __shared__ int s[64];
          int t = threadIdx.x;
          int tr = n-t-1;
          s[t] = d[t];
          __syncthreads();
          d[t] = s[tr];
        }
        ```

    === "dynamicReverse"

        ```cuda
        __global__ void dynamicReverse(int *d, int n)
        {
          extern __shared__ int s[];
          int t = threadIdx.x;
          int tr = n-t-1;
          s[t] = d[t];
          __syncthreads();
          d[t] = s[tr];
        }
        ```

    === "main"

        ```cuda
        int main(void)
        {
          const int n = 64;
          int a[n], r[n], d[n];
          
          for (int i = 0; i < n; i++) {
            a[i] = i;
            r[i] = n-i-1;
            d[i] = 0;
          }

          int *d_d;
          cudaMalloc(&d_d, n * sizeof(int)); 
          
          // run version with static shared memory
          cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
          staticReverse<<<1,n>>>(d_d, n);
          cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
          for (int i = 0; i < n; i++) 
            if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
          
          // run dynamic shared memory version
          cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
          dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
          cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
          for (int i = 0; i < n; i++) 
            if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
        }
        ```

If the shared memory array size is known at **compile time**, as in the staticReverse kernel, then we can explicitly declare an array of that size, as we do with the array `s`.

```cuda
__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}
```

The other three kernels in this example use dynamically allocated shared memory, which can be used when the amount of shared memory is **not known at compile time**. In this case the shared memory allocation size per thread block must be specified (in bytes) using an optional third execution configuration parameter, as in the following excerpt.

```cuda
dynamicReverse<<<1, n, n*sizeof(int)>>>(d_d, n);
```

The dynamic shared memory kernel, dynamicReverse(), declares the shared memory array using an unsized extern array syntax, `extern __shared__ int s[]`.

```cuda
__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}
```

## 2. Barrier synchronization

### Introduction

<div class="video-wrapper">
  <iframe width="800" height="450" src="https://www.youtube.com/embed/OjWij5-L0AA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

When sharing data between threads, we need to be careful to avoid race conditions, because while threads in a block run logically in parallel, not all threads can execute physically at the same time.

### Synchronize block of threads

Let’s say that two threads A and B each load a data element from global memory and store it to shared memory. Then, thread A wants to read B’s element from shared memory, and vice versa. Let’s assume that A and B are threads in two different warps. If B has not finished writing its element before A tries to read it, we have a race condition, which can lead to undefined behavior and incorrect results.

To ensure correct results when parallel threads cooperate, we must synchronize the threads. CUDA provides a simple barrier synchronization primitive, `__syncthreads()`. A thread’s execution can only proceed past a `__syncthreads()` after all threads in its block have executed the `__syncthreads()`. Thus, we can avoid the race condition described above by calling `__syncthreads()` after the store to shared memory and before any threads load from shared memory.

### Deadlocks

!!! example "barrier synchronization"

    === "deadlock"

        ```cuda
        int s = threadIdx.x / 2;
        if (threadIdx.x > s) {
            value[threadIdx.x] = 2*s;
            __syncthreads();
        }
        else {
            value[threadIdx.x] = s;
            __syncthreads();
        }
        ```

    === "correct barrier"

        ```cuda
        int s = threadIdx.x / 2;
        if (threadIdx.x > s) {
            value[threadIdx.x] = 2*s;
        }
        else {
            value[threadIdx.x] = s;
        }
        __syncthreads();
        ```


It’s important to be aware that calling `__syncthreads()` in divergent code is undefined and can lead to deadlock—all threads within a thread block must call `__syncthreads()` at the same point.

## 3. Page-Locked Host Memory

!!! quote "From the CUDA Programming Guide"

    The runtime provides functions to allow the use of page-locked (also known as pinned) host memory (as opposed to regular pageable host memory allocated by malloc()):

    * cudaHostAlloc() and cudaFreeHost() allocate and free page-locked host memory;
    * cudaHostRegister() page-locks a range of memory allocated by malloc() (see reference manual for limitations).

    Using page-locked host memory has several benefits:

    * Copies between page-locked host memory and device memory can be performed concurrently with kernel execution for some devices as mentioned in Asynchronous Concurrent Execution.
    * On some devices, page-locked host memory can be mapped into the address space of the device, eliminating the need to copy it to or from device memory as detailed in Mapped Memory.
    * On systems with a front-side bus, bandwidth between host memory and device memory is higher if host memory is allocated as page-locked and even higher if in addition it is allocated as write-combining as described in Write-Combining Memory.

Explaining *pinned memory*:

<div class="video-wrapper">
  <iframe width="800" height="450" src="https://www.youtube.com/embed/ShT7raBPP8k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

Setting flags for `cudaHostAlloc`:

<div class="video-wrapper">
  <iframe width="800" height="450" src="https://www.youtube.com/embed/7hk45jtc72k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
