# Week 01 - 2021.12.06-10.

???+ info "Overview"

    * What is the CUDA programming model?
    * Hierarchy of thread groups
    * Kernels and other language extensions

???+ tip "Resources"

    This material heavily borrows from the following sources:

    * [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

    * [Introduction to GPUs by New York University](https://nyu-cds.github.io/python-gpu/02-cuda/)

    * [Introduction to parallel programming and CUDA by Felice Pantaleo](https://indico.cern.ch/event/863657/)

    
## Introduction

### 1. CUDA®: A General-Purpose Parallel Computing Platform and Programming Model

In November 2006, NVIDIA®  introduced CUDA® , which originally stood for “Compute Unified Device Architecture”, a general purpose parallel computing platform and programming model that leverages the parallel compute engine in NVIDIA GPUs to solve many complex computational problems in a more efficient way than on a CPU.

CUDA comes with a software environment that allows developers to use C++ as a high-level programming language. Other languages, application programming interfaces, or directives-based approaches are supported, such as FORTRAN, DirectCompute, OpenACC.

![GPU Computing Applications](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/gpu-computing-applications.png)

### 2. A Scalable Programming Model

???+ quote "At the core of the  CUDA parallel programming model there are three key abstractions:"

    * a hierarchy of thread groups
    * shared memories
    * barrier synchronization

They are exposed to the programmer as a **minimal set of language extensions**.

These abstractions provide **fine-grained data parallelism and thread parallelism**, nested within **coarse-grained data parallelism and task parallelism**.

!!! tip "Further reading and material" 

    Optional reading and exercise on this topic at [abstractions: granularity](./week01/#abstractions-granularity).

## Programming model

### 3. Kernels

CUDA C++ extends C++ by allowing the programmer to define C++ functions, called *kernels*, that, when called, are executed `N` times in parallel by `N` different CUDA threads, as opposed to only once like regular C++ functions.

A kernel is defined using the `__global__` declaration specifier and the number of CUDA threads that execute that kernel for a given kernel call is specified using a new `<<<...>>>` execution configuration syntax.

Each thread that executes the kernel is given a *unique thread ID* that is accessible within the kernel through built-in variables.

???+ example "Kernel and execution configuration example"

    ```cuda
    // Kernel definition
    __global__ void VecAdd(float* A, float* B, float* C)
    {
        int i = threadIdx.x;
        C[i] = A[i] + B[i];
    }

    int main()
    {
        ...
        // Kernel invocation with N threads
        VecAdd<<<1, N>>>(A, B, C);
        ...
    }
    ```

!!! question "01. Question: raw to digi conversion kernel"

    * Find the kernel that converts raw data to digis.
    * Where is it launched?
    * What is the execution configuration?
    * How do we access individual threads in the kernel?

    [Go to exercise 01_01](./week01_exercises.md#exercise_01_01)

### 4. Thread hierarchy

A kernel is executed in parallel by an array of threads:

* All threads run the same code.
* Each thread has an ID that it uses to compute memory addresses and make control decisions.

![thread_img_01](https://nyu-cds.github.io/python-gpu/fig/02-threadblocks.png){ width=300 }

Threads are arranged as a grid of thread blocks:

* Different kernels can have different grid/block configuration
* Threads from the same block have access to a shared memory and their execution can be synchronized

![thread_img_02](https://nyu-cds.github.io/python-gpu/fig/02-threadgrid.png)

Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series.

This independence requirement allows thread blocks to be scheduled in any order across any number of cores, enabling programmers to write code that scales with the number of cores.

Threads within a block can cooperate by sharing data through some shared memory and by synchronizing their execution to coordinate memory accesses.

The grid of blocks and the thread blocks can be 1, 2, or 3-dimensional.

![thread_img_03](https://nyu-cds.github.io/python-gpu/fig/02-threadmapping.png)

The CUDA architecture is built around a scalable array of multithreaded **Streaming Multiprocessors** (SMs) as shown below.

Each SM has a set of execution units, a set of registers and a chunk of shared memory.

![sm_img_01](https://nyu-cds.github.io/python-gpu/fig/02-sm.png)

### 5. Language extensions

[From CUDA Toolkit Documentation: Language Extensions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions):
#### `__global__`

The __global__ execution space specifier declares a function as being a kernel. Such a function is:

* Executed on the device,
* Callable from the host,
* Callable from the device for devices of compute capability 3.2 or higher (see CUDA Dynamic Parallelism for more details).
A __global__ function must have void return type, and cannot be a member of a class.

Any call to a __global__ function must specify its execution configuration as described in [Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration).

A call to a __global__ function is asynchronous, meaning it returns before the device has completed its execution.

#### `__device__`

The __device__ execution space specifier declares a function that is:

* Executed on the device,
* Callable from the device only.

The __global__ and __device__ execution space specifiers cannot be used together.

#### `__host__`

The __host__ execution space specifier declares a function that is:

* Executed on the host,
* Callable from the host only.

It is equivalent to declare a function with only the __host__ execution space specifier or to declare it without any of the __host__, __device__, or __global__ execution space specifier; in either case the function is compiled for the host only.

The __global__ and __host__ execution space specifiers cannot be used together.

The __device__ and __host__ execution space specifiers can be used together however, in which case the function is compiled for both the host and the device.

!!! question "02. Question: host and device functions"

    * Give an example of `global`, `device` and `host-device` functions in `CMSSW`.
    
    * Can you find an example where `host` and `device` code diverge? How is this achieved?

    [Go to exercise 01_02](./week01_exercises.md#exercise_01_02)

!!! question "03. Exercise: Write a kernel in which"

    * if we're running on the `device` each thread prints which `block` and `thread` it is associated with, for example `block 1 thread 3`

    * if we're running on the `host` each thread just prints `host`.

    [Go to exercise 01_03](./week01_exercises.md#exercise_01_03)

### 6. Execution Configuration

[From CUDA Toolkit Documentation: Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)

Any call to a __global__ function must specify the execution configuration for that call. The execution configuration defines the dimension of the grid and blocks that will be used to execute the function on the device, as well as the associated stream (see CUDA Runtime for a description of streams).

The execution configuration is specified by inserting an expression of the form `<<< Dg, Db, Ns, S >>>` between the function name and the parenthesized argument list, where:

* `Dg` is of type dim3 (see dim3) and specifies the dimension and size of the grid, such that `Dg.x * Dg.y * Dg.z` equals the number of blocks being launched;

* `Db` is of type dim3 (see dim3) and specifies the dimension and size of each block, such that `Db.x * Db.y * Db.z` equals the number of threads per block;

* `Ns` is of type `size_t` and specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory; this dynamically allocated memory is used by any of the variables declared as an external array as mentioned in __shared__; Ns is an optional argument which defaults to 0;

* `S` is of type `cudaStream_t` and specifies the associated stream; S is an optional argument which defaults to 0.

## Abstractions: Granularity 

???+ abstract "Granularity"

    If ![T_comp](https://latex.codecogs.com/svg.image?T_{comp}) is the computation time and ![T_comm](https://latex.codecogs.com/svg.image?T_{comm}) denotes the communication time, then the Granularity G of a task can be calculated as

    ![T_comp](https://latex.codecogs.com/svg.image?G=\frac{T_{comp}}{T_{comm}})

    Granularity is usually measured in terms of the number of instructions executed in a particular task.

???+ abstract "Fine-grained parallelism"

    Fine-grained parallelism means individual tasks are relatively small in terms of code size and execution time. The data is transferred among processors frequently in amounts of one or a few memory words.

???+ abstract "Coarse-grained parallelism"

    Coarse-grained is the opposite in the sense that data is communicated infrequently, after larger amounts of computation.

![Fine-course grained parallelism](https://raw.githubusercontent.com/czangela/cms-gpu-knowledge-transfer/gh-pages/img/w01_img02.png)

The CUDA abstractions provide fine-grained data parallelism and thread parallelism, nested within coarse-grained data parallelism and task parallelism. They guide the programmer to partition the problem into **coarse sub-problems** that can be solved independently in parallel by blocks of threads, and each sub-problem into **finer pieces that can be solved cooperatively in parallel by all threads within the block**.

This decomposition preserves language expressivity by allowing threads to cooperate when solving each sub-problem, and at the same time enables automatic scalability.

Indeed, each block of threads can be scheduled on any of the available multiprocessors within a GPU, in any order, concurrently or sequentially, so that a compiled CUDA program can execute on any number of multiprocessors.

![CUDA Automatic Scalability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/automatic-scalability.png){ width=800 }

!!! warning "The following exercise requires knowledge about *barrier synchronization* and *shared memory*."

    Follow-up on `__syncthreads()` and [shared memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory).

!!! question "04. Exercise: Fine-grained vs coarse-grained parallelism"

    * Give examples in the `MatMulKernel` kernel of **coarse-grained** and **fine-grained data parallelism** (as defined in CUDA abstraction model) as well as sequential execution.

    [Go to exercise 01_04](./week01_exercises.md#exercise_01_04)