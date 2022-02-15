# Week 01 - Exercises

## Exercise_01_01

### Question

!!! question "01. Question: raw to digi conversion kernel"

    * Find the kernel that converts raw data to digis.
    * Where is it launched?
    * What is the execution configuration?
    * How do we access individual threads in the kernel?

    To search the source code use [CMSSW dxr - software cross-reference](https://cmssdt.cern.ch/dxr/CMSSW/source/).

### Solution

??? example "01.a. Find the kernel that converts raw data to digis."

    ```
    // Kernel to perform Raw to Digi conversion
    __global__ void RawToDigi_kernel(const SiPixelROCsStatusAndMapping *cablingMap,
                                    const unsigned char *modToUnp,
                                    const uint32_t wordCounter,
                                    const uint32_t *word,
                                    const uint8_t *fedIds,
                                    uint16_t *xx,
                                    uint16_t *yy,
                                    uint16_t *adc,
                                    uint32_t *pdigi,
    ...
    ```                                
??? example "01.b. Where is it launched?"

    It is launched in the `makeClustersAsync` function:
    ```cuda
    void SiPixelRawToClusterGPUKernel::makeClustersAsync(bool isRun2,
                                                       const SiPixelClusterThresholds clusterThresholds,
                                                       const SiPixelROCsStatusAndMapping *cablingMap,
                                                       const unsigned char *modToUnp,
                                                       const SiPixelGainForHLTonGPU *gains,
                                                       const WordFedAppender &wordFed,
                                                       SiPixelFormatterErrors &&errors,
                                                       const uint32_t wordCounter,
    ...       
    // Launch rawToDigi kernel
      RawToDigi_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
          cablingMap,
          modToUnp,
          wordCounter,
          word_d.get(),
          fedId_d.get(),
          digis_d.view().xx(),
          digis_d.view().yy(),
          digis_d.view().adc(),
    ...
    ```                                            

??? example "01.c. What is the execution configuration?"

    For the `RawToDigi_kernel` he execution configuration is defined as
    ```cuda
    <<<blocks, threadsPerBlock, 0, stream>>>
    ```

    Where 
    ```cuda
    const int threadsPerBlock = 512;
    const int blocks = (wordCounter + threadsPerBlock - 1) / threadsPerBlock;  // fill it all
    ```

    In this case 

    ![blocks](https://latex.codecogs.com/svg.image?blocks=\left&space;\lceil&space;\frac{wordCounter}{threadsPerBlock}&space;\right&space;\rceil)

??? example "01.d. How do we access individual threads in the kernel?"

    ```cuda
    int32_t first = threadIdx.x + blockIdx.x * blockDim.x;
    ```

## Exercise_01_02

### Question

!!! question "02. Question: host and device functions"

    * Give an example of `global`, `device` and `host-device` functions in `CMSSW`.
    
    * Can you find an example where `host` and `device` code diverge? How is this achieved?

### Solution

??? example "02.a. Give an example of `global`, `device` and `host-device` functions in `CMSSW`."

    For example see `__global__` kernel in [previous exercise](weeks/week01_exercises/#solution_1).

    `__device__` function in [RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.cu](https://cmssdt.cern.ch/dxr/CMSSW/source/RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.cu#57):
    ```cuda
    __device__ pixelgpudetails::DetIdGPU getRawId(const SiPixelROCsStatusAndMapping *cablingMap,
                                                    uint8_t fed,
                                                    uint32_t link,
                                                    uint32_t roc) {
        uint32_t index = fed * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
        pixelgpudetails::DetIdGPU detId = {
            cablingMap->rawId[index], cablingMap->rocInDet[index], cablingMap->moduleId[index]};
        return detId;
    }
    ```

    `__host__` `__device__`function in [HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h](https://cmssdt.cern.ch/dxr/CMSSW/source/HeterogeneousCore/CUDAUtilities/interface/OneToManyAssoc.h#191):
    ```cuda
    __host__ __device__ __forceinline__ void add(CountersOnly const &co) {
        for (int32_t i = 0; i < totOnes(); ++i) {
    #ifdef __CUDA_ARCH__
            atomicAdd(off.data() + i, co.off[i]);
    #else
            auto &a = (std::atomic<Counter> &)(off[i]);
            a += co.off[i];
    #endif
        }
    }
    ```

??? example "02.b. Can you find an example where host and device code diverge? How is this achieved?"

    In the [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#host) we can read that:

    The `__device__` and `__host__` execution space specifiers can be used together however, in which case the function is compiled for both the host and the device.
    
    The `__CUDA_ARCH__` macro introduced in Application Compatibility can be used to differentiate code paths between host and device:

    ```cuda
    __host__ __device__ func()
    {
    #if __CUDA_ARCH__ >= 800
    // Device code path for compute capability 8.x
    #elif __CUDA_ARCH__ >= 700
    // Device code path for compute capability 7.x
    #elif __CUDA_ARCH__ >= 600
    // Device code path for compute capability 6.x
    #elif __CUDA_ARCH__ >= 500
    // Device code path for compute capability 5.x
    #elif __CUDA_ARCH__ >= 300
    // Device code path for compute capability 3.x
    #elif !defined(__CUDA_ARCH__) 
    // Host code path
    #endif
    }
    ```

    Based on this we can see how execution diverges in the previous `add` function:
    ```cuda
    __host__ __device__ __forceinline__ void add(CountersOnly const &co) {
        for (int32_t i = 0; i < totOnes(); ++i) {
    #ifdef __CUDA_ARCH__
            atomicAdd(off.data() + i, co.off[i]);
    #else
            auto &a = (std::atomic<Counter> &)(off[i]);
            a += co.off[i];
    #endif
        }
    }
    ```

## Exercise_01_03

### Exercise

!!! question "03. Exercise: Write a kernel in which"

    * if we're running on the `device` each thread prints which `block` and `thread` it is associated with, for example `block 1 thread 3`

    * if we're running on the `host` each thread just prints `host`.

    * Test your program!

??? tip "How can you "hide" your GPU?"

    Try using `CUDA_VISIBLE_DEVICES` from the command line.

## Exercise_01_04

### Exercise

!!! question "04. Exercise: Fine-grained vs coarse-grained parallelism: Give examples in the `MatMulKernel` kernel of **coarse-grained** and **fine-grained data parallelism** (as defined in CUDA abstraction model) as well as sequential execution."

    === "MatMulKernel"

        ```cuda
        // Thread block size
        #define BLOCK_SIZE 16

         __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
        {
            // Block row and column
            int blockRow = blockIdx.y;
            int blockCol = blockIdx.x;

            // Each thread block computes one sub-matrix Csub of C
            Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

            // Each thread computes one element of Csub
            // by accumulating results into Cvalue
            float Cvalue = 0;

            // Thread row and column within Csub
            int row = threadIdx.y;
            int col = threadIdx.x;

            // Loop over all the sub-matrices of A and B that are
            // required to compute Csub
            // Multiply each pair of sub-matrices together
            // and accumulate the results
            for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

                // Get sub-matrix Asub of A
                Matrix Asub = GetSubMatrix(A, blockRow, m);

                // Get sub-matrix Bsub of B
                Matrix Bsub = GetSubMatrix(B, m, blockCol);

                // Shared memory used to store Asub and Bsub respectively
                __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
                __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

                // Load Asub and Bsub from device memory to shared memory
                // Each thread loads one element of each sub-matrix
                As[row][col] = GetElement(Asub, row, col);
                Bs[row][col] = GetElement(Bsub, row, col);

                // Synchronize to make sure the sub-matrices are loaded
                // before starting the computation
                __syncthreads();
                // Multiply Asub and Bsub together
                for (int e = 0; e < BLOCK_SIZE; ++e)
                    Cvalue += As[row][e] * Bs[e][col];

                // Synchronize to make sure that the preceding
                // computation is done before loading two new
                // sub-matrices of A and B in the next iteration
                __syncthreads();
            }

            // Write Csub to device memory
            // Each thread writes one element
            SetElement(Csub, row, col, Cvalue);
        }
        ```

    === "Matrix definition"

        ```cuda
        // Matrices are stored in row-major order:
        // M(row, col) = *(M.elements + row * M.stride + col)
        typedef struct {
            int width;
            int height;
            int stride; 
            float* elements;
        } Matrix;
        ```

    === "GetElement and SetElement"

        ```cuda
        // Get a matrix element
        __device__ float GetElement(const Matrix A, int row, int col)
        {
            return A.elements[row * A.stride + col];
        }

        // Set a matrix element
        __device__ void SetElement(Matrix A, int row, int col,
                                float value)
        {
            A.elements[row * A.stride + col] = value;
        }
        ```
    
    === "GetSubMatrix"

        ```cuda
        // Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
        // located col sub-matrices to the right and row sub-matrices down
        // from the upper-left corner of A
        __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
        {
            Matrix Asub;
            Asub.width    = BLOCK_SIZE;
            Asub.height   = BLOCK_SIZE;
            Asub.stride   = A.stride;
            Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                                + BLOCK_SIZE * col];
            return Asub;
        }
        ```

![Matrix Multiplication with Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/matrix-multiplication-with-shared-memory.png){ width=900 }

### Solution

??? example "04.a. Give examples in the `MatMulKernel` kernel of **coarse-grained data parallelism**."

    Coarse-grained data parallel problems in the CUDA programming model are problems that can be solved independently in parallel by blocks of threads.

    For example in the `MatMulKernel`:

    ```cuda
     // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    ```

    The computation of `Csub` is independent of the computation of other submatrices of `C`. The work is divided between `blocks`, no synchronization is performed between computing different `submatrices of C`.


??? example "04.b. Give examples in the `MatMulKernel` kernel of **fine-grained data parallelism**."

    Fine-grained data parallel problems in the CUDA programming model are finer pieces that can be solved cooperatively in parallel by all threads within the block.

    For example in the `MatMulKernel`:

    ```cuda
    // Get sub-matrix Asub of A
    Matrix Asub = GetSubMatrix(A, blockRow, m);

    // Get sub-matrix Bsub of B
    Matrix Bsub = GetSubMatrix(B, m, blockCol);

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();
    ```

    Loading data into shared memory blocks of `matrix A and B` is executed parellel by all threads within the block.

??? example "04.c. Give examples in the `MatMulKernel` kernel of **sequential execution**."

    ```cuda
    // Multiply Asub and Bsub together
    for (int e = 0; e < BLOCK_SIZE; ++e)
        Cvalue += As[row][e] * Bs[e][col];
    ```

    The computation of `Cvalue` for each thread is sequential, we execute `BLOCK_SIZE` additions and multiplications.

    On the other hand the computation of `Cvalue` is also a good example of *fine-grained* data parallelism, since there is one value computed by each thread in the block parallel.

    To identify *fine-grained* parallelism one just needs to look for block-level synchronization:

    ```cuda
    for (int e = 0; e < BLOCK_SIZE; ++e)
        Cvalue += As[row][e] * Bs[e][col];

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
    ```
