# Further reading

## 1. Performance measurement

1. Memory bandwidth (sometimes also referred to as memory throughput)
2. Processing throughput

To calculate the memory bandwidth we need to understand two concepts related to it.

**Memory interface width:** It is the physical bit-width of the memory bus. Every clock-cycle data is transferred along a memory bus to and from the on-card memory. The memory interface width is the physical count of the bits of how many can fit down the bus per clock cycle.

**Memory clock rate:** (or memory clock cycle) Measured in the unit of *Hz* it is the rate of memory bus clock. For example a memory bus operating on a 1GHz rate is capable of issuing memory tranfers with the bus 10^9 times per second.

Talking about clock rate it is important to take a detour to *SDR*, *DDR* and *QDR*.

![SDR,QDR,DDR image](https://raw.githubusercontent.com/czangela/cms-gpu-knowledge-transfer/gh-pages/img/d02_img01.png)

???+ example

    A simple analogy is of course public transport, where memory interface width would be the number of seats on a bus and the memory clock rate is the frequency of buses in unit time (hour let's say).
    Multiplying these two numbers we can calculate how many people can be transported along one bus line in a certain time period.

    ![bus image](https://image.freepik.com/free-vector/vector-empty-school-bus-interior-with-blue-seats_33099-2489.jpg)

## 2. Parallel Programming models

Suggested reading: [SIMD < SIMT < SMT: parallelism in NVIDIA GPUs by yosefk](https://yosefk.com/blog/simd-simt-smt-parallelism-in-nvidia-gpus.html)

!!! quote

    NVIDIA call their parallel programming model SIMT - "Single Instruction, Multiple Threads". Two other different, but related parallel programming models are SIMD - "Single Instruction, Multiple Data", and SMT - "Simultaneous Multithreading". Each model exploits a different source of parallelism:

    In SIMD, elements of short vectors are processed in parallel.
    In SMT, instructions of several threads are run in parallel.
    SIMT is somewhere in between – an interesting hybrid between vector processing and hardware threading.

??? tip "Other resources"

    [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

    [https://towardsdatascience.com/the-ai-illustrated-guide-why-are-gpus-so-powerful-99f4ae85a5c3](https://towardsdatascience.com/the-ai-illustrated-guide-why-are-gpus-so-powerful-99f4ae85a5c3)

    [https://streamhpc.com/blog/2016-07-19/performance-can-be-measured-as-throughput-latency-or-processor-utilisation/](https://streamhpc.com/blog/2016-07-19/performance-can-be-measured-as-throughput-latency-or-processor-utilisation/)

    [https://www.gamersnexus.net/dictionary/3-memory-interface](https://www.gamersnexus.net/dictionary/3-memory-interface)
