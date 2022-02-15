# gpuClustering.h - countModules

Our kernel function is the following:

``` cuda
  template <bool isPhase2>
  __global__ void countModules(uint16_t const* __restrict__ id,
                               uint32_t* __restrict__ moduleStart,
                               int32_t* __restrict__ clusterId,
                               int numElements) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int nMaxModules = isPhase2 ? phase2PixelTopology::numberOfModules : phase1PixelTopology::numberOfModules;
    assert(nMaxModules < maxNumModules);
    for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
      clusterId[i] = i;
      if (invalidModuleId == id[i])
        continue;
      auto j = i - 1;
      while (j >= 0 and id[j] == invalidModuleId)
        --j;
      if (j < 0 or id[j] != id[i]) {
        // boundary...
        auto loc = atomicInc(moduleStart, nMaxModules);
        moduleStart[loc + 1] = i;
      }
    }
  }
```