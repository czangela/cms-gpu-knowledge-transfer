# gpuClustering.h - findClus

Let's look at the histogram filling without the debug messages which we might explain later.

``` cuda
// fill histo
for (int i = first; i < msize; i += blockDim.x) {
  if (id[i] == invalidModuleId)  // skip invalid pixels
    continue;
  hist.count(y[i]);
}
__syncthreads();
if (threadIdx.x < 32)
  ws[threadIdx.x] = 0;  // used by prefix scan...
__syncthreads();
hist.finalize(ws);
__syncthreads();
for (int i = first; i < msize; i += blockDim.x) {
  if (id[i] == invalidModuleId)  // skip invalid pixels
    continue;
  hist.fill(y[i], i - firstPixel);
}
```

!!! tip "Competitive filling"

    We will fill our histogram `hist` with the values `i-firstPixel`. We make sure that each pixel get's in the right bin, corresponding to their column in the module map.

    ``` cuda
    hist.fill(y[i], i - firstPixel);
    ```

    Here as we know, `y` is the column value array of the digis. 
    
    If we iterate through the histogram, pixels in the first column will be processed sooner to pixels in the next column, and so on.

    What we don't know however is what order we are going to process our pixels in one column/bin.
    Filling the histogram is competitive between the threads, the following image illustrates this.

![Iteration order of pixels in cluster](../img/documentation/m0.png)

!!! info "Figure 1 - Order in histogram"

    Not to misunderstand, we don't fill our histogram in this order, this is the iteration order of `cms::cuda::HistoContainer`.

    This iteration order can change, and most probably will change from reconstruction to reconstruction.

We see the relative positions of pixels in our cluster:

![Iteration order of pixels in cluster](../img/documentation/m1.png)

!!! info "Figure 2 - Our HistoContainer"

    Our ordering will be defined as top to bottom inside bins and left to right between bins.

    At least for this example, it doesn't really matter if one imagines the order in one bin the other way around.

![Id/position stored in histogram visualized](../img/documentation/m2.png)

!!! info "Figure 3 - What we store in the HistoContainer"

    On the left, we see how we will later iterate through the histogram.
    
    In the middle and on the right we see what we are actually storing in the histogram.

    We are storing `i-firstPixel`. This is the **relative position of our digis in the digi view/array**.
    
    We don't need all data about digis stored there, just their position in the `digi array` or `digi view`.

    Actually, not even that. We only need their **relative position** and not the **absolute**. That is because all digis belonging to the same module are **consecutive in the digi array**.

    This way we can save precious space, because we would need `32 bits` to store the absolute position of a digi, however, this way we can use only `16 bits`.

    Hmm, `16 bits` means `2^16 = 65536` maximum possible relative positions. 

    How do we know there are no more digis in the module?

    On one hand, it is very unlikely, since in `phase 1` our module dimensions are `80*2*52*8 = 66560`.

    Module occupancy is much lower than `0.98` in any layer, so we're good.

    Still, we're making constraints on the maximum number of digis in the mdodule. 

    Currently, this is 

    ``` cpp
    //6000 max pixels required for HI operations with no measurable impact on pp performance
    constexpr uint32_t maxPixInModule = 6000;
    ```

    We actually don't use this here. We will use this for something else, namely iterating through the digis in the module. Why we will need this will be uncovered soon.

This example only contained one cluster, but in reality we will likely have some consecutive clusters. 

![Iteration order of pixels in cluster](../img/documentation/m3.png)

!!! info "Figure 4 - Multiple clusters in one module"

    Again, in reality our cluster will be more spread out, we are only drawing them this close together for the sake of the example.

    Adding multiple clusters to the game the iteration order will change.

## Nearest neighbours and counting them

A crucial part of the code is the following:

``` cuda
// allocate space for duplicate pixels: a pixel can appear more than once
// with different charge in the same event
constexpr int maxNeighbours = 10;
assert((hist.size() / blockDim.x) <= maxiter);
// nearest neighbour
uint16_t nn[maxiter][maxNeighbours];
uint8_t nnn[maxiter];  // number of nn
for (uint32_t k = 0; k < maxiter; ++k)
  nnn[k] = 0;

__syncthreads();  // for hit filling!
```

!!! tip "nn or nearest neighbours"

    We want to store the relative position of the nearest neighbours of every digi. **Basically, that's why we created the histogram in the first place.** With the histogram the job is half-done, we know the neighbours of every digi columnwise.

    We will create local arrays to store the neighbours.

    We consider neighbours to be **8-connected**.

    ![4-8-connected-image](https://sites.ualberta.ca/~ccwj/teaching/image/morph/Figs/PNG/connectivity.png)

    This would max out the number of possible neighbours to be, well, 8. But as the comment above (and below) explains, due to some read-out inconsistencies or whatnot we can have duplicate digis. So we allocate some extra space for them, but we'll also use some `assertions` later on to make sure we don't exceed our self-made limit.

    ``` cpp
    // allocate space for duplicate pixels: a pixel can appear more than once
    // with different charge in the same event
    constexpr int maxNeighbours = 10;
    ```

!!! warning "How many digis?"

    We finally get to answer why we have an upper limit on the number of digis that we get to check rigorously in our kernel.

    ``` cpp
    //6000 max pixels required for HI operations with no measurable impact on pp performance
    constexpr uint32_t maxPixInModule = 6000;
    ```
    It is connected to this:

    ```cpp
    assert((hist.size() / blockDim.x) <= maxiter);
    ```

    We want to store nearest neighbours for every digi/pixel, but we don't know in advance how many there are. But we do need to fix the number of threads in advance, that is compile time constant.

    `nn` is thread local, so what we will do, is make it two dimensional and let the threads iterate through the module digis, always increasing the `position` by `blockDim.x`, and store nearest neighbours of the next digi in the next row of the `nn` array.

    ``` cpp
    // nearest neighbour
    uint16_t nn[maxiter][maxNeighbours];
    uint8_t nnn[maxiter];  // number of nn
    ```

    For example, if `blockDim.x = 16` and we have `120` digis/hits in our event, then `thread 3` will process the following digis:

    | digi id  |  | nn place |
    |-----|----|-------|
    | 3   | -> | nn[0] |
    | 19  | -> | nn[1] |
    | 35  | -> | nn[2] |
    | 51  | -> | nn[3] |
    | 67  | -> | nn[4] |
    | 83  | -> | nn[5] |
    | 99  | -> | nn[6] |
    | 115 | -> | nn[7] |

    We must decide  (or do we) the size of `nn` in compile time too, so that's why we have `maxiter` and this dangereous looking message:

    ``` cuda
    #ifdef __CUDA_ARCH__
    // assume that we can cover the whole module with up to 16 blockDim.x-wide iterations
    constexpr int maxiter = 16;
    if (threadIdx.x == 0 && (hist.size() / blockDim.x) >= maxiter)
      printf("THIS IS NOT SUPPOSED TO HAPPEN too many hits in module %d: %d for block size %d\n",
             thisModuleId,
             hist.size(),
             blockDim.x);
    #else
        auto maxiter = hist.size();
    #endif
    ```

    It really isn't supposed to happen. Why?
    
    ??? question  "Why? What would happen in our code if this were true?"

        ```cpp
        threadIdx.x == 0 && (hist.size() / blockDim.x) >= maxiter
        ```

        Let's say `hist.size() = 300`.

        Well, then `thread 3` would try to put the nearest neighbours of the following digis in the following non-existing places:

        | digi id  |  | nn place |
        |-----|----|--------|
        | 259 | -> | :skull_and_crossbones: nn[16] :skull_and_crossbones: |
        | 275 | -> | :skull_and_crossbones: nn[17] :skull_and_crossbones: |
        | 291 | -> | :skull_and_crossbones: nn[18] :skull_and_crossbones: |


        We really don't want to have out of bounds indexing errors. It could happen in theory, that's why we run simulations and try to find out are expected (max) occupancy in advance and set **`maxiter`** accordingly.

!!! info "nnn or number of nearest neighbours"

    We will keep track of the number of nearest neighbours as well in a separate array.

    ``` cpp
    uint8_t nnn[maxiter];  // number of nn
    ```

    We could actually get rid of this and follow a different approach, can you find out how?

    ??? question "How?"

        ??? question "No, but really, think about it"

            ??? question "I'm serious"

                Ok, well. Technically, when you create `nn`

                ``` cpp
                // nearest neighbour
                uint16_t nn[maxiter][maxNeighbours];
                ```

                You could do this:

                ``` cpp
                // nearest neighbour
                uint16_t nn[maxiter][maxNeighbours+1];
                ```

                And save one field at the end of each row for a special value e.g. and initialize all values to `numeric_limits<uint16_t>::max()-1` and later only iterate until we reach this value.

                This solution actually uses a bit more space `16 vs 8 bits` and requires us to do some initialization.
## Filling nn

Let's look at how we actually fill our nearest neighbours arrays:

``` cuda
// fill NN
for (auto j = threadIdx.x, k = 0U; j < hist.size(); j += blockDim.x, ++k) {
  assert(k < maxiter);
  auto p = hist.begin() + j;
  auto i = *p + firstPixel;
  assert(id[i] != invalidModuleId);
  assert(id[i] == thisModuleId);  // same module
  int be = Hist::bin(y[i] + 1);
  auto e = hist.end(be);
  ++p;
  assert(0 == nnn[k]);
  for (; p < e; ++p) {
    auto m = (*p) + firstPixel;
    assert(m != i);
    assert(int(y[m]) - int(y[i]) >= 0);
    assert(int(y[m]) - int(y[i]) <= 1);
    if (std::abs(int(x[m]) - int(x[i])) > 1)
      continue;
    auto l = nnn[k]++;
    assert(l < maxNeighbours);
    nn[k][l] = *p;
  }
}
```

!!! tip "Current iteration, keeping track of `k`"

    We will use `k` to keep track of which iteration we are currently in:
    ``` cpp
    for (auto j = threadIdx.x, k = 0U; j < hist.size(); j += blockDim.x, ++k) 
    ```

    It shall not overflow `maxiter`
    ``` cpp
    assert(k < maxiter);
    ```

    There hasn't been any nearest neighbours added to `nn[k]` so our counter of them nnn[k] should be zero, we check this here:

    ``` cpp
    assert(0 == nnn[k]);
    ```

    When we find a neighbour, we add it to `nn[k]`:

    ``` cpp
    auto l = nnn[k]++;
    assert(l < maxNeighbours);
    nn[k][l] = *p;
    ```

    We also check that our index `l` is within bounds.

!!! tip "Pointer to hist element `p`"

    We we look at the `j`th element in the histogram and set the pointer `p` to this element, `i` will be the absolute position of our digi.

    ``` cpp
    auto p = hist.begin() + j;
    auto i = *p + firstPixel;
    ```

    We made sure of these conditions when we created and filled the histogram

    ``` cpp
    assert(id[i] != invalidModuleId);
    assert(id[i] == thisModuleId);  // same module
    ```

    Let's find the pointer to the last element in the next bin, this will be `e`, probably short for `end`.

    Also, we increase `p` by one so we only start considering digis that come after `p`

    ``` cpp
    int be = Hist::bin(y[i] + 1);
    auto e = hist.end(be);
    ++p;
    ```

!!! success "`m`, or possible neighbours"

    Finally we iterate over elements from `p++` until `e`

    ``` cpp
    for (; p < e; ++p) {
      auto m = (*p) + firstPixel;
      assert(m != i);
      ...
    }
    ```

    We know that our column is correct:

    ``` cpp
    assert(int(y[m]) - int(y[i]) >= 0);
    assert(int(y[m]) - int(y[i]) <= 1);
    ```

    So we only need to check whether our row value is `<=1`

    ``` cpp
    if (std::abs(int(x[m]) - int(x[i])) > 1)
      continue;
    ```

    If our row is within bounds, we add `m` to `nn`.

???+ example 

    In this example:

    ![Iteration order of pixels in cluster](../img/documentation/m1.png)

    For `m` we consider the following values

    | i - firstPixel| | *(++p) | `m` | | | | | | 
    |----|----|----|----|----|----|----|----|----|
    | 4  | -> | 1  | 9  | 13 | 12 | 5  | 2  | 16 |
    | 1  | -> | 9  | 13 | 12 | 5  | 2  | 26 |    |
    | 9  | -> | 13 | 12 | 5  | 2  | 16 |    |    |
    | 13 | -> | 12 | 5  | 2  | 16 |    |    |    |

    And for the `nearest neighbours` we get:

    | i - firstPixel| | nn values | |  |
    |----|----|----|----|----|
    | 4  | -> | 9  | 12  | 5 |
    | 1  | -> | 9  | |  |
    | 9  | -> | 12 |  |   |
    | 13 | -> | 2 |    |    |

## Assign same `clusterId` to clusters

Essentially, the following piece of code assigns the same `clusterId` to all pixels/digis in a cluster.

These `clusterId`s won't be ordered, start from `0`, but they will be the same in for neighbouring pixels. They will also be in the range `0` to `numElements`, which is the maximum number of digis for this particular event.

``` cpp
bool more = true;
int nloops = 0;
while (__syncthreads_or(more)) {
    if (1 == nloops % 2) {
    for (auto j = threadIdx.x, k = 0U; j < hist.size(); j += blockDim.x, ++k) {
        auto p = hist.begin() + j;
        auto i = *p + firstPixel;
        auto m = clusterId[i];
        while (m != clusterId[m])
        m = clusterId[m];
        clusterId[i] = m;
    }
    } else {
    more = false;
    for (auto j = threadIdx.x, k = 0U; j < hist.size(); j += blockDim.x, ++k) {
        auto p = hist.begin() + j;
        auto i = *p + firstPixel;
        for (int kk = 0; kk < nnn[k]; ++kk) {
        auto l = nn[k][kk];
        auto m = l + firstPixel;
        assert(m != i);
        auto old = atomicMin_block(&clusterId[m], clusterId[i]);
        // do we need memory fence?
        if (old != clusterId[i]) {
            // end the loop only if no changes were applied
            more = true;
        }
        atomicMin_block(&clusterId[i], old);
        }  // nnloop
    }    // pixel loop
    }
    ++nloops;
}  // end while
```

We also get a bit of explanation, namely

``` cpp
// for each pixel, look at all the pixels until the end of the module;
// when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
// after the loop, all the pixel in each cluster should have the id equeal to the lowest
// pixel in the cluster ( clus[i] == i ).
```

This is all true, but we don't see actually why though.

We need to understand what `nn` and `nnn` and the `hist` is, and how everything is connected, and why our `while` loop is divided into two parts.

So let's dig in.

``` cpp
bool more = true;
int nloops = 0;
```

`more` will be set to true every loop if we updated the `cluterId` for the current pixel, and so it will tell us to terminate our loop or not

`nloops` or number of loops

``` cpp
while (__syncthreads_or(more)) {
```

One can reason intuitavely what this does, or can consult the [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

!!! quote "int __syncthreads_or(int predicate);"

    is identical to __syncthreads() with the additional feature that it evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non-zero for any of them.

So `more`, our local variable is scanned for every thread in the block. We terminate the loop if we didn't update any `cluterId`s in the previous iteration.

