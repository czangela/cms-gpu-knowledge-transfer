# gpuClustering.h - countModules

The whole kernel:

``` cuda linenums="1" title="countModules kernel"
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

## 1. Init for clustering

Again we have some part of the code here that has nothing to do with counting the modules.

``` cuda linenums="9"
for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
    clusterId[i] = i;
```

We initialise the `clusterId`s for the `findClus` kernel.

## 2. Digi order

Let's say we have a snippet from our `id` array.

Instead of having numbers for the `id` we'll use letters, `A`, `B`, `C` and `D`, and mark `invalid` module ids with ❌.

<table>
    <tr>
        <td>id</td><td>A</td><td>A</td><td>❌</td><td>❌</td><td>A</td><td>B</td><td>❌</td><td>B</td><td>B</td><td>C</td><td>C</td><td>❌</td><td>❌</td><td>D</td>
    </tr>
</table>

!!! warning "Digis ordered by modules"

    It is a prerequisite and we know that digis belonging to one module will appear **consecutive** in our buffer. They might be separated by `invalid` `digis/hits`.

## 3. Look for boundary elements

Let's use our example digi array from the previous point.

In the first row we'll show `id` and in the second column the `threadIdx.x`.

<table>
    <tr>
        <td>id</td><td>A</td><td>A</td><td>❌</td><td>❌</td><td>A</td><td>B</td><td>❌</td><td>B</td><td>B</td><td>C</td><td>C</td><td>❌</td><td>❌</td><td>D</td>
    </tr>
    <tr>
        <td>thid.x</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td>
    </tr>
</table>

Let's execute some of our code:

```cuda linenums="11"
if (invalidModuleId == id[i])
  continue;
auto j = i - 1;
```

<table>
    <tr>
        <td>id</td><td>A</td><td>A</td><td>❌</td><td>❌</td><td>A</td><td>B</td><td>❌</td><td>B</td><td>B</td><td>C</td><td>C</td><td>❌</td><td>❌</td><td>D</td>
    </tr>
    <tr>
        <td>thid.x</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td>
    </tr>
    <tr>
        <td>i</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td>
    </tr>
    <tr>
        <td>j</td><td>-1</td><td>0</td><td>❌</td><td>❌</td><td>3</td><td>4</td><td>❌</td><td>6</td><td>7</td><td>8</td><td>9</td><td>❌</td><td>❌</td><td>12</td>
    </tr>
</table>

``` cuda linenums="14"
while (j >= 0 and id[j] == invalidModuleId)
  --j;
```

<table>
    <tr>
        <td>id</td><td>A</td><td>A</td><td>❌</td><td>❌</td><td>A</td><td>B</td><td>❌</td><td>B</td><td>B</td><td>C</td><td>C</td><td>❌</td><td>❌</td><td>D</td>
    </tr>
    <tr>
        <td>thid.x</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td>
    </tr>
    <tr>
        <td>i</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td>
    </tr>
    <tr>
        <td>j before</td><td>-1</td><td>0</td><td>❌</td><td>❌</td><td>3</td><td>4</td><td>❌</td><td>6</td><td>7</td><td>8</td><td>9</td><td>❌</td><td>❌</td><td>12</td>
    </tr>
    <tr>
        <td>while</td><td></td><td></td><td></td><td></td><td>↓</td><td></td><td></td><td>↓</td><td></td><td></td><td></td><td></td><td></td><td>↓</td>
    </tr>
    <tr>
        <td>j after</td><td>-1</td><td>0</td><td>❌</td><td>❌</td><td>1</td><td>4</td><td>❌</td><td>5</td><td>7</td><td>8</td><td>9</td><td>❌</td><td>❌</td><td>10</td>
    </tr>
</table>

```cuda linenums="16"
if (j < 0 or id[j] != id[i]) {
  // boundary...
  auto loc = atomicInc(moduleStart, nMaxModules);
  moduleStart[loc + 1] = i;
}
```

Let's set `cond = (j < 0 or id[j] != id[i])`. Check when this will be true (`T` is true, `F` is false, ❌ is not evaluated because that thread terminated early):

<table>
    <tr>
        <td>id</td><td>A</td><td>A</td><td>❌</td><td>❌</td><td>A</td><td>B</td><td>❌</td><td>B</td><td>B</td><td>C</td><td>C</td><td>❌</td><td>❌</td><td>D</td>
    </tr>
    <tr>
        <td>thid.x</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td>
    </tr>
    <tr>
        <td>i</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td>
    </tr>
    <tr>
        <td>j after</td><td>-1</td><td>0</td><td>❌</td><td>❌</td><td>1</td><td>4</td><td>❌</td><td>5</td><td>7</td><td>8</td><td>9</td><td>❌</td><td>❌</td><td>10</td>
    </tr>
    <tr>
        <td>cond</td><td>T</td><td>F</td><td>❌</td><td>❌</td><td>F</td><td>T</td><td>❌</td><td>F</td><td>F</td><td>T</td><td>F</td><td>❌</td><td>❌</td><td>T</td>
    </tr>
</table>

Let's just look at the first and last rows and get rid if `False` and not evaluated threads for `cond` to better see what is happening.

<table>
    <tr>
        <td>id</td><td>A</td><td>A</td><td>❌</td><td>❌</td><td>A</td><td>B</td><td>❌</td><td>B</td><td>B</td><td>C</td><td>C</td><td>❌</td><td>❌</td><td>D</td>
    </tr>
    <tr>
        <td>cond</td><td>T</td><td></td><td></td><td></td><td></td><td>T</td><td></td><td></td><td></td><td>T</td><td></td><td></td><td></td><td>T</td>
    </tr>
</table>

## 4. set `moduleStart` for each module

``` cuda linenums="18"
auto loc = atomicInc(moduleStart, nMaxModules);
moduleStart[loc + 1] = i;
```

We fill the `moduleStart` array with starting module indices. Note that we can't make sure that the first module we mark is `A` and then `B`, etc. This code is executed competitively so we might have different `moduleStart` array each execution:

<table>
    <tr>
        <td>pos</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td>
    </tr>
    <tr>
        <td>moduleStart</td><td>4</td><td>0</td><td>5</td><td>9</td><td>13</td><td>0</td><td>0</td>
    </tr>
    <tr>
        <td>moduleStart</td><td>4</td><td>0</td><td>9</td><td>13</td><td>5</td><td>0</td><td>0</td>
    </tr>
    <tr>
        <td>moduleStart</td><td>4</td><td>13</td><td>0</td><td>9</td><td>5</td><td>0</td><td>0</td>
    </tr>
</table>

The order will be determined by in what order each thread reaches the line `18`.

```cuda linenums="18"
auto loc = atomicInc(moduleStart, nMaxModules);
```

## 5. Conclusion

!!! tip "Conclusion"

    We initialise our `clusterId`s for our later clustering algorrithm in `findClus` and fill our `moduleStart` array with the indices of the first `digi/hit` in each module.