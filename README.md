CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Avi Serebrenik
  * [https://www.linkedin.com/in/avi-serebrenik-a7386426b/](), [https://aviserebrenik.wixsite.com/cvsite]()
* Tested on: Windows 11, i7-13620H @ 2.4GHz 32GB, RTX 4070 8GB (Laptop)

# Overview
This project implements and compares different scanning (adding up values in an array until that index) and stream compaction (removing 0s from an array in this case) algorithms. They are implemented on the GPU with CUDA, with a naive CPU version for performance comparison. The algorithms are the following:

## 1. CPU Scan and Compaction
 - *Overview:* These serve as benchmarks for the algorithms and follow a naive implementation of scanning and stream compaction.
 - *Implementation:* The scanning algorithm simply loops over the entire input array and sets the output array to be the last index + the last index of the input array. This is O(n). Compaction has two different algorithms, one using the scan function above and one without it. For the one without, I again simply loop over the input array, and if the value is not 0, I add it to an incremented index of the output array. This incremented index serves to then signal the size of the final array. For the method with scanning, I first create a temporary boolean array storing whether the input array value is non 0. I then scan this boolean array, with the final value giving me the size of the output array needed. Finally, I loop over the input array and if my bool array at that index is 1, then I check the scan array's value for what index of the output array this input value should be stored at (i.e. if boolarray(index): out_data(scanArray(index)) = in_data(index)). Both of these are also O(n).

## 2. Naive GPU Scan
 - *Overview:* This is a truly "naive" parallelized scan algorithm that I purposefully didn't optimize more to have as a good basic benchmark.
 - *Implementation:* The algorithm runs a for loop from i = 1 to i < ilog2ceil(n) -- padding non multiples of 2 arrays with 0s. In this for loop, I launch a kernel that simply sums based on a power of 2. Meaning, it sums pairs of adjacent values on the first loop, then sums pairs of those resulting sums, and so on. This results in O(log(n)).

## 3. Work Efficient GPU Scan
 - *Overview:* This is an in-place scanning algorithm that works in two parts, an "Up Sweep" and a "Down Sweep." This works by treating the array as a balanced binary tree, which we can think of creating with the up sweep and fully balancing and ordering with the down sweep. For a detailed explanation, I would recommend this [guide by Nvidia](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).
 - *Implementation:* To optimize this algorithm, I forgo using a modulo operator and instead use half as many blocks in each loop and multiply thread indices appropriately, so adjacent threads do more similar work. This is also O(log(n)), but for a more precise comparison, please check the results portion below.

## 4. Work Efficient GPU Compaction
 - *Overview:* This uses the same algorithm as the CPU compaction with scan, using the work-efficient GPU scan.
 - *Implementation:* Most of this is the same as 3. combined with 1., but for the compaction part, we need to launch a few simple helping kernels. Since we don't need to pad these kernels, they use a smaller blocksize and thus this is bottlenecked by the scanning algorithm segment with only a little extra work added for compaction. Thus, this is also O(log(n)).

## 5. Thrust Exclusive Scan
 - *Overview:* This uses the thrust library's built-in exclusive scan function for comparison.
 - *Implementation:* I simply create device vectors for my input and output arrays and then use thrust::exclusive_scan on them.

# Results
All of these results run with a block size of 32 threads. I found that this is what consistently yields the best results, since the block count is varied, and since we end up using less than a warp's worth for the last few loops of the work-efficient algorithms, I wanted to minimize the threadcount per block--ie, making it 32.
The algorithms are tested for both power of 2 arrays and non-power-of-2 arrays, which have the same length and subtract 3.
Here are the results for varying array lengths for the scan algorithm in graph form, with the exact values visible further down:
![scanp2.png]
![scannp2.png]
As we can see, there isn't much difference between the two different algorithms, except for Thrust, which seems to speed up relatively for non-power-of-2 arrays.
We can also see that CPU and Naive are actually quite good for smaller array sizes. This makes sense, since the work-efficient algorithm comes with extra overhead, such as using two different kernels that we launch log(n) times. The speedup is there, but this overhead is larger than the time we save with this algorithm. CPU is faster than Naive because Naive has a lot of idling warps that really don't need to be launched. We're wasting a lot of time doing nothing, and the CPU is honestly not that slow for these values.
Later, when the array is around 2^20, work-efficient becomes the fastest (even overtaking thrust!), since it is around here where the time we save with this algorithm, and by optimizing the number of blocks and threads we launch, makes up for the amount of overhead caused by the extra kernel launches. However, Thrust soon overtakes this algorithm, since I imagine it is very well optimized. Unfortunately, the Github page of its documentation was down at the time of me writing this, so I can not say how exactly it is optimized. Interestingly, work-efficient and thrust start to converge again at 2^30, but I could not test for larger values due to memory overflow.

The results for compaction are similar:
![compact.png]
Since I didn't have a naive GPU method for this, I combined the 2^n and non-2^n sized arrays into this one graph, so we can clearly see that there isn't much difference between their runtime. We also see a similar trend as with the scan algorithms, where CPU starts out being the fastest, but the work-efficient GPU method overtakes it at 2^20th. However, the GPU algorithm suffers above 2^28th and the CPU method overtakes it again. I am not sure why this is the case, especially since this algorithm is pretty much the same as the scan one, with two added kernels that should be quite fast. To investigate, I profiled only the compaction algorithms with Nsight Compute, the result of which can be found in the file "Compaction_Compute_Results." These results showed me that my two extra kernels, "kernMapToBoolean" and "kernScatter" were indeed very slow, and while it gave me the suggestion that I should double my warps, it showed that I do not have any expected speedup, as shown in this image:
![kernelTimes.png]
I still decided to increase my block size by 2 just for these two kernels, and captured the results again, which showed no real improvement (as somewhat expected by the program itself). These results can be seen in "Compaction_Compute_Results2," and in the image below:
![kernelTimes2.png]
I believe that these kernels are so slow because we are not accessing contiguous memory in them, and when our arrays get this large, this becomes a massive problem, since values can be all over the place. This is supported by how massive the memory throughput is for these kernels. The CPU has no such issues with much more working memory, thus it overtakes this algorithm.

Since the graphs make the exact values hard to see, and since the time is on a logarithmic scale, here are the exact values measured that are plotted in the graphs:
| Scan - pow2 | CPU     | Naive     | Work Efficient | Thrust   |
|------------|---------|-----------|----------------|----------|
| 8          | 0.0005  | 0.106048  | 0.263168       | 1.07731  |
| 16         | 0.1022  | 0.369184  | 0.396192       | 1.3809   |
| 20         | 1.767   | 1.68672   | 0.67232        | 1.88704  |
| 24         | 28.1866 | 31.927    | 8.45683        | 2.49914  |
| 28         | 437.757 | 488.105   | 109.804        | 13.1427  |
| 30         | 1941.75 | 12043.5   | 461.029        | 325.351  |

| Scan - non-pow2 | CPU     | Naive     | Work Efficient | Thrust   |
|----------------|---------|-----------|----------------|----------|
| 8              | 0.0005  | 0.073568  | 0.263168       | 0.038944 |
| 16             | 0.1113  | 0.314912  | 0.3136         | 0.121856 |
| 20             | 1.7528  | 1.82982   | 0.506528       | 0.701952 |
| 24             | 28.1238 | 30.2253   | 8.45827        | 1.5712   |
| 28             | 462.555 | 499.592   | 109.485        | 11.3465  |
| 30             | 2125.17 | 11120.8   | 455.907        | 309.002  |

| Compact | CPU--no scan(pow2) | CPU--no scan (non-pow2) | CPU--with scan | Work Efficient (pow2) | Work Efficient (non-pow2) |
|---------|--------------------|-------------------------|----------------|-----------------------|---------------------------|
| 8       | 0.0006             | 0.0005                  | 0.0017         | 0.219136              | 0.13824                   |
| 16      | 0.1475             | 0.1559                  | 0.2258         | 0.48016               | 0.45456                   |
| 20      | 2.2089             | 2.2477                  | 3.872          | 1.18496               | 1.1031                    |
| 24      | 40.2907            | 38.8878                 | 123.698        | 18.6854               | 17.6905                   |
| 28      | 609.81             | 650.047                 | 966.531        | 265.185               | 260.862                   |
| 30      | 2477.73            | 2500.5                  | 5201.82        | 37064.4               | 35130                     |

The testing results for 2^20th size arrays is below:
```

****************
** SCAN TESTS **
****************
    [  17  43   3   7   9  15  44  39  27   8  30   8   0 ...   8   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1.7731ms    (std::chrono Measured)
    [   0  17  60  63  70  79  94 138 177 204 212 242 250 ... 25686617 25686625 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1.8213ms    (std::chrono Measured)
    [   0  17  60  63  70  79  94 138 177 204 212 242 250 ... 25686497 25686530 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 1.6519ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 1.97299ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.603488ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.64752ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.38461ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.477024ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   3   3   1   1   3   0   1   3   0   2   2   2 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 2.4111ms    (std::chrono Measured)
    [   3   3   3   1   1   3   1   3   2   2   2   1   3 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.6017ms    (std::chrono Measured)
    [   3   3   3   1   1   3   1   3   2   2   2   1   3 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 3.6381ms    (std::chrono Measured)
    [   3   3   3   1   1   3   1   3   2   2   2   1   3 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 1.48467ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 1.08035ms    (CUDA Measured)
    passed
```
