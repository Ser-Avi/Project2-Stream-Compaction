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
 - *Implementation:* I simply create device vectors for my input and output arrays and then use thrust::excusive_scan on them.

# Results


