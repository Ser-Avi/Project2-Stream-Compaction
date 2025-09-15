#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;

        int blockSize = 32;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* data, int d)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx = idx * d - 1;
            if (idx > n || idx < 0) return;
            data[idx] += data[idx - (d >> 1)];
        }

        __global__ void kernChangeOneVal(int index, int* data, int val)
        {
            data[index] = val;
        }

        __global__ void kernDownSweep(int n, int* data, int d)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx = idx * d - 1;
            if (idx > n || idx < 0) return;
            // Left child will become copy of parent
            // Right child will be sum of left and parent
            int left = idx - (d >> 1);
            int t = data[left];
            data[left] = data[idx];
            data[idx] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int reqSize = ilog2ceil(n);
            int ceil = 1 << reqSize;

            int* dev_array;
            cudaMalloc((void**)&dev_array, sizeof(int) * ceil);
            cudaMemcpy(dev_array, idata, sizeof(int) * ceil, cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            for (int d = 1; d < reqSize + 1; ++d)
            {
                // we halve the number of blocks, since we need half as many threads
                // same applies below
                dim3 blocksPerGrid(((ceil >> (d - 1)) + blockSize - 1) / blockSize);
                kernUpSweep << <blocksPerGrid, blockSize>> > (ceil - 1, dev_array, 1 << d);
            }
            // this does this: array[n - 1] = 0; I think the fastest way, but looks crazy
            kernChangeOneVal << <1, 1 >> > (ceil - 1, dev_array, 0);
            
            for (int d = reqSize; d > 0; --d)
            {
                dim3 blocksPerGrid(((ceil >> (d - 1)) + blockSize - 1) / blockSize);
                kernDownSweep << <blocksPerGrid, blockSize >> > (ceil - 1, dev_array, 1 << d);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_array, sizeof(int) * ceil, cudaMemcpyDeviceToHost);

            cudaFree(dev_array);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO
            int reqSize = ilog2ceil(n);
            int ceil = 1 << reqSize;

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 scanBlocksPerGrid((ceil + blockSize - 1) / blockSize);

            int* dev_iArray;            // input array buffer
            int* dev_boolArray;         // boolean array buffer
            int* dev_boolScan;           // scan of the boolean array buffer
            int* dev_outArray;          // the output array buffer
            cudaMalloc((void**)&dev_boolArray, sizeof(int) * n);
            cudaMalloc((void**)&dev_iArray, sizeof(int) * n);
            cudaMalloc((void**)&dev_boolScan, sizeof(int) * ceil);
            cudaMalloc((void**)&dev_outArray, sizeof(int) * n);
            // these 0 paddings are here just as a safety net, but in theory this should work without them
            // theory is not practice though
            Common::kernResetIntBuffer<<<fullBlocksPerGrid, blockSize >>>(n, dev_iArray, 0);
            Common::kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (n, dev_boolArray, 0);
            cudaMemcpy(dev_iArray, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // Populate bool array
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_boolArray, dev_iArray);
            
            // Scan on bool array
            Common::kernResetIntBuffer << <scanBlocksPerGrid, blockSize >> > (ceil, dev_boolScan, 0);    // padding for scan to work properly
            cudaMemcpy(dev_boolScan, dev_boolArray, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            for (int d = 1; d < reqSize + 1; ++d)
            {
                scanBlocksPerGrid = dim3(((ceil >> (d - 1)) + blockSize - 1) / blockSize);
                kernUpSweep << <scanBlocksPerGrid, blockSize >> > (ceil - 1, dev_boolScan, 1 << d);
            }
            kernChangeOneVal << <1, 1 >> > (ceil - 1, dev_boolScan, 0);
            for (int d = reqSize; d > 0; --d)
            {
                scanBlocksPerGrid = dim3(((ceil >> (d - 1)) + blockSize - 1) / blockSize);
                kernDownSweep << <scanBlocksPerGrid, blockSize >> > (ceil - 1, dev_boolScan, 1 << d);
            }

            cudaMemcpy(odata, dev_boolScan, sizeof(int) * (n + (n % 2)), cudaMemcpyDeviceToHost);
            int size = odata[n - ((n + 1) % 2)];

            //// Compact
            Common::kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (n, dev_outArray, 0);    // this is here so garbage values don't accidentally get added
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_outArray, dev_iArray, dev_boolArray, dev_boolScan);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_outArray, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_boolArray);
            cudaFree(dev_iArray);
            cudaFree(dev_boolScan);
            cudaFree(dev_outArray);
            return size;
        }
    }
}
