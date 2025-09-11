#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;

        int blockSize = 128;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int* data, int d)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx > n) return;
            idx = idx * d - 1;
            if (idx > n) return;
            data[idx] += data[idx - (d >> 1)];
        }

        __global__ void kernChangeOneVal(int index, int* data, int val)
        {
            data[index] = val;
        }

        __global__ void kernDownSweep(int n, int* data, int d)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx > n) return;
            idx = idx * d - 1;
            if (idx > n) return;
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
            //timer().startGpuTimer();
            // TODO
            int reqSize = ilog2ceil(n);
            int ceil = 1 << reqSize;

            dim3 fullBlocksPerGrid((ceil + blockSize - 1) / blockSize);

            int* array;
            cudaMalloc((void**)&array, sizeof(int) * ceil);
            cudaMemcpy(array, idata, sizeof(int) * ceil, cudaMemcpyHostToDevice);
            for (int d = 1; d < reqSize + 1; ++d)
            {
                kernUpSweep << <fullBlocksPerGrid, blockSize>> > (ceil - 1, array, 1 << d);
            }
            // this does this: array[n - 1] = 0; I think the fastest way, but looks crazy
           kernChangeOneVal << <1, 1 >> > (ceil - 1, array, 0);
            
            for (int d = reqSize; d > 0; --d)
            {
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (ceil - 1, array, 1 << d);
            }
            cudaMemcpy(odata, array, sizeof(int) * ceil, cudaMemcpyDeviceToHost);

            cudaFree(array);
            //timer().endGpuTimer();
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
            timer().startGpuTimer();
            // TODO
            int reqSize = ilog2ceil(n);
            int ceil = 1 << reqSize;

            dim3 fullBlocksPerGrid((ceil + blockSize - 1) / blockSize);

            int* iArray;
            int* boolArray;
            int* boolSum;
            cudaMalloc((void**)&boolArray, sizeof(int) * ceil);
            cudaMalloc((void**)&iArray, sizeof(int) * ceil);
            cudaMalloc((void**)&boolSum, sizeof(int) * ceil);
            cudaMemcpy(iArray, idata, sizeof(int) * ceil, cudaMemcpyHostToDevice);

            // Populate bool array
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (ceil, boolArray, iArray);
            
            // Scan on bool array
            scan(n, boolSum, boolArray);
            cudaMemcpy(odata, boolSum, sizeof(int) * ceil, cudaMemcpyDeviceToHost);
            int size = odata[ceil - 1];

            // Compact
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (ceil, boolArray, iArray, boolArray, boolSum);

            cudaMemcpy(odata, boolArray, sizeof(int) * ceil, cudaMemcpyDeviceToHost);

            cudaFree(boolArray);
            cudaFree(iArray);
            cudaFree(boolSum);
            timer().endGpuTimer();
            return size;
        }
    }
}
