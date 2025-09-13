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
            if (idx > n || idx < 0) return;
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
            if (idx > n || idx < 0) return;
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
            timer().startGpuTimer();
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
            timer().endGpuTimer();
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

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            dim3 scanBlocksPerGrid((ceil + blockSize - 1) / blockSize);

            int* iArray;
            int* boolArray;
            int* boolSum;
            int* outTemp;
            cudaMalloc((void**)&boolArray, sizeof(int) * n);
            cudaMalloc((void**)&iArray, sizeof(int) * n);
            cudaMalloc((void**)&boolSum, sizeof(int) * ceil);
            cudaMalloc((void**)&outTemp, sizeof(int) * n);
            Common::kernResetIntBuffer<<<fullBlocksPerGrid, blockSize >>>(n, iArray, 0);
            Common::kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (n, boolArray, 0);
            cudaMemcpy(iArray, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            // Populate bool array
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, boolArray, iArray);
            
            // Scan on bool array
            Common::kernResetIntBuffer << <scanBlocksPerGrid, blockSize >> > (ceil, boolSum, 0);    // padding for scan to work properly
            cudaMemcpy(boolSum, boolArray, sizeof(int) * n, cudaMemcpyDeviceToDevice);
            for (int d = 1; d < reqSize + 1; ++d)
            {
                kernUpSweep << <scanBlocksPerGrid, blockSize >> > (ceil - 1, boolSum, 1 << d);
            }
            kernChangeOneVal << <1, 1 >> > (ceil - 1, boolSum, 0);
            for (int d = reqSize; d > 0; --d)
            {
                kernDownSweep << <scanBlocksPerGrid, blockSize >> > (ceil - 1, boolSum, 1 << d);
            }

            cudaMemcpy(odata, boolSum, sizeof(int) * (n + (n % 2)), cudaMemcpyDeviceToHost);
            int size = odata[n - ((n + 1) % 2)];

            //// Compact
            Common::kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (n, outTemp, 0);    // this is here so garbage values don't accidentally get added
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, outTemp, iArray, boolArray, boolSum);

            cudaMemcpy(odata, outTemp, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(boolArray);
            cudaFree(iArray);
            cudaFree(boolSum);
            cudaFree(outTemp);
            timer().endGpuTimer();
            return size;
        }
    }
}
