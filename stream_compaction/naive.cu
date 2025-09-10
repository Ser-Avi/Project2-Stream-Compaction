#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;

        int blockSize = 128;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernelScan(int n, int* odata, const int* idata, int d)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int pow2 = 1 << (d - 1);
            if (idx >= n) return;

            if (idx >= pow2)
            {
                odata[idx] = idata[idx - pow2] + idata[idx];
            }
            else
            {
                odata[idx] = idata[idx];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            int reqSize = ilog2ceil(n);
            int ceil = 1 << reqSize;

            int* arrayA;
            int* arrayB;

            cudaMalloc((void**)&arrayA, sizeof(int) * ceil);
            cudaMalloc((void**)&arrayB, sizeof(int) * ceil);
            cudaMemcpy(arrayB, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            for (int d = 1; d <= ilog2ceil(n); ++d)
            {
                kernelScan << <fullBlocksPerGrid, blockSize >> > (ceil, arrayA, arrayB, d);
                std::swap(arrayA, arrayB);
            }

            cudaMemcpy(odata, arrayB, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // shift to exclusive
            for (int i = n - 1; i > 0; --i)
            {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;

            cudaFree(arrayA);
            cudaFree(arrayB);

            timer().endGpuTimer();
        }
    }
}
