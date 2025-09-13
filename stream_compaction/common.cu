#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            // TODO
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx > n - 1 || idx < 0) return;
            bools[idx] = idata[idx] != 0;
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            // TODO
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx > n - 1 || idx < 0) return;
            if (bools[idx] == 1)
            {
                odata[indices[idx]] = idata[idx];
            }
        }

        /// <summary>
        /// Resets buffer to the set value - used for padding with 0s
        /// </summary>
        /// <param name="N"></param>
        /// <param name="intBuffer"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        __global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index < N && index > -1) {
                intBuffer[index] = value;
            }
        }

    }
}
