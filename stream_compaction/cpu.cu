#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; ++i)
            {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int oIdx = 0;
            for (int i = 0; i < n; ++i)
            {
                if (idata[i] != 0)
                {
                    odata[oIdx] = idata[i];
                    ++oIdx;
                }
            }
            timer().endCpuTimer();
            return oIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* t = new int [n];
            for (int i = 0; i < n; ++i)
            {
                t[i] = idata[i] == 0 ? 0 : 1;
            }

            // scan - can't reuse cpu scan func because of timer measurements
            odata[0] = 0;
            for (int i = 1; i < n; ++i)
            {
                odata[i] = odata[i - 1] + t[i - 1];
            }

            int len = odata[n - 1]; // array length is the scan of t

            for (int i = 0; i < n; ++i)
            {
                if (t[i] == 1)
                {
                    int index = odata[i];
                    odata[index] = idata[i];
                }
            }

            timer().endCpuTimer();
            return len;
        }
    }
}
