#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernUpSweep(int n, int* data, int d);
        __global__ void kernDownSweep(int n, int* data, int d);
        __global__ void kernChangeOneVal(int index, int* data, int val);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
