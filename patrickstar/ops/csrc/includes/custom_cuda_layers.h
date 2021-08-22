#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#include <cooperative_groups.h>
#include <curand_kernel.h>

void launch_param_update(const float* input, __half* output, int size, cudaStream_t stream);
void launch_param_update_half(const float* input, __half* output, int size, cudaStream_t stream);
