// Copyright (C) 2021 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

// // Copyright (C) 2021 THL A29 Limited, a Tencent company.
// // All rights reserved.
// // Licensed under the BSD 3-Clause License (the "License"); you may
// // not use this file except in compliance with the License. You may
// // obtain a copy of the License at
// // https://opensource.org/licenses/BSD-3-Clause
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" basis,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// // implied. See the License for the specific language governing
// // permissions and limitations under the License.
// // See the AUTHORS file for names of contributors.

#include "cpu_adam.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <omp.h>
#include <torch/extension.h>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

#define ROUND_DOWN(size, step) ((size) & ~((step)-1))

// C++ interface

void Adam_Optimizer::Step(float* _params,
                          float* grads,
                          float* _exp_avg,
                          float* _exp_avg_sq,
                          size_t _param_size,
                          bool param_half_precision,
                          bool grad_half_precision,
                          float loss_scale)
{
    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;

    float step_size = -1 * _alpha / _bias_correction1;
    float w_decay = -1 * _alpha * _weight_decay;
    size_t rounded_size = 0;

    __half* grads_cast_h;
    __half* params_cast_h;
    if (grad_half_precision) {
        grads_cast_h = reinterpret_cast<__half*>(grads);
    }
    if (param_half_precision) {
        params_cast_h = reinterpret_cast<__half*>(_params);
    }

#if defined(__AVX512__) or defined(__AVX256__)

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);

    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(_bias_correction2);

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    AVX_Data weight_decay4;
    if (_weight_decay > 0)
        weight_decay4.data = (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
    rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH);

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }

#pragma omp parallel for
        for (size_t i = t; i < offset; i += SIMD_WIDTH) {
            AVX_Data grad_4;
            if (grad_half_precision) {
                grad_4.data = SIMD_LOAD_HALF(grads_cast_h + i);
            } else {
                grad_4.data = SIMD_LOAD(grads + i);
            }
            if (loss_scale > 0) {
                AVX_Data loss_scale_vec;
                loss_scale_vec.data = SIMD_LOAD_SCALAR(loss_scale);
                grad_4.data = SIMD_DIV(grad_4.data, loss_scale_vec.data);
            }
            AVX_Data momentum_4;
            momentum_4.data = SIMD_LOAD(_exp_avg + i);
            AVX_Data variance_4;
            variance_4.data = SIMD_LOAD(_exp_avg_sq + i);

            AVX_Data param_4;
            if (param_half_precision) {
                param_4.data = SIMD_LOAD_HALF(params_cast_h + i);
            } else {
                param_4.data = SIMD_LOAD(_params + i);
            }

            if (_weight_decay > 0 && !_adamw_mode) {
                grad_4.data = SIMD_FMA(param_4.data, weight_decay4.data, grad_4.data);
            }
            momentum_4.data = SIMD_MUL(momentum_4.data, betta1_4.data);
            momentum_4.data = SIMD_FMA(grad_4.data, betta1_minus1_4.data, momentum_4.data);

            variance_4.data = SIMD_MUL(variance_4.data, betta2_4.data);
            grad_4.data = SIMD_MUL(grad_4.data, grad_4.data);
            variance_4.data = SIMD_FMA(grad_4.data, betta2_minus1_4.data, variance_4.data);

            grad_4.data = SIMD_SQRT(variance_4.data);
            grad_4.data = SIMD_FMA(grad_4.data, bias2_sqrt.data, eps_4.data);
            grad_4.data = SIMD_DIV(momentum_4.data, grad_4.data);
            if (_weight_decay > 0 && _adamw_mode) {
                param_4.data = SIMD_FMA(param_4.data, weight_decay4.data, param_4.data);
            }
            param_4.data = SIMD_FMA(grad_4.data, step_size_4.data, param_4.data);

            if (param_half_precision) {
                SIMD_STORE_HALF((float*)(params_cast_h + i), param_4.data);
            } else {
                SIMD_STORE(_params + i, param_4.data);
            }

            SIMD_STORE(_exp_avg + i, momentum_4.data);
            SIMD_STORE(_exp_avg_sq + i, variance_4.data);
        }
    }

#endif

    if (_param_size > rounded_size) {
        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }

#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = grad_half_precision ? (float)grads_cast_h[k] : grads[k];
                if (loss_scale > 0) {
                  grad /= loss_scale;
                }
                float param = param_half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;

                if (param_half_precision)
                    params_cast_h[k] = (__half)param;
                else
                    _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
        }
    }
}

void Adam_Optimizer::Step_4(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            bool param_half_precision,
                            bool grad_half_precision,
                            float loss_scale)
{
    size_t rounded_size = 0;

    __half* grads_cast_h;
    __half* params_cast_h;
    if (grad_half_precision) {
        grads_cast_h = reinterpret_cast<__half*>(grads);
    }
    if (param_half_precision) {
        params_cast_h = reinterpret_cast<__half*>(_params);
    }

#if defined(__AVX512__) or defined(__AVX256__)

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(_bias_correction2);

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha / _bias_correction1;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float w_decay = -1 * _alpha * _weight_decay;
    AVX_Data weight_decay4;
    if (_weight_decay > 0)
        weight_decay4.data = (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
    rounded_size = ROUND_DOWN(_param_size, (SIMD_WIDTH << 2));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 2)) {
            AVX_Data grad_4[4];
            if (grad_half_precision) {
                grad_4[0].data = SIMD_LOAD_HALF(grads_cast_h + i);
                grad_4[1].data = SIMD_LOAD_HALF(grads_cast_h + i + SIMD_WIDTH);
                grad_4[2].data = SIMD_LOAD_HALF(grads_cast_h + i + (SIMD_WIDTH << 1));
                grad_4[3].data = SIMD_LOAD_HALF(grads_cast_h + i + SIMD_WIDTH * 3);
            } else {
                grad_4[0].data = SIMD_LOAD(grads + i);
                grad_4[1].data = SIMD_LOAD(grads + i + SIMD_WIDTH);
                grad_4[2].data = SIMD_LOAD(grads + i + (SIMD_WIDTH << 1));
                grad_4[3].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 3);
            }
            if (loss_scale > 0) {
                AVX_Data loss_scale_vec;
                loss_scale_vec.data = SIMD_LOAD_SCALAR(loss_scale);
                grad_4[0].data = SIMD_DIV(grad_4[0].data, loss_scale_vec.data);
                grad_4[1].data = SIMD_DIV(grad_4[1].data, loss_scale_vec.data);
                grad_4[2].data = SIMD_DIV(grad_4[2].data, loss_scale_vec.data);
                grad_4[3].data = SIMD_DIV(grad_4[3].data, loss_scale_vec.data);
            }
            AVX_Data momentum_4[4];
            momentum_4[0].data = SIMD_LOAD(_exp_avg + i);
            momentum_4[1].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH);
            momentum_4[2].data = SIMD_LOAD(_exp_avg + i + (SIMD_WIDTH << 1));
            momentum_4[3].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 3);

            AVX_Data variance_4[4];
            variance_4[0].data = SIMD_LOAD(_exp_avg_sq + i);
            variance_4[1].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH);
            variance_4[2].data = SIMD_LOAD(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            variance_4[3].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 3);

            AVX_Data param_4[4];
            if (param_half_precision) {
                param_4[0].data = SIMD_LOAD_HALF(params_cast_h + i);
                param_4[1].data = SIMD_LOAD_HALF(params_cast_h + i + SIMD_WIDTH);
                param_4[2].data = SIMD_LOAD_HALF(params_cast_h + i + (SIMD_WIDTH << 1));
                param_4[3].data = SIMD_LOAD_HALF(params_cast_h + i + SIMD_WIDTH * 3);
            } else {
                param_4[0].data = SIMD_LOAD(_params + i);
                param_4[1].data = SIMD_LOAD(_params + i + SIMD_WIDTH);
                param_4[2].data = SIMD_LOAD(_params + i + (SIMD_WIDTH << 1));
                param_4[3].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 3);
            }

            if (_weight_decay > 0 && !_adamw_mode) {
                grad_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, grad_4[0].data);
                grad_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, grad_4[1].data);
                grad_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, grad_4[2].data);
                grad_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, grad_4[3].data);
            }

            momentum_4[0].data = SIMD_MUL(momentum_4[0].data, betta1_4.data);
            momentum_4[0].data = SIMD_FMA(grad_4[0].data, betta1_minus1_4.data, momentum_4[0].data);
            momentum_4[1].data = SIMD_MUL(momentum_4[1].data, betta1_4.data);
            momentum_4[1].data = SIMD_FMA(grad_4[1].data, betta1_minus1_4.data, momentum_4[1].data);
            momentum_4[2].data = SIMD_MUL(momentum_4[2].data, betta1_4.data);
            momentum_4[2].data = SIMD_FMA(grad_4[2].data, betta1_minus1_4.data, momentum_4[2].data);
            momentum_4[3].data = SIMD_MUL(momentum_4[3].data, betta1_4.data);
            momentum_4[3].data = SIMD_FMA(grad_4[3].data, betta1_minus1_4.data, momentum_4[3].data);

            variance_4[0].data = SIMD_MUL(variance_4[0].data, betta2_4.data);
            variance_4[1].data = SIMD_MUL(variance_4[1].data, betta2_4.data);
            variance_4[2].data = SIMD_MUL(variance_4[2].data, betta2_4.data);
            variance_4[3].data = SIMD_MUL(variance_4[3].data, betta2_4.data);
            grad_4[0].data = SIMD_MUL(grad_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_MUL(grad_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_MUL(grad_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_MUL(grad_4[3].data, grad_4[3].data);
            variance_4[0].data = SIMD_FMA(grad_4[0].data, betta2_minus1_4.data, variance_4[0].data);
            variance_4[1].data = SIMD_FMA(grad_4[1].data, betta2_minus1_4.data, variance_4[1].data);
            variance_4[2].data = SIMD_FMA(grad_4[2].data, betta2_minus1_4.data, variance_4[2].data);
            variance_4[3].data = SIMD_FMA(grad_4[3].data, betta2_minus1_4.data, variance_4[3].data);

            grad_4[0].data = SIMD_SQRT(variance_4[0].data);
            grad_4[1].data = SIMD_SQRT(variance_4[1].data);
            grad_4[2].data = SIMD_SQRT(variance_4[2].data);
            grad_4[3].data = SIMD_SQRT(variance_4[3].data);

            grad_4[0].data = SIMD_FMA(grad_4[0].data, bias2_sqrt.data, eps_4.data);
            grad_4[1].data = SIMD_FMA(grad_4[1].data, bias2_sqrt.data, eps_4.data);
            grad_4[2].data = SIMD_FMA(grad_4[2].data, bias2_sqrt.data, eps_4.data);
            grad_4[3].data = SIMD_FMA(grad_4[3].data, bias2_sqrt.data, eps_4.data);
            grad_4[0].data = SIMD_DIV(momentum_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_DIV(momentum_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_DIV(momentum_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_DIV(momentum_4[3].data, grad_4[3].data);

            if (_weight_decay > 0 && _adamw_mode) {
                param_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, param_4[0].data);
                param_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, param_4[1].data);
                param_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, param_4[2].data);
                param_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, param_4[3].data);
            }

            param_4[0].data = SIMD_FMA(grad_4[0].data, step_size_4.data, param_4[0].data);
            param_4[1].data = SIMD_FMA(grad_4[1].data, step_size_4.data, param_4[1].data);
            param_4[2].data = SIMD_FMA(grad_4[2].data, step_size_4.data, param_4[2].data);
            param_4[3].data = SIMD_FMA(grad_4[3].data, step_size_4.data, param_4[3].data);

            if (param_half_precision) {
                SIMD_STORE_HALF((float*)(params_cast_h + i), param_4[0].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + SIMD_WIDTH), param_4[1].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + (SIMD_WIDTH << 1)), param_4[2].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + SIMD_WIDTH * 3), param_4[3].data);
            } else {
                SIMD_STORE(_params + i, param_4[0].data);
                SIMD_STORE(_params + i + SIMD_WIDTH, param_4[1].data);
                SIMD_STORE(_params + i + (SIMD_WIDTH << 1), param_4[2].data);
                SIMD_STORE(_params + i + SIMD_WIDTH * 3, param_4[3].data);
            }

            SIMD_STORE(_exp_avg + i, momentum_4[0].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH, momentum_4[1].data);
            SIMD_STORE(_exp_avg + i + (SIMD_WIDTH << 1), momentum_4[2].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 3, momentum_4[3].data);

            SIMD_STORE(_exp_avg_sq + i, variance_4[0].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH, variance_4[1].data);
            SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH << 1), variance_4[2].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 3, variance_4[3].data);
        }
    }
#endif
    if (_param_size > rounded_size)
        Step((param_half_precision ? (float*)(params_cast_h + rounded_size) : _params + rounded_size),
             (grad_half_precision ? (float*)(grads_cast_h + rounded_size) : grads + rounded_size),
             (_exp_avg + rounded_size),
             (_exp_avg_sq + rounded_size),
             (_param_size - rounded_size),
             param_half_precision,
             grad_half_precision,
             loss_scale);
}

int create_adam_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float eps = 1e-8,
                          float weight_decay = 0,
                          bool adamw_mode = true,
                          bool should_log = false)
{
    auto opt =
        std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

void Adam_Optimizer::Step_8(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            bool param_half_precision,
                            bool grad_half_precision,
                            float loss_scale)
{
    size_t rounded_size = 0;

    __half* grads_cast_h;
    __half* params_cast_h;
    if (grad_half_precision) {
        grads_cast_h = reinterpret_cast<__half*>(grads);
    }
    if (param_half_precision) {
        params_cast_h = reinterpret_cast<__half*>(_params);
    }

#if defined(__AVX512__) or defined(__AVX256__)

    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);

    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(_bias_correction2);

    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    float step_size = -1 * _alpha / _bias_correction1;
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    float w_decay = -1 * _alpha * _weight_decay;
    AVX_Data weight_decay4;
    if (_weight_decay > 0)
        weight_decay4.data = (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));
    rounded_size = ROUND_DOWN(_param_size, (SIMD_WIDTH << 3));

    for (size_t t = 0; t < rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > rounded_size) copy_size = rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#pragma omp parallel for
        for (size_t i = t; i < offset; i += (SIMD_WIDTH << 3)) {
            AVX_Data grad_4[8];
            if (grad_half_precision) {
                grad_4[0].data = SIMD_LOAD_HALF(grads_cast_h + i);
                grad_4[1].data = SIMD_LOAD_HALF(grads_cast_h + i + SIMD_WIDTH);
                grad_4[2].data = SIMD_LOAD_HALF(grads_cast_h + i + (SIMD_WIDTH << 1));
                grad_4[3].data = SIMD_LOAD_HALF(grads_cast_h + i + SIMD_WIDTH * 3);
                grad_4[4].data = SIMD_LOAD_HALF(grads_cast_h + i + (SIMD_WIDTH << 2));
                grad_4[5].data = SIMD_LOAD_HALF(grads_cast_h + i + SIMD_WIDTH * 5);
                grad_4[6].data = SIMD_LOAD_HALF(grads_cast_h + i + SIMD_WIDTH * 6);
                grad_4[7].data = SIMD_LOAD_HALF(grads_cast_h + i + SIMD_WIDTH * 7);
            } else {
                grad_4[0].data = SIMD_LOAD(grads + i);
                grad_4[1].data = SIMD_LOAD(grads + i + SIMD_WIDTH);
                grad_4[2].data = SIMD_LOAD(grads + i + (SIMD_WIDTH << 1));
                grad_4[3].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 3);
                grad_4[4].data = SIMD_LOAD(grads + i + (SIMD_WIDTH << 2));
                grad_4[5].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 5);
                grad_4[6].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 6);
                grad_4[7].data = SIMD_LOAD(grads + i + SIMD_WIDTH * 7);
            }
            if (loss_scale > 0) {
                AVX_Data loss_scale_vec;
                loss_scale_vec.data = SIMD_LOAD_SCALAR(loss_scale);
                grad_4[0].data = SIMD_DIV(grad_4[0].data, loss_scale_vec.data);
                grad_4[1].data = SIMD_DIV(grad_4[1].data, loss_scale_vec.data);
                grad_4[2].data = SIMD_DIV(grad_4[2].data, loss_scale_vec.data);
                grad_4[3].data = SIMD_DIV(grad_4[3].data, loss_scale_vec.data);
                grad_4[4].data = SIMD_DIV(grad_4[4].data, loss_scale_vec.data);
                grad_4[5].data = SIMD_DIV(grad_4[5].data, loss_scale_vec.data);
                grad_4[6].data = SIMD_DIV(grad_4[6].data, loss_scale_vec.data);
                grad_4[7].data = SIMD_DIV(grad_4[7].data, loss_scale_vec.data);
            }

            AVX_Data momentum_4[8];
            momentum_4[0].data = SIMD_LOAD(_exp_avg + i);
            momentum_4[1].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH);
            momentum_4[2].data = SIMD_LOAD(_exp_avg + i + (SIMD_WIDTH << 1));
            momentum_4[3].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 3);
            momentum_4[4].data = SIMD_LOAD(_exp_avg + i + (SIMD_WIDTH << 2));
            momentum_4[5].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 5);
            momentum_4[6].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 6);
            momentum_4[7].data = SIMD_LOAD(_exp_avg + i + SIMD_WIDTH * 7);

            AVX_Data variance_4[8];
            variance_4[0].data = SIMD_LOAD(_exp_avg_sq + i);
            variance_4[1].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH);
            variance_4[2].data = SIMD_LOAD(_exp_avg_sq + i + (SIMD_WIDTH << 1));
            variance_4[3].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 3);
            variance_4[4].data = SIMD_LOAD(_exp_avg_sq + i + (SIMD_WIDTH << 2));
            variance_4[5].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 5);
            variance_4[6].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 6);
            variance_4[7].data = SIMD_LOAD(_exp_avg_sq + i + SIMD_WIDTH * 7);

            AVX_Data param_4[8];
            if (param_half_precision) {
                param_4[0].data = SIMD_LOAD_HALF(params_cast_h + i);
                param_4[1].data = SIMD_LOAD_HALF(params_cast_h + i + SIMD_WIDTH);
                param_4[2].data = SIMD_LOAD_HALF(params_cast_h + i + (SIMD_WIDTH << 1));
                param_4[3].data = SIMD_LOAD_HALF(params_cast_h + i + SIMD_WIDTH * 3);
                param_4[4].data = SIMD_LOAD_HALF(params_cast_h + i + (SIMD_WIDTH << 2));
                param_4[5].data = SIMD_LOAD_HALF(params_cast_h + i + SIMD_WIDTH * 5);
                param_4[6].data = SIMD_LOAD_HALF(params_cast_h + i + SIMD_WIDTH * 6);
                param_4[7].data = SIMD_LOAD_HALF(params_cast_h + i + SIMD_WIDTH * 7);
            } else {
                param_4[0].data = SIMD_LOAD(_params + i);
                param_4[1].data = SIMD_LOAD(_params + i + SIMD_WIDTH);
                param_4[2].data = SIMD_LOAD(_params + i + (SIMD_WIDTH << 1));
                param_4[3].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 3);
                param_4[4].data = SIMD_LOAD(_params + i + (SIMD_WIDTH << 2));
                param_4[5].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 5);
                param_4[6].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 6);
                param_4[7].data = SIMD_LOAD(_params + i + SIMD_WIDTH * 7);
            }

            if (_weight_decay > 0 && !_adamw_mode) {
                grad_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, grad_4[0].data);
                grad_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, grad_4[1].data);
                grad_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, grad_4[2].data);
                grad_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, grad_4[3].data);
                grad_4[4].data = SIMD_FMA(param_4[4].data, weight_decay4.data, grad_4[4].data);
                grad_4[5].data = SIMD_FMA(param_4[5].data, weight_decay4.data, grad_4[5].data);
                grad_4[6].data = SIMD_FMA(param_4[6].data, weight_decay4.data, grad_4[6].data);
                grad_4[7].data = SIMD_FMA(param_4[7].data, weight_decay4.data, grad_4[7].data);
            }

            momentum_4[0].data = SIMD_MUL(momentum_4[0].data, betta1_4.data);
            momentum_4[0].data = SIMD_FMA(grad_4[0].data, betta1_minus1_4.data, momentum_4[0].data);
            momentum_4[1].data = SIMD_MUL(momentum_4[1].data, betta1_4.data);
            momentum_4[1].data = SIMD_FMA(grad_4[1].data, betta1_minus1_4.data, momentum_4[1].data);
            momentum_4[2].data = SIMD_MUL(momentum_4[2].data, betta1_4.data);
            momentum_4[2].data = SIMD_FMA(grad_4[2].data, betta1_minus1_4.data, momentum_4[2].data);
            momentum_4[3].data = SIMD_MUL(momentum_4[3].data, betta1_4.data);
            momentum_4[3].data = SIMD_FMA(grad_4[3].data, betta1_minus1_4.data, momentum_4[3].data);
            momentum_4[4].data = SIMD_MUL(momentum_4[4].data, betta1_4.data);
            momentum_4[4].data = SIMD_FMA(grad_4[4].data, betta1_minus1_4.data, momentum_4[4].data);
            momentum_4[5].data = SIMD_MUL(momentum_4[5].data, betta1_4.data);
            momentum_4[5].data = SIMD_FMA(grad_4[5].data, betta1_minus1_4.data, momentum_4[5].data);
            momentum_4[6].data = SIMD_MUL(momentum_4[6].data, betta1_4.data);
            momentum_4[6].data = SIMD_FMA(grad_4[6].data, betta1_minus1_4.data, momentum_4[6].data);
            momentum_4[7].data = SIMD_MUL(momentum_4[7].data, betta1_4.data);
            momentum_4[7].data = SIMD_FMA(grad_4[7].data, betta1_minus1_4.data, momentum_4[7].data);

            variance_4[0].data = SIMD_MUL(variance_4[0].data, betta2_4.data);
            variance_4[1].data = SIMD_MUL(variance_4[1].data, betta2_4.data);
            variance_4[2].data = SIMD_MUL(variance_4[2].data, betta2_4.data);
            variance_4[3].data = SIMD_MUL(variance_4[3].data, betta2_4.data);
            variance_4[4].data = SIMD_MUL(variance_4[4].data, betta2_4.data);
            variance_4[5].data = SIMD_MUL(variance_4[5].data, betta2_4.data);
            variance_4[6].data = SIMD_MUL(variance_4[6].data, betta2_4.data);
            variance_4[7].data = SIMD_MUL(variance_4[7].data, betta2_4.data);
            grad_4[0].data = SIMD_MUL(grad_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_MUL(grad_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_MUL(grad_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_MUL(grad_4[3].data, grad_4[3].data);
            grad_4[4].data = SIMD_MUL(grad_4[4].data, grad_4[4].data);
            grad_4[5].data = SIMD_MUL(grad_4[5].data, grad_4[5].data);
            grad_4[6].data = SIMD_MUL(grad_4[6].data, grad_4[6].data);
            grad_4[7].data = SIMD_MUL(grad_4[7].data, grad_4[7].data);
            variance_4[0].data = SIMD_FMA(grad_4[0].data, betta2_minus1_4.data, variance_4[0].data);
            variance_4[1].data = SIMD_FMA(grad_4[1].data, betta2_minus1_4.data, variance_4[1].data);
            variance_4[2].data = SIMD_FMA(grad_4[2].data, betta2_minus1_4.data, variance_4[2].data);
            variance_4[3].data = SIMD_FMA(grad_4[3].data, betta2_minus1_4.data, variance_4[3].data);
            variance_4[4].data = SIMD_FMA(grad_4[4].data, betta2_minus1_4.data, variance_4[4].data);
            variance_4[5].data = SIMD_FMA(grad_4[5].data, betta2_minus1_4.data, variance_4[5].data);
            variance_4[6].data = SIMD_FMA(grad_4[6].data, betta2_minus1_4.data, variance_4[6].data);
            variance_4[7].data = SIMD_FMA(grad_4[7].data, betta2_minus1_4.data, variance_4[7].data);

            grad_4[0].data = SIMD_SQRT(variance_4[0].data);
            grad_4[1].data = SIMD_SQRT(variance_4[1].data);
            grad_4[2].data = SIMD_SQRT(variance_4[2].data);
            grad_4[3].data = SIMD_SQRT(variance_4[3].data);
            grad_4[4].data = SIMD_SQRT(variance_4[4].data);
            grad_4[5].data = SIMD_SQRT(variance_4[5].data);
            grad_4[6].data = SIMD_SQRT(variance_4[6].data);
            grad_4[7].data = SIMD_SQRT(variance_4[7].data);

            grad_4[0].data = SIMD_FMA(grad_4[0].data, bias2_sqrt.data, eps_4.data);
            grad_4[1].data = SIMD_FMA(grad_4[1].data, bias2_sqrt.data, eps_4.data);
            grad_4[2].data = SIMD_FMA(grad_4[2].data, bias2_sqrt.data, eps_4.data);
            grad_4[3].data = SIMD_FMA(grad_4[3].data, bias2_sqrt.data, eps_4.data);
            grad_4[4].data = SIMD_FMA(grad_4[4].data, bias2_sqrt.data, eps_4.data);
            grad_4[5].data = SIMD_FMA(grad_4[5].data, bias2_sqrt.data, eps_4.data);
            grad_4[6].data = SIMD_FMA(grad_4[6].data, bias2_sqrt.data, eps_4.data);
            grad_4[7].data = SIMD_FMA(grad_4[7].data, bias2_sqrt.data, eps_4.data);
            grad_4[0].data = SIMD_DIV(momentum_4[0].data, grad_4[0].data);
            grad_4[1].data = SIMD_DIV(momentum_4[1].data, grad_4[1].data);
            grad_4[2].data = SIMD_DIV(momentum_4[2].data, grad_4[2].data);
            grad_4[3].data = SIMD_DIV(momentum_4[3].data, grad_4[3].data);
            grad_4[4].data = SIMD_DIV(momentum_4[4].data, grad_4[4].data);
            grad_4[5].data = SIMD_DIV(momentum_4[5].data, grad_4[5].data);
            grad_4[6].data = SIMD_DIV(momentum_4[6].data, grad_4[6].data);
            grad_4[7].data = SIMD_DIV(momentum_4[7].data, grad_4[7].data);

            if (_weight_decay > 0 && _adamw_mode) {
                param_4[0].data = SIMD_FMA(param_4[0].data, weight_decay4.data, param_4[0].data);
                param_4[1].data = SIMD_FMA(param_4[1].data, weight_decay4.data, param_4[1].data);
                param_4[2].data = SIMD_FMA(param_4[2].data, weight_decay4.data, param_4[2].data);
                param_4[3].data = SIMD_FMA(param_4[3].data, weight_decay4.data, param_4[3].data);
                param_4[4].data = SIMD_FMA(param_4[4].data, weight_decay4.data, param_4[4].data);
                param_4[5].data = SIMD_FMA(param_4[5].data, weight_decay4.data, param_4[5].data);
                param_4[6].data = SIMD_FMA(param_4[6].data, weight_decay4.data, param_4[6].data);
                param_4[7].data = SIMD_FMA(param_4[7].data, weight_decay4.data, param_4[7].data);
            }

            param_4[0].data = SIMD_FMA(grad_4[0].data, step_size_4.data, param_4[0].data);
            param_4[1].data = SIMD_FMA(grad_4[1].data, step_size_4.data, param_4[1].data);
            param_4[2].data = SIMD_FMA(grad_4[2].data, step_size_4.data, param_4[2].data);
            param_4[3].data = SIMD_FMA(grad_4[3].data, step_size_4.data, param_4[3].data);
            param_4[4].data = SIMD_FMA(grad_4[4].data, step_size_4.data, param_4[4].data);
            param_4[5].data = SIMD_FMA(grad_4[5].data, step_size_4.data, param_4[5].data);
            param_4[6].data = SIMD_FMA(grad_4[6].data, step_size_4.data, param_4[6].data);
            param_4[7].data = SIMD_FMA(grad_4[7].data, step_size_4.data, param_4[7].data);

            if (param_half_precision) {
                SIMD_STORE_HALF((float*)(params_cast_h + i), param_4[0].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + SIMD_WIDTH), param_4[1].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + (SIMD_WIDTH << 1)), param_4[2].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + SIMD_WIDTH * 3), param_4[3].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + (SIMD_WIDTH << 2)), param_4[4].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + SIMD_WIDTH * 5), param_4[5].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + SIMD_WIDTH * 6), param_4[6].data);
                SIMD_STORE_HALF((float*)(params_cast_h + i + SIMD_WIDTH * 7), param_4[7].data);
            } else {
                SIMD_STORE(_params + i, param_4[0].data);
                SIMD_STORE(_params + i + SIMD_WIDTH, param_4[1].data);
                SIMD_STORE(_params + i + (SIMD_WIDTH << 1), param_4[2].data);
                SIMD_STORE(_params + i + SIMD_WIDTH * 3, param_4[3].data);
                SIMD_STORE(_params + i + (SIMD_WIDTH << 2), param_4[4].data);
                SIMD_STORE(_params + i + SIMD_WIDTH * 5, param_4[5].data);
                SIMD_STORE(_params + i + SIMD_WIDTH * 6, param_4[6].data);
                SIMD_STORE(_params + i + SIMD_WIDTH * 7, param_4[7].data);
            }

            SIMD_STORE(_exp_avg + i, momentum_4[0].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH, momentum_4[1].data);
            SIMD_STORE(_exp_avg + i + (SIMD_WIDTH << 1), momentum_4[2].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 3, momentum_4[3].data);
            SIMD_STORE(_exp_avg + i + (SIMD_WIDTH << 2), momentum_4[4].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 5, momentum_4[5].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 6, momentum_4[6].data);
            SIMD_STORE(_exp_avg + i + SIMD_WIDTH * 7, momentum_4[7].data);

            SIMD_STORE(_exp_avg_sq + i, variance_4[0].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH, variance_4[1].data);
            SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH << 1), variance_4[2].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 3, variance_4[3].data);
            SIMD_STORE(_exp_avg_sq + i + (SIMD_WIDTH << 2), variance_4[4].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 5, variance_4[5].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 6, variance_4[6].data);
            SIMD_STORE(_exp_avg_sq + i + SIMD_WIDTH * 7, variance_4[7].data);
        }
    }
#endif
    if (_param_size > rounded_size)
        Step_4((param_half_precision ? (float*)(params_cast_h + rounded_size) : _params + rounded_size),
               (grad_half_precision ? (float*)(grads_cast_h + rounded_size) : grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               param_half_precision,
               grad_half_precision,
               loss_scale);
}

int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq,
                 float loss_scale)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    // assert(params.options().dtype() == grads.options().dtype());

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.size(0),
                (params.options().dtype() == at::kHalf),
                (grads.options().dtype() == at::kHalf),
                loss_scale);

    opt->SynchronizeStreams();
    return 0;
}

int destroy_adam_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");
}
