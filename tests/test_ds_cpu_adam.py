# Copyright (C) 2021 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

import argparse
import torch
import time
import numpy as np
import pytest
import copy
import math
import unittest

from patrickstar.ops.op_builder import CPUAdamBuilder
from patrickstar.utils import logger


def torch_adam_update(step, lr, beta1, beta2, eps, weight_decay,
                      bias_correction, fp32_data_tensor, fp32_grad_tensor,
                      exp_avg, exp_avg_sq, loss_scale):
    if loss_scale > 0:
        fp32_grad_tensor.div_(loss_scale)
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    if weight_decay != 0:
        fp32_grad_tensor = fp32_grad_tensor.add(fp32_data_tensor,
                                                alpha=weight_decay)

    exp_avg.mul_(beta1).add_(fp32_grad_tensor, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(fp32_grad_tensor,
                                    fp32_grad_tensor,
                                    value=1 - beta2)
    if False:
        # Maintains the maximum of all 2nd moment running avg. till now
        torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
        # Use the max. for normalizing running avg. of gradient
        denom = (max_exp_avg_sqs[i].sqrt() /
                 math.sqrt(bias_correction2)).add_(eps)
    else:
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1

    fp32_data_tensor.addcdiv_(exp_avg, denom, value=-step_size)


class TestAccess(unittest.TestCase):
    def setUp(self):
        self.ds_opt_adam = CPUAdamBuilder().load()
        betas = [0.9, 0.99]
        lr = 0.9
        eps = 1e-6
        weight_decay = 0
        self.ds_opt_adam.create_adam(0, lr, betas[0], betas[1], eps,
                                     weight_decay, False, True)
        self.opt_id = 0

    def check_res(self, step, lr, eps, beta1, beta2, weight_decay, shape,
                  grad_dtype, loss_scale):
        p_data = torch.rand(shape)
        p_data_copy = p_data.clone()
        p_grad = torch.rand(shape, dtype=grad_dtype)
        if loss_scale > 0:
            p_grad.mul_(loss_scale)
        p_grad_copy = p_grad.clone().float()
        exp_avg = torch.rand(shape)
        exp_avg_copy = exp_avg.clone()
        exp_avg_sq = torch.rand(shape)
        exp_avg_sq_copy = exp_avg_sq.clone()

        self.ds_opt_adam.adam_update(
            self.opt_id,
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            True,
            p_data.view(-1),  #fp32 data
            p_grad.view(-1),  #fp32 grad
            exp_avg.view(-1),
            exp_avg_sq.view(-1),
            loss_scale)

        torch_adam_update(
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            True,
            p_data_copy,  #fp32 data
            p_grad_copy,  #fp32 grad
            exp_avg_copy,
            exp_avg_sq_copy,
            loss_scale)

        # torch_adam_update update the grad inplace.
        if loss_scale > 0:
            p_grad.div_(loss_scale)

        data_diff = torch.max(torch.abs(p_data_copy - p_data))
        self.assertLess(
            data_diff, 1e-4,
            f"p_data diff {data_diff}. Failed check, step {step}, lr {lr} eps {eps} beta1 {beta1} beta2 {beta2} weight_decay {weight_decay}"
        )
        max_grad_diff = torch.max(torch.abs(p_grad_copy - p_grad))
        self.assertTrue(max_grad_diff < 1e-4, f"diff {max_grad_diff}")
        max_exp_avg_diff = torch.max(torch.abs(exp_avg_copy - exp_avg))
        self.assertTrue(max_exp_avg_diff < 1e-4,
                        f"max_exp_avg_diff {max_exp_avg_diff}")
        max_exp_avg_sq_diff = torch.max(torch.abs(exp_avg_sq_copy -
                                                  exp_avg_sq))
        self.assertTrue(max_exp_avg_sq_diff < 1e-4,
                        f"max_exp_avg_sq_diff {max_exp_avg_sq_diff}")
        logger.debug(
            f'Passed check, step {step}, lr {lr} eps {eps} beta1 {beta1} beta2 {beta2} '
            f'weight_decay {weight_decay} grad_dtype {grad_dtype}')

    def test(self):
        for shape in [(1023, ), (1024, 32)]:
            for step in range(1, 10):
                for lr in [0.01, 0.1]:
                    for eps in [1e-8]:
                        for beta1 in [0.9, 0.8]:
                            for beta2 in [0.999, 0.9]:
                                for weight_decay in [0.001, 0]:
                                    for grad_dtype in [
                                            torch.float, torch.half
                                    ]:
                                        for loss_scale in [-1, 2**5]:
                                            self.check_res(
                                                step, lr, eps, beta1, beta2,
                                                weight_decay, shape,
                                                grad_dtype, loss_scale)


if __name__ == "__main__":
    unittest.main()
