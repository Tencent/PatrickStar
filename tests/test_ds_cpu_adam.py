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

import math
import unittest

import torch
from common import distributed_test


def torch_adam_update(
    step,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction,
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    loss_scale,
    use_adamw,
):
    if loss_scale > 0:
        grad.div_(loss_scale)
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    if weight_decay != 0:
        if use_adamw:
            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)
        else:
            grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1

    param.addcdiv_(exp_avg, denom, value=-step_size)


class TestAccess(unittest.TestCase):
    def setUp(self):
        pass

    def check_res(
        self,
        step,
        lr,
        eps,
        beta1,
        beta2,
        weight_decay,
        shape,
        grad_dtype,
        loss_scale,
        use_adamw,
        cpu_adam_op,
    ):
        self.opt_id = 0
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

        cpu_adam_op.create_adam(0, lr, beta1, beta2, eps, weight_decay, use_adamw, True)
        cpu_adam_op.adam_update(
            self.opt_id,
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            True,
            p_data.view(-1),  # fp32 data
            p_grad.view(-1),  # fp32 grad
            exp_avg.view(-1),
            exp_avg_sq.view(-1),
            loss_scale,
        )

        torch_adam_update(
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            True,
            p_data_copy,  # fp32 data
            p_grad_copy,  # fp32 grad
            exp_avg_copy,
            exp_avg_sq_copy,
            loss_scale,
            use_adamw,
        )

        # torch_adam_update update the grad inplace.
        if loss_scale > 0:
            p_grad.div_(loss_scale)

        data_diff = torch.max(torch.abs(p_data_copy - p_data))
        self.assertLess(
            data_diff,
            1e-4,
            f"p_data diff {data_diff}. Failed check, step {step}, lr {lr} eps "
            f"{eps} beta1 {beta1} beta2 {beta2} weight_decay {weight_decay}",
        )
        max_grad_diff = torch.max(torch.abs(p_grad_copy - p_grad))
        self.assertTrue(max_grad_diff < 1e-4, f"diff {max_grad_diff}")
        max_exp_avg_diff = torch.max(torch.abs(exp_avg_copy - exp_avg))
        self.assertTrue(max_exp_avg_diff < 1e-4, f"max_exp_avg_diff {max_exp_avg_diff}")
        max_exp_avg_sq_diff = torch.max(torch.abs(exp_avg_sq_copy - exp_avg_sq))
        self.assertTrue(
            max_exp_avg_sq_diff < 1e-4, f"max_exp_avg_sq_diff {max_exp_avg_sq_diff}"
        )

    @distributed_test(world_size=[1])
    def test_ds_adam(self):
        from patrickstar.ops.op_builder.cpu_adam import CPUAdamBuilder

        try:
            # The pre-compiled cpu adam extension.
            from .adam import cpu_adam_op
        except ImportError:
            cpu_adam_op = CPUAdamBuilder().load()

        lr = 0.9
        eps = 1e-6
        weight_decay = 0
        for use_adamw in [False, True]:
            for shape in [(1023,), (1024, 32)]:
                for step in range(1, 2):
                    for lr in [0.01]:
                        for eps in [1e-8]:
                            for beta1 in [0.9]:
                                for beta2 in [0.999]:
                                    for weight_decay in [0.001]:
                                        for grad_dtype in [torch.float, torch.half]:
                                            for loss_scale in [-1, 2 ** 5]:
                                                self.check_res(
                                                    step,
                                                    lr,
                                                    eps,
                                                    beta1,
                                                    beta2,
                                                    weight_decay,
                                                    shape,
                                                    grad_dtype,
                                                    loss_scale,
                                                    use_adamw,
                                                    cpu_adam_op,
                                                )


if __name__ == "__main__":
    unittest.main()
