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

from collections import OrderedDict
import itertools

import torch

from patrickstar.core import is_registered
from patrickstar.utils import logger


def state_dict(module, client, destination=None, prefix="", keep_vars=False):
    def _save_to_state_dict(module, destination, prefix, keep_vars):
        for name, param in module._parameters.items():
            if param is not None:
                if is_registered(param):
                    if param.ps_attr.is_local():
                        if param.ps_attr.is_chunk_based():
                            ps_data = client.get_fp32(param)
                            destination[prefix + name] = (
                                ps_data if keep_vars else ps_data.detach()
                            )
                        else:
                            destination[prefix + name] = (
                                param if keep_vars else param.detach()
                            )
                    else:
                        # Do not save remote params
                        continue
                else:
                    # Params that are not registered are treated as normal.
                    destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in module._buffers.items():
            if buf is not None and name not in module._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()

    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, module in module._modules.items():
        if module is not None:
            state_dict(
                module, client, destination, prefix + name + ".", keep_vars=keep_vars
            )
    # TODO(zilinzhu): Figure out when we will use these hooks.
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def _load_from_state_dict(
    module,
    client,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # TODO(zilinzhu): Figure out when we will use these hooks.
    for hook in module._load_state_dict_pre_hooks.values():
        hook(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    persistent_buffers = {
        k: v
        for k, v in module._buffers.items()
        if k not in module._non_persistent_buffers_set
    }
    local_name_params = itertools.chain(
        module._parameters.items(), persistent_buffers.items()
    )
    local_state = {k: v for k, v in local_name_params if v is not None}

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]

            if (
                isinstance(param, torch.nn.Parameter)
                and is_registered(param)
                and param.ps_attr.is_chunk_based()
            ):
                if param.ps_attr.is_local():
                    param.data = client.get_fp32(param)
                else:
                    continue

            if input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append(
                    "size mismatch for {}: copying a param with shape {} from checkpoint, "
                    "the shape in current model is {}.".format(
                        key, input_param.shape, param.shape
                    )
                )
                continue
            try:
                with torch.no_grad():
                    param.copy_(input_param)
            except MemoryError as ex:
                error_msgs.append(
                    'While copying the parameter named "{}", '
                    "whose dimensions in the model are {} and "
                    "whose dimensions in the checkpoint are {}, "
                    "an exception occurred : {}.".format(
                        key, param.size(), input_param.size(), ex.args
                    )
                )

            if (
                isinstance(param, torch.nn.Parameter)
                and is_registered(param)
                and param.ps_attr.is_chunk_based()
            ):
                if param.ps_attr.is_local():
                    fp32_data = param.data
                    client.access(param, torch.device("cpu:0"))
                    param.data.copy_(fp32_data)
                    client.release(param)

        elif strict:
            missing_keys.append(key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix):
                input_name = key[len(prefix) :]
                input_name = input_name.split(".", 1)[
                    0
                ]  # get the name of param/buffer/child
                if input_name not in module._modules and input_name not in local_state:
                    unexpected_keys.append(key)


def load_state_dict(module, client, state_dict, strict=False):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        # mypy isn't aware that "_metadata" exists in state_dict
        state_dict._metadata = metadata  # type: ignore[attr-defined]

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        # TODO(zilinzhu): There are some module type may need to be dealt with separately,
        # e.g. BatchNorm, InstanceNorm... (Classes with their own _load_from_state_dict
        # instead of inheriting from nn.Module.)
        _load_from_state_dict(
            module,
            client,
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")  # noqa: F821

    load(module)
    del load

    if unexpected_keys:
        logger.warning(
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in unexpected_keys)
            ),
        )

    missing_keys = list(filter(lambda k: k[-6:] != ".dummy", missing_keys))
    if missing_keys:
        logger.warning(
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in missing_keys)
            ),
        )

    if strict:
        if unexpected_keys or missing_keys:
            raise RuntimeError("Failed to load model strictly.")
