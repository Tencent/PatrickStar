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

import torch
import warnings
from typing import Any, Iterable, List, Tuple
from patrickstar.utils import see_memory_usage

CPU_CHECKPOINT = True


def move_to_device(item, device, criterion_func):
    """
    Move tensor on to specified device by changing the storage.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to move or (possibly nested) container of tensors to move.
        device: target device
        criterion_func: Function to restrict move operation to items meet criterion
    Returns:
        None
    """
    if criterion_func(item):
        device_copy = item.to(device)
        item.data = device_copy.data
        return item
    elif isinstance(item, list):
        return [move_to_device(v, device, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([move_to_device(v, device, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: move_to_device(v, device, criterion_func) for k, v in item.items()}
    else:
        return item


def extract_tensors(all_objects):
    """
    Separate objects in list/tuple into tensors and non-tensors and create a mapping to enable re-aggregation.
    The order of tensors and non-tensors is preserved in their respective output groups.
    Parameters:
        all_objects (list/tuple): Objects containing tensors and non-tensors to be split.
    Returns:
        tuple: Containing tensors, non-tensors, and bools of whether each position in original list/tuple was a tensor.
    """
    tensor_objects = [v for v in all_objects if torch.is_tensor(v)]
    non_tensor_objects = [v for v in all_objects if not torch.is_tensor(v)]
    tensor_flags = [torch.is_tensor(v) for v in all_objects]
    if type(all_objects) is tuple:
        return tuple(tensor_objects), tuple(non_tensor_objects), tuple(tensor_flags)
    return tensor_objects, non_tensor_objects, tensor_flags


def is_activation_to_checkpoint(item):
    """
    Is an activation to be checkpointed
    """
    return torch.is_tensor(item) and item.is_floating_point()


def get_cpu_activations_for_backward(args, inputs):
    new_args = []
    for i, (arg, inp) in enumerate(zip(args, inputs)):
        if not is_activation_to_checkpoint(arg):
            new_args.append(arg)
            continue

        arg.data = inp.data
        new_args.append(arg)

    return new_args


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn(
            "None of the inputs have requires_grad=True. Gradients will be None"
        )


def merge_tensors(tensor_objects, non_tensor_objects, tensor_flags):
    """
    Merge two lists (or tuples) of tensors and non-tensors using a mapping of positions in merged list (or tuple).
    Parameters:
        tensor_objects (list/tuple): Tensors to merge.
        non_tensor_objects (list/tuple): Non-tensors to merge.
        tensor_flags (list/tuple): Indicates whether each position in output is a tensor.
    Returns:
        tuple: Merge of tensors and non-tensors
    """
    merged_objects = []
    tensor_idx = 0
    non_tensor_idx = 0

    real_tensor_flags = None

    # remove the flags that are assigned to the size of the flattened tensors
    real_tensor_flags = tensor_flags

    for is_tensor in real_tensor_flags:
        if is_tensor:
            merged_objects.append(tensor_objects[tensor_idx])
            tensor_idx += 1
        else:
            merged_objects.append(non_tensor_objects[non_tensor_idx])
            non_tensor_idx += 1

    return tuple(merged_objects)


# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_devices = list(
        set(
            arg.get_device()
            for arg in args
            if isinstance(arg, torch.Tensor) and arg.is_cuda
        )
    )

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states) -> None:
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)


def copy_to_device(item, device, criterion_func):
    """
    Return a copy of tensor on specified device.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to copy or (possibly nested) container of tensors to copy.
        device: target device
        criterion_func: Function to restrict copy operation to items meet criterion
    Returns:
        None
    """
    if criterion_func(item):
        return item.to(device)
    elif isinstance(item, list):
        return [copy_to_device(v, device, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([copy_to_device(v, device, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: copy_to_device(v, device, criterion_func) for k, v in item.items()}
    else:
        return item


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        def save_args_for_backward(*all_args):
            tensor_args, non_tensor_args, tensor_flags = extract_tensors(
                all_objects=all_args
            )
            ctx.save_for_backward(*tensor_args)
            ctx.non_tensor_args = non_tensor_args
            ctx.tensor_flags = tensor_flags

        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.had_autocast_in_fwd = torch.is_autocast_enabled()
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)

        if CPU_CHECKPOINT:
            inputs = copy_to_device(
                args,
                device=torch.device("cpu"),
                criterion_func=is_activation_to_checkpoint,
            )
        else:
            inputs = args
        cuda_device = torch.cuda.current_device()
        inputs_cuda = copy_to_device(
            args, device=cuda_device, criterion_func=is_activation_to_checkpoint
        )

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        # ctx.inputs = []
        # ctx.tensor_indices = []
        # tensor_inputs = []
        # for i, arg in enumerate(args):
        #     if torch.is_tensor(arg):
        #         tensor_inputs.append(arg)
        #         ctx.tensor_indices.append(i)
        #         ctx.inputs.append(None)
        #     else:
        #         ctx.inputs.append(arg)

        # ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*inputs_cuda)

        del inputs_cuda
        if CPU_CHECKPOINT:
            new_args = get_cpu_activations_for_backward(args, inputs)
            save_args_for_backward(*new_args)
        else:
            save_args_for_backward(*args)

        # Tensors returned from forward() may not be differentiable.
        if torch.is_tensor(outputs):
            non_grad_outputs = [outputs] if not outputs.is_floating_point() else []
        else:
            non_grad_outputs = [
                o for o in outputs if torch.is_tensor(o) and not o.is_floating_point()
            ]
        ctx.mark_non_differentiable(*non_grad_outputs)

        if torch.is_tensor(outputs):
            # all_outputs += [outputs]
            return outputs
        else:
            # all_outputs += outputs
            outputs, _, _ = extract_tensors(all_objects=outputs)
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument."
            )
        # # Copy the list to avoid modifying original list.
        # inputs = list(ctx.inputs)
        # tensor_indices = ctx.tensor_indices
        # tensors = ctx.saved_tensors

        # # Fill in inputs with appropriate saved tensors.
        # for i, idx in enumerate(tensor_indices):
        #     inputs[idx] = tensors[i]

        cuda_device = torch.cuda.current_device()
        if CPU_CHECKPOINT:
            inputs = move_to_device(
                ctx.saved_tensors, cuda_device, is_activation_to_checkpoint
            )
            detached_inputs = detach_variable(inputs)
        else:
            inputs = ctx.saved_tensors
            detached_inputs = detach_variable(inputs)

        # Add non tensor input args
        detached_inputs = merge_tensors(
            tensor_objects=detached_inputs,
            non_tensor_objects=ctx.non_tensor_args,
            tensor_flags=ctx.tensor_flags,
        )
        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices

        see_memory_usage("checkpoint before fwd")
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            # detached_inputs = detach_variable(tuple(inputs))
            with torch.enable_grad(), torch.cuda.amp.autocast(ctx.had_autocast_in_fwd):
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary"
            )

        see_memory_usage("checkpoint before backward")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        see_memory_usage("checkpoint after backward")
        return (None, None) + grads


def checkpoint(function, *args, **kwargs):
    r"""Checkpoint a model or part of the model

    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.

    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retrieved, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.

    The output of :attr:`function` can contain non-Tensor values and gradient
    recording is only performed for the Tensor values. Note that if the output
    consists of nested structures (ex: custom objects, lists, dicts etc.)
    consisting of Tensors, these Tensors nested in custom structures will not
    be considered as part of autograd.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.

    .. warning::
        If checkpointed segment contains tensors detached from the computational
        graph by `detach()` or `torch.no_grad()`, the backward pass will raise an
        error. This is because `checkpoint` makes all the outputs require
        gradients which causes issues when a tensor is defined to have no
        gradient in the model. To circumvent this, detach the tensors outside of
        the `checkpoint` function.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients. At least one of the outputs needs to have
        :code:`requires_grad=True` as well.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    return CheckpointFunction.apply(function, preserve, *args)


def checkpoint_sequential(functions, segments, input, **kwargs):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. The inputs of each checkpointed segment will be saved for
    re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    .. warning:
        Since PyTorch 1.4, it allows only one Tensor as the input and
        intermediate outputs, just like :class:`torch.nn.Sequential`.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        input: A Tensor that is input to :attr:`functions`
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input)
            return input

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint(
            run_function(start, end, functions), input, preserve_rng_state=preserve
        )
    return run_function(end + 1, len(functions) - 1, functions)(input)
