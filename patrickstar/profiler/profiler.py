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

from patrickstar.core.chunk_list import ChunkList, ChunkListType
from patrickstar.core.chunk_tensor_index import ChunkTensorIndex
from patrickstar.utils import logger


def display_access_info(chunk_list: ChunkListType):
    logger.debug('----------- SHOW ACCESS INFO -----------')
    for chunk_id in chunk_list.chunk_type_to_id_list_map[ChunkListType.PARAM_FP16]:
        chunk = chunk_list.chunk_id_to_chunk_dict_map[chunk_id]
        logger.debug(
            f'\t {chunk_id} cpu_access_moments {chunk.cpu_access_moments}')
        logger.debug(
            f'\t {chunk_id} gpu_access_moments {chunk.gpu_access_moments}')


def display_chunk_info(chunk_tensor_index: ChunkTensorIndex, chunk_list: ChunkList):
    logger.info(f'Print chunk list info.')

    overall_size = 0
    for type, type_chunk_list in chunk_tensor_index.chunk_type_to_chunk_id_list_map.items():
        logger.info(f'Chunk list {type}')
        for chunk_id in type_chunk_list:
            chunk = chunk_list[chunk_id]
            comm_group_id, comm_group_offset, _ = chunk_tensor_index.chunk_id_to_comm_group_map[chunk_id]
            assert comm_group_id is not None

            logger.info(
                f'Chunk id {chunk.chunk_id}, status {chunk.get_status()}, '
                f'comm group {comm_group_id, comm_group_offset}, '
                f'capacity {chunk.capacity / 1024 / 1024} M elems, '
                f'dtype {chunk.data_type} device {chunk.get_device()}'
            )
            for info in chunk_tensor_index.generate_tensor_info_in_order(chunk_id):
                assert info.chunk_id == chunk_id, f'{info.chunk_id} vs {chunk_id}'
                logger.debug(
                    f'** tensor: chunk_id {chunk_id}, start {info.start_offset}, '
                    f'end {info.start_offset + info.numel}, size {info.numel}, '
                    f'tensor_id {info.tensor_id}, status {info.status()}, name {info.tensor_name}'
                )
            overall_size += chunk.get_chunk_space()

    logger.info(f'OVERALL CHUNK SIZE {overall_size / 1024 / 1024 / 1024} GB')
