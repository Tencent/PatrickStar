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

from typing import List


class ChunkTensorIndex(object):
    def __init__(self):
        self.dict_tensor_id_chunk_id = {}
        self.dict_chunk_id_tensor_id = {}

    def delete_chunk_id(self, chunk_id):
        tid_delete_list = []
        for tid, cid in self.dict_tensor_id_chunk_id.items():
            if cid == chunk_id:
                tid_delete_list.append(tid)

        for tid in tid_delete_list:
            del self.dict_tensor_id_chunk_id[tid]

        del self.dict_chunk_id_tensor_id[chunk_id]

    def delete_tensor(self, tensor_id):
        cid_delete_list = []
        for cid, tid in self.dict_chunk_id_tensor_id.items():
            if tid == tensor_id:
                cid_delete_list.append(cid)

        for cid in cid_delete_list:
            del self.dict_chunk_id_tensor_id[cid]

        del self.dict_tensor_id_chunk_id[tid]

    def add_tensor(self, tensor_id, chunk_id):
        self.dict_tensor_id_chunk_id[tensor_id] = chunk_id
        self.dict_chunk_id_tensor_id[chunk_id] = tensor_id

    def tensor_id_to_chunk_id(self, tensor_id) -> int:
        return self.dict_tensor_id_chunk_id.get(tensor_id)

    def chunk_id_to_tensor_id_list(self, chunk_id):
        return self.dict_chunk_id_tensor_id.get(chunk_id)
