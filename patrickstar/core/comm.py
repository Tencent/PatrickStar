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

from patrickstar.utils import get_world_size


class CommGroupInfo(object):
    def __init__(self, chunk_type, id):
        self.chunk_type = chunk_type
        self.id = id

    def __str__(self):
        return f"({self.chunk_type}, {self.id})"


class CommInfo(object):
    def __init__(self, chunk_type, group_id, offset):
        assert offset < get_world_size()
        self.group = CommGroupInfo(chunk_type=chunk_type, id=group_id)
        self.offset = offset

    @property
    def chunk_type(self):
        return self.group.chunk_type

    @property
    def group_id(self):
        return self.group.id

    def __str__(self):
        return f"({self.group}, {self.offset})"
