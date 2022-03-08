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


class CommGroupInfo:
    groups = {}

    def __init__(self, id):
        self.id = id
        self.elements = []

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f"comm_group({self.id})"


groups = {}


def get_comm_group(group_id):
    if group_id in groups:
        return groups[group_id]
    group = CommGroupInfo(id=group_id)
    groups[group_id] = group
    return group


class CommInfo:
    def __init__(self, chunk_id):
        world_size = get_world_size()
        group_id = chunk_id // world_size
        self.group = get_comm_group(group_id)
        self.group.elements.append(chunk_id)
        self.offset = chunk_id % world_size

    @property
    def group_id(self):
        return self.group.id

    def __str__(self):
        return f"({self.group}, {self.offset})"
