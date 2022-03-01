# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from patrickstar.utils import get_world_size


class CommGroupInfo(object):
    groups = {}

    def __init__(self, chunk_type, id):
        self.chunk_type = chunk_type
        self.id = id
        self.elements = []

    def __hash__(self):
        return hash((self.chunk_type, self.id))

    def __eq__(self, other):
        return (self.chunk_type, self.id) == (other.chunk_type, other.id)

    def __str__(self):
        return f"({self.chunk_type}, {self.id})"


groups = {}


def get_comm_group(chunk_type, group_id):
    if (chunk_type, group_id) in groups:
        return groups[(chunk_type, group_id)]
    group = CommGroupInfo(chunk_type=chunk_type, id=group_id)
    groups[(chunk_type, group_id)] = group
    return group


class CommInfo(object):
    num_chunk_type = {}

    def __init__(self, chunk_type, chunk_id):
        if chunk_type not in CommInfo.num_chunk_type:
            CommInfo.num_chunk_type[chunk_type] = 0
        world_size = get_world_size()
        group_id = CommInfo.num_chunk_type[chunk_type] // world_size
        self.group = get_comm_group(chunk_type, group_id)
        self.group.elements.append(chunk_id)
        self.offset = CommInfo.num_chunk_type[chunk_type] % world_size
        CommInfo.num_chunk_type[chunk_type] += 1

    @property
    def chunk_type(self):
        return self.group.chunk_type

    @property
    def group_id(self):
        return self.group.id

    def __str__(self):
        return f"({self.group}, {self.offset})"
