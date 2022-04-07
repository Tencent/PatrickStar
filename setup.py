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

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
from patrickstar.ops.op_builder import CPUAdamBuilder


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


require_list = fetch_requirements("requirements.txt")

setup(
    name="patrickstar",
    version="0.4.6",
    description="PatrickStart library",
    long_description="PatrickStar: Parallel Training of Large Language Models via a Chunk-based Parameter Server",
    long_description_content_type="text/markdown",
    author="Tencent PatrickStar Team",
    author_email="fangjiarui123@gmail.com",
    url="https://fangjiarui.github.io/",
    install_requires=require_list,
    setup_requires=require_list,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    license="BSD",
    ext_modules=[CPUAdamBuilder().builder()],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
