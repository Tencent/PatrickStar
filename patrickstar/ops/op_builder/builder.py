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

"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import sys
import time
from pathlib import Path
import subprocess
import shlex
import shutil
import tempfile
import distutils.ccompiler
import distutils.log
import distutils.sysconfig
from distutils.errors import CompileError, LinkError
from abc import ABC, abstractmethod

YELLOW = "\033[93m"
END = "\033[0m"
WARNING = f"{YELLOW} [WARNING] {END}"

DEFAULT_TORCH_EXTENSION_PATH = "/tmp/torch_extensions"
DEFAULT_COMPUTE_CAPABILITIES = "6.0;6.1;7.0"

try:
    import torch
except ImportError:
    print(
        f"{WARNING} unable to import torch, please install it if you want to pre-compile any deepspeed ops."
    )


def installed_cuda_version():
    import torch.utils.cpp_extension

    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    assert (
        cuda_home is not None
    ), "CUDA_HOME does not exist, unable to compile CUDA op(s)"
    # Ensure there is not a cuda version mismatch between torch and nvcc compiler
    output = subprocess.check_output(
        [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output_split = output.split()
    release_idx = output_split.index("release")
    release = output_split[release_idx + 1].replace(",", "").split(".")
    # Ignore patch versions, only look at major + minor
    cuda_major, cuda_minor = release[:2]
    return int(cuda_major), int(cuda_minor)


def get_default_compute_capatabilities():
    compute_caps = DEFAULT_COMPUTE_CAPABILITIES
    import torch.utils.cpp_extension

    if (
        torch.utils.cpp_extension.CUDA_HOME is not None
        and installed_cuda_version()[0] >= 11
    ):
        if installed_cuda_version()[0] == 11 and installed_cuda_version()[1] == 0:
            # Special treatment of CUDA 11.0 because compute_86 is not supported.
            compute_caps += ";8.0"
        else:
            compute_caps += ";8.0;8.6"
    return compute_caps


# list compatible minor CUDA versions - so that for example pytorch built with cuda-11.0 can be used
# to build deepspeed and system-wide installed cuda 11.2
cuda_minor_mismatch_ok = {
    10: ["10.0", "10.1", "10.2"],
    11: ["11.0", "11.1", "11.2", "11.3"],
}


def assert_no_cuda_mismatch():
    cuda_major, cuda_minor = installed_cuda_version()
    sys_cuda_version = f"{cuda_major}.{cuda_minor}"
    torch_cuda_version = ".".join(torch.version.cuda.split(".")[:2])
    # This is a show-stopping error, should probably not proceed past this
    if sys_cuda_version != torch_cuda_version:
        if (
            cuda_major in cuda_minor_mismatch_ok
            and sys_cuda_version in cuda_minor_mismatch_ok[cuda_major]
            and torch_cuda_version in cuda_minor_mismatch_ok[cuda_major]
        ):
            print(
                f"Installed CUDA version {sys_cuda_version} does not match the "
                f"version torch was compiled with {torch.version.cuda} "
                "but since the APIs are compatible, accepting this combination"
            )
            return
        raise Exception(
            f"Installed CUDA version {sys_cuda_version} does not match the "
            f"version torch was compiled with {torch.version.cuda}, unable to compile "
            "cuda/cpp extensions without a matching cuda version."
        )


def assert_torch_info(torch_info):
    install_torch_version = torch_info["version"]
    install_cuda_version = torch_info["cuda_version"]

    current_cuda_version = ".".join(torch.version.cuda.split(".")[:2])
    current_torch_version = ".".join(torch.__version__.split(".")[:2])

    if (
        install_cuda_version != current_cuda_version
        or install_torch_version != current_torch_version
    ):
        raise RuntimeError(
            "PyTorch and CUDA version mismatch! DeepSpeed ops were compiled and installed "
            "with a different version than what is being used at runtime. Please re-install "
            f"DeepSpeed or switch torch versions. DeepSpeed install versions: "
            f"torch={install_torch_version}, cuda={install_cuda_version}, runtime versions:"
            f"torch={current_torch_version}, cuda={current_cuda_version}"
        )


class OpBuilder(ABC):
    def __init__(self, name):
        self.name = name
        self.jit_mode = False

    @abstractmethod
    def absolute_name(self):
        """
        Returns absolute build path for cases where the op is pre-installed, e.g., deepspeed.ops.adam.cpu_adam
        will be installed as something like: deepspeed/ops/adam/cpu_adam.so
        """
        pass

    @abstractmethod
    def sources(self):
        """
        Returns list of source files for your op, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        """
        pass

    def include_paths(self):
        """
        Returns list of include paths, relative to root of deepspeed package (i.e., DeepSpeed/deepspeed)
        """
        return []

    def nvcc_args(self):
        """
        Returns optional list of compiler flags to forward to nvcc when building CUDA sources
        """
        return []

    def cxx_args(self):
        """
        Returns optional list of compiler flags to forward to the build
        """
        return []

    def is_compatible(self):
        """
        Check if all non-python dependencies are satisfied to build this op
        """
        return True

    def extra_ldflags(self):
        return []

    def libraries_installed(self, libraries):
        valid = False
        for lib in libraries:
            result = subprocess.Popen(
                f"dpkg -l {lib}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            valid = valid or result.wait() == 0
        return valid

    def has_function(self, funcname, libraries, verbose=False):
        """
        Test for existence of a function within a tuple of libraries.

        This is used as a smoke test to check whether a certain library is avaiable.
        As a test, this creates a simple C program that calls the specified function,
        and then distutils is used to compile that program and link it with the specified libraries.
        Returns True if both the compile and link are successful, False otherwise.
        """
        tempdir = None  # we create a temporary directory to hold various files
        filestderr = None  # handle to open file to which we redirect stderr
        oldstderr = None  # file descriptor for stderr
        try:
            # Echo compile and link commands that are used.
            if verbose:
                distutils.log.set_verbosity(1)

            # Create a compiler object.
            compiler = distutils.ccompiler.new_compiler(verbose=verbose)

            # Configure compiler and linker to build according to Python install.
            distutils.sysconfig.customize_compiler(compiler)

            # Create a temporary directory to hold test files.
            tempdir = tempfile.mkdtemp()

            # Define a simple C program that calls the function in question
            prog = (
                "void %s(void); int main(int argc, char** argv) { %s(); return 0; }"
                % (funcname, funcname)
            )

            # Write the test program to a file.
            filename = os.path.join(tempdir, "test.c")
            with open(filename, "w") as f:
                f.write(prog)

            # Redirect stderr file descriptor to a file to silence compile/link warnings.
            if not verbose:
                filestderr = open(os.path.join(tempdir, "stderr.txt"), "w")
                oldstderr = os.dup(sys.stderr.fileno())
                os.dup2(filestderr.fileno(), sys.stderr.fileno())

            # Attempt to compile the C program into an object file.
            cflags = shlex.split(os.environ.get("CFLAGS", ""))
            objs = compiler.compile(
                [filename], extra_preargs=self.strip_empty_entries(cflags)
            )

            # Attempt to link the object file into an executable.
            # Be sure to tack on any libraries that have been specified.
            ldflags = shlex.split(os.environ.get("LDFLAGS", ""))
            compiler.link_executable(
                objs,
                os.path.join(tempdir, "a.out"),
                extra_preargs=self.strip_empty_entries(ldflags),
                libraries=libraries,
            )

            # Compile and link succeeded
            return True

        except CompileError:
            return False

        except LinkError:
            return False

        except:  # noqa E722
            return False

        finally:
            # Restore stderr file descriptor and close the stderr redirect file.
            if oldstderr is not None:
                os.dup2(oldstderr, sys.stderr.fileno())
            if filestderr is not None:
                filestderr.close()

            # Delete the temporary directory holding the test program and stderr files.
            if tempdir is not None:
                shutil.rmtree(tempdir)

    def strip_empty_entries(self, args):
        """
        Drop any empty strings from the list of compile and link flags
        """
        return [x for x in args if len(x) > 0]

    def cpu_arch(self):
        if not self.command_exists("lscpu"):
            self.warning(
                f"{self.name} attempted to query 'lscpu' to detect the CPU architecture. "
                "However, 'lscpu' does not appear to exist on "
                "your system, will fall back to use -march=native."
            )
            return "-march=native"

        result = subprocess.check_output("lscpu", shell=True)
        result = result.decode("utf-8").strip().lower()
        if "ppc64le" in result:
            # gcc does not provide -march on PowerPC, use -mcpu instead
            return "-mcpu=native"
        return "-march=native"

    def simd_width(self):
        if not self.command_exists("lscpu"):
            self.warning(
                f"{self.name} attempted to query 'lscpu' to detect the existence "
                "of AVX instructions. However, 'lscpu' does not appear to exist on "
                "your system, will fall back to non-vectorized execution."
            )
            return "-D__SCALAR__"

        result = subprocess.check_output("lscpu", shell=True)
        result = result.decode("utf-8").strip().lower()
        if "genuineintel" in result:
            if "avx512" in result:
                return "-D__AVX512__"
            elif "avx2" in result:
                return "-D__AVX256__"
        return "-D__SCALAR__"

    def python_requirements(self):
        """
        Override if op wants to define special dependencies, otherwise will
        take self.name and load requirements-<op-name>.txt if it exists.
        """
        path = f"requirements/requirements-{self.name}.txt"
        requirements = []
        if os.path.isfile(path):
            with open(path, "r") as fd:
                requirements = [r.strip() for r in fd.readlines()]
        return requirements

    def command_exists(self, cmd):
        if "|" in cmd:
            cmds = cmd.split("|")
        else:
            cmds = [cmd]
        valid = False
        for cmd in cmds:
            result = subprocess.Popen(f"type {cmd}", stdout=subprocess.PIPE, shell=True)
            valid = valid or result.wait() == 0

        if not valid and len(cmds) > 1:
            print(
                f"{WARNING} {self.name} requires one of the following commands '{cmds}', but it does not exist!"
            )
        elif not valid and len(cmds) == 1:
            print(
                f"{WARNING} {self.name} requires the '{cmd}' command, but it does not exist!"
            )
        return valid

    def warning(self, msg):
        print(f"{WARNING} {msg}")

    def deepspeed_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            return os.path.join(
                Path(__file__).parent.parent.parent.parent.absolute(), code_path
            )

    def builder(self):
        from torch.utils.cpp_extension import CppExtension

        return CppExtension(
            name=self.absolute_name(),
            sources=self.strip_empty_entries(self.sources()),
            include_dirs=self.strip_empty_entries(self.include_paths()),
            extra_compile_args={"cxx": self.strip_empty_entries(self.cxx_args())},
            extra_link_args=self.strip_empty_entries(self.extra_ldflags()),
        )

    def load(self, verbose=True):
        return self.jit_load(verbose)

    def jit_load(self, verbose=True):
        if not self.is_compatible():
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible due to hardware/software issue."
            )
        try:
            import ninja  # noqa F401
        except ImportError:
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to ninja not being installed."
            )

        if isinstance(self, CUDAOpBuilder):
            assert_no_cuda_mismatch()

        self.jit_mode = True
        from torch.utils.cpp_extension import load

        # Ensure directory exists to prevent race condition in some cases
        ext_path = os.path.join(
            os.environ.get("TORCH_EXTENSIONS_DIR", DEFAULT_TORCH_EXTENSION_PATH),
            self.name,
        )
        os.makedirs(ext_path, exist_ok=True)

        start_build = time.time()
        sources = [self.deepspeed_src_path(path) for path in self.sources()]
        extra_include_paths = [
            self.deepspeed_src_path(path) for path in self.include_paths()
        ]
        op_module = load(
            name=self.name,
            sources=self.strip_empty_entries(sources),
            extra_include_paths=self.strip_empty_entries(extra_include_paths),
            extra_cflags=self.strip_empty_entries(self.cxx_args()),
            extra_cuda_cflags=self.strip_empty_entries(self.nvcc_args()),
            extra_ldflags=self.strip_empty_entries(self.extra_ldflags()),
            verbose=verbose,
        )
        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")
        return op_module


class CUDAOpBuilder(OpBuilder):
    def compute_capability_args(self, cross_compile_archs=None):
        """
        Returns nvcc compute capability compile flags.

        1. `TORCH_CUDA_ARCH_LIST` takes priority over `cross_compile_archs`.
        2. If neither is set default compute capabilities will be used
        3. Under `jit_mode` compute capabilities of all visible cards will be used plus PTX

        Format:

        - `TORCH_CUDA_ARCH_LIST` may use ; or whitespace separators. Examples:

        TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6" pip install ...
        TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" pip install ...

        - `cross_compile_archs` uses ; separator.

        """

        ccs = []
        if self.jit_mode:
            # Compile for underlying architectures since we know those at runtime
            for i in range(torch.cuda.device_count()):
                cc_major, cc_minor = torch.cuda.get_device_capability(i)
                cc = f"{cc_major}.{cc_minor}"
                if cc not in ccs:
                    ccs.append(cc)
            ccs = sorted(ccs)
            ccs[-1] += "+PTX"
        else:
            # Cross-compile mode, compile for various architectures
            # env override takes priority
            cross_compile_archs_env = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cross_compile_archs_env is not None:
                if cross_compile_archs is not None:
                    print(
                        f"{WARNING} env var `TORCH_CUDA_ARCH_LIST={cross_compile_archs_env}` "
                        f"overrides `cross_compile_archs={cross_compile_archs}`"
                    )
                cross_compile_archs = cross_compile_archs_env.replace(" ", ";")
            else:
                if cross_compile_archs is None:
                    cross_compile_archs = get_default_compute_capatabilities()
            ccs = cross_compile_archs.split(";")

        args = []
        for cc in ccs:
            num = cc[0] + cc[2]
            args.append(f"-gencode=arch=compute_{num},code=sm_{num}")
            if cc.endswith("+PTX"):
                args.append(f"-gencode=arch=compute_{num},code=compute_{num}")

        return args

    def version_dependent_macros(self):
        # Fix from apex that might be relevant for us as well, related to https://github.com/NVIDIA/apex/issues/456
        torch_major = int(torch.__version__.split(".")[0])
        torch_minor = int(torch.__version__.split(".")[1])
        version_ge_1_1 = []
        if (torch_major > 1) or (torch_major == 1 and torch_minor > 0):
            version_ge_1_1 = ["-DVERSION_GE_1_1"]
        version_ge_1_3 = []
        if (torch_major > 1) or (torch_major == 1 and torch_minor > 2):
            version_ge_1_3 = ["-DVERSION_GE_1_3"]
        version_ge_1_5 = []
        if (torch_major > 1) or (torch_major == 1 and torch_minor > 4):
            version_ge_1_5 = ["-DVERSION_GE_1_5"]
        return version_ge_1_1 + version_ge_1_3 + version_ge_1_5

    def is_compatible(self):
        return super().is_compatible()

    def builder(self):
        from torch.utils.cpp_extension import CUDAExtension

        assert_no_cuda_mismatch()
        return CUDAExtension(
            name=self.absolute_name(),
            sources=self.strip_empty_entries(self.sources()),
            include_dirs=self.strip_empty_entries(self.include_paths()),
            libraries=self.strip_empty_entries(self.libraries_args()),
            extra_compile_args={
                "cxx": self.strip_empty_entries(self.cxx_args()),
                "nvcc": self.strip_empty_entries(self.nvcc_args()),
            },
        )

    def cxx_args(self):
        if sys.platform == "win32":
            return ["-O2"]
        else:
            return ["-O3", "-std=c++14", "-g", "-Wno-reorder"]

    def nvcc_args(self):
        args = [
            "-O3",
            "--use_fast_math",
            "-std=c++17" if sys.platform == "win32" else "-std=c++14",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ]

        return args + self.compute_capability_args()

    def libraries_args(self):
        if sys.platform == "win32":
            return ["cublas", "curand"]
        else:
            return []
