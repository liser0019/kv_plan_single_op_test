# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SparseKvPlan 单算子工程 wheel 打包:通过 cmake_build 驱动顶层 CMakeLists.txt,
# 产出 _C.so 并放入 sparse_kv_plan_op 包目录。

import os
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    """调用顶层 CMakeLists.txt 构建 _C.so,并复制到包目录。"""

    def build_extension(self, ext):
        source_dir = os.path.abspath(os.path.dirname(__file__))
        build_dir = os.path.abspath(self.build_temp)
        os.makedirs(build_dir, exist_ok=True)

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format("Debug" if self.debug else "Release"),
        ]
        # 透传常用覆盖项(空值会被 CMake 侧忽略)
        for var in ("NPU_ARCH", "ASCEND_CANN_PACKAGE_HOME", "BISHENG_CXX"):
            value = os.environ.get(var, "")
            if value:
                cmake_args.append("-D{}={}".format(var, value))

        subprocess.check_call(["cmake", source_dir, *cmake_args], cwd=build_dir)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "_C", "--", "-j"], cwd=build_dir
        )

        # 产物复制到包内
        built = os.path.join(build_dir, "_C.so")
        if not os.path.exists(built):
            # 兼容 cmake 在子目录放置产物的情况
            found = subprocess.check_output(
                ["find", build_dir, "-name", "_C.so", "-type", "f"], text=True
            ).strip().splitlines()
            if not found:
                raise RuntimeError(f"_C.so not found under {build_dir}")
            built = found[0]
        target = os.path.join(self.build_lib, "sparse_kv_plan_op")
        os.makedirs(target, exist_ok=True)
        self.copy_file(built, os.path.join(target, "_C.so"))


setup(
    name="sparse_kv_plan_op",
    version="1.0.0",
    description="Single-op direct launch project for the SparseKvPlan operator (Ascend 950 / dav-3510)",
    packages=["sparse_kv_plan_op"],
    ext_modules=[Extension("_C", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
)
