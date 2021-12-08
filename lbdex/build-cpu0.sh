#!/usr/bin/env bash

LLVM_DIR=~/llvm
LLVM_RELEASE_DIR=${LLVM_DIR}/release
LLVM_TEST_DIR=${LLVM_DIR}/test

if ! test -d ${LLVM_TEST_DIR}; then
  mkdir ${LLVM_DIR}/test
  pushd ${LLVM_TEST_DIR}
  ln -s ../llvm-project/clang clang
  ln -s ../llvm-project/llvm llvm
  popd
  cp -rf llvm/modify/llvm/* ${LLVM_TEST_DIR}/llvm/.
  cp -rf Cpu0 ${LLVM_TEST_DIR}/llvm/lib/Target/.
  cp -rf regression-test/Cpu0 ${LLVM_TEST_DIR}/llvm/test/CodeGen/.
  OS=`uname -s`
  echo "OS =" ${OS}
  pushd ${LLVM_TEST_DIR}
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=Cpu0 -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_PARALLEL_COMPILE_JOBS=4 -DLLVM_PARALLEL_LINK_JOBS=1 -G "Ninja" ../llvm
  ninja
  popd
else
  echo "${LLVM_TEST_DIR} has existed already"
  exit 1
fi

