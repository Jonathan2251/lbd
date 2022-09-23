#!/usr/bin/env bash

# If your memory is large enough, add swap size ubuntu as follows,
# 1. search "add swap size ubuntu"
# 2. https://bogdancornianu.com/change-swap-size-in-ubuntu/
# 3. add to boot by change to /swapfile                                 swap            swap    default              0       0
# https://linuxize.com/post/how-to-add-swap-space-on-ubuntu-18-04/
# 4. reboot

LLVM_DIR=~/llvm
LLVM_DEBUG_DIR=${LLVM_DIR}/debug
LLVM_TEST_DIR=${LLVM_DIR}/test

pushd ${LLVM_DIR}

if ! test -d ${LLVM_DIR}/llvm-project; then
  git clone https://github.com/llvm/llvm-project.git
  cd llvm-project
  git checkout -b 12.x origin/release/12.x
  git checkout e8a397203c67adbeae04763ce25c6a5ae76af52c
  cd ..
else
  echo "${LLVM_DIR}/llvm-project has existed already"
#  exit 1
fi

if ! test -d ${LLVM_DEBUG_DIR}; then
  mkdir ${LLVM_DEBUG_DIR}
  cp -rf llvm-project/clang ${LLVM_DEBUG_DIR}
  cp -rf llvm-project/llvm ${LLVM_DEBUG_DIR}
# build compiler-rt for llvm-test-suite
  cp -rf llvm-project/compiler-rt ${LLVM_DEBUG_DIR}
  mkdir ${LLVM_DEBUG_DIR}/build
  pushd ${LLVM_DEBUG_DIR}/build
  OS=`uname -s`
  echo "OS =" ${OS}
  cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_PARALLEL_COMPILE_JOBS=4 -DLLVM_PARALLEL_LINK_JOBS=1 -G "Ninja" ../llvm
  ninja
  popd
fi

popd
echo "Please remember to add ${LLVM_DEBUG_DIR}/build/bin to variable \${PATH} to your \
  environment for clang++, clang."
