#!/usr/bin/env bash

# Please set your LLVM_DIR
LLVM_DIR=~/llvm

# Cpu0 set the other installed DIRs here
LLVM_SRC_DIR=${LLVM_DIR}/llvm-project
LLVM_TEST_SUITE_DIR=${LLVM_SRC_DIR}/test-suite
LLVM_DEBUG_DIR=${LLVM_DIR}/debug

# ref.
# https://llvm.org/docs/TestSuiteGuide.html
# https://github.com/mollybuild/RISCV-Measurement/blob/master/Build-RISCV-LLVM-and-run-testsuite.md

### Prerequisites
# On Ubuntu,
# sudo apt-get install tcl tk tcl-dev tk-dev
# On macos,
# brew install tcl-tk
# ${LLVM_DEBUG_DIR}/build: build with clang and compiler-rt, -DLLVM_ENABLE_PROJECTS="clang;compiler-rt" --> ref. https://github.com/Jonathan2251/lbd/blob/master/lbdex/install_llvm/build-llvm.sh

build() {
  cd test-suite-build
  cmake -DCMAKE_C_COMPILER=${LLVM_DEBUG_DIR}/build/bin/clang -C../test-suite/cmake/caches/O3.cmake -DCMAKE_C_FLAGS=-fPIE -DCMAKE_CXX_FLAGS=-fPIE ../test-suite
  make
}

if ! test -d ${LLVM_TEST_SUITE_DIR}; then
  pushd ${LLVM_SRC_DIR}
  git clone https://github.com/llvm/llvm-test-suite.git test-suite
  cd test-suite
  git checkout -b 12.x origin/release/12.x
  popd
  pushd ${LLVM_DEBUG_DIR}
  ln -s ../llvm-project/test-suite test-suite
#  sed 's/^add_subdirectory(XRay)/#add_subdirectory(XRay)/g' test-suite/MicroBenchmarks/CMakeLists.txt
  rm -rf test-suite-build
  mkdir test-suite-build
  build;
else
  pushd ${LLVM_DEBUG_DIR}
  build;
  echo "${LLVM_TEST_SUITE_DIR} has existed already"
  exit 1
fi
