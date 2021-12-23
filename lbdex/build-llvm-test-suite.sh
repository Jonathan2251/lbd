#!/usr/bin/env bash

# Please set your LLVM_DIR
LLVM_DIR=~/llvm

# Cpu0 set the other installed DIRs here
LLVM_SRC_DIR=${LLVM_DIR}/llvm-project
LLVM_TEST_SUITE_DIR=${LLVM_SRC_DIR}/test-suite
LLVM_RELEASE_DIR=${LLVM_DIR}/release

# ref.
# https://llvm.org/docs/TestSuiteGuide.html
# https://github.com/mollybuild/RISCV-Measurement/blob/master/Build-RISCV-LLVM-and-run-testsuite.md

### Prerequisites
# On Ubuntu,
# sudo apt-get install tcl tk tcl-dev tk-dev
# On macos,
# brew install tcl-tk

build() {
  sed -i 's/^add_subdirectory(XRay)/#add_subdirectory(XRay)/g' test-suite/MicroBenchmarks/CMakeLists.txt
  mkdir test-suite-build
  cd test-suite-build
  cmake -DCMAKE_C_COMPILER=clang -C../test-suite/cmake/caches/O3.cmake -DCMAKE_C_FLAGS=-fPIE -DCMAKE_CXX_FLAGS=-fPIE ../test-suite
  make
}

if ! test -d ${LLVM_TEST_SUITE_DIR}; then
  pushd ${LLVM_SRC_DIR}
  git clone https://github.com/llvm/llvm-test-suite.git test-suite
  popd
  pushd ${LLVM_RELEASE_DIR}
  ln -s ../llvm-project/test-suite test-suite
  build;
else
#  pushd ${LLVM_RELEASE_DIR}
#  build;
  echo "${LLVM_TEST_SUITE_DIR} has existed already"
  exit 1
fi
