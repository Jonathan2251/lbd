#!/usr/bin/env bash

LLVM_DIR=~/llvm
LLVM_RELEASE_DIR=${LLVM_DIR}/release
LLVM_TEST_DIR=${LLVM_DIR}/test

pushd ${LLVM_DIR}

if ! test -d ${LLVM_DIR}/llvm-project; then
  git clone git://github.com/llvm/llvm-project.git
  cd llvm-project
  git checkout -b 12.x origin/release/12.x
  git checkout e8a397203c67adbeae04763ce25c6a5ae76af52c
  cd ..
else
  echo "${LLVM_DIR}/llvm-project has existed already"
#  exit 1
fi

if ! test -d ${LLVM_RELEASE_DIR}; then
  mkdir ${LLVM_RELEASE_DIR}
  cp -rf llvm-project/clang ${LLVM_RELEASE_DIR}
  cp -rf llvm-project/llvm ${LLVM_RELEASE_DIR}
# build compiler-rt for llvm-test-suite
  cp -rf llvm-project/compiler-rt ${LLVM_RELEASE_DIR}
  mkdir ${LLVM_RELEASE_DIR}/build
  pushd ${LLVM_RELEASE_DIR}/build
  OS=`uname -s`
  echo "OS =" ${OS}
  cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS="clang;compiler-rt" \
  -DLLVM_PARALLEL_COMPILE_JOBS=4 -DLLVM_PARALLEL_LINK_JOBS=1 -G "Ninja" ../llvm
  ninja
  popd
fi

popd
echo "Please remember to add ${LLVM_RELEASE_DIR}/bin to variable \${PATH} to your \
  environment for clang++, clang."
