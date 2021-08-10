#!/usr/bin/env bash

LLVM_DIR=~/llvm
LLVM_RELEASE_DIR=${LLVM_DIR}/release
LLVM_TEST_DIR=${LLVM_DIR}/test

if ! test -d ${LLVM_DIR}; then
  mkdir ${LLVM_DIR}
  pushd ${LLVM_DIR}
  git clone https://github.com/llvm/llvm-project.git
  cd llvm-project
  git branch -b 12.x origin/release/12.x
  git checkout e8a397203c67adbeae04763ce25c6a5ae76af52c
  cd ..
else
  echo "${LLVM_DIR} has existed already"
  exit 1
fi

if ! test -d ${LLVM_RELEASE_DIR}; then
  mkdir ${LLVM_RELEASE_DIR}
  cp -rf llvm-project/clang ${LLVM_RELEASE_DIR}
  cp -rf llvm-project/llvm ${LLVM_RELEASE_DIR}
  mkdir ${LLVM_RELEASE_DIR}/build
  pushd ${LLVM_RELEASE_DIR}/build
  OS=`uname -s`
  echo "OS =" ${OS}
  cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS="clang" -G "Unix Makefiles" ../llvm
  make -j4
  popd
fi

