#!/usr/bin/env bash

VERSION=3.9.0
LLVM_DIR=~/llvm
LLVM_RELEASE_DIR=${LLVM_DIR}/release
LLVM_TEST_DIR=${LLVM_DIR}/test

if ! test -d ${LLVM_DIR}; then
  mkdir ${LLVM_DIR}
fi

if [ -e /proc/cpuinfo ]; then
    export procs=`cat /proc/cpuinfo | grep processor | wc -l`
else
    export procs=1
fi

if ! test -d ${LLVM_RELEASE_DIR}; then
  mkdir ${LLVM_RELEASE_DIR}
  tar -xf llvm-${VERSION}.src.tar.xz -C ${LLVM_RELEASE_DIR}
  mv ${LLVM_RELEASE_DIR}/llvm-${VERSION}.src ${LLVM_RELEASE_DIR}/src
  tar -xf cfe-${VERSION}.src.tar.xz -C ${LLVM_RELEASE_DIR}/src/tools
  mv ${LLVM_RELEASE_DIR}/src/tools/cfe-${VERSION}.src \
  ${LLVM_RELEASE_DIR}/src/tools/clang
  mkdir ${LLVM_RELEASE_DIR}/cmake_release_build
  pushd ${LLVM_RELEASE_DIR}/cmake_release_build
  OS=`uname -s`
  echo "OS =" ${OS}
  if [ "$OS" == "Linux" ]; then
    cmake -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../src
    make -j$procs -l$procs
  else [ "$OS" == "Darwin" ];
    cmake -DCMAKE_BUILD_TYPE=Release -G "Xcode" ../src
    xcodebuild -project "LLVM.xcodeproj"
  fi
  popd
fi

