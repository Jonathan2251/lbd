#!/usr/bin/env bash

export VERSION=3.7.0
export LLVM_DIR=~/llvm
export LLVM_RELEASE_DIR=${LLVM_DIR}/release
export LLVM_TEST_DIR=${LLVM_DIR}/test

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

if ! test -d ${LLVM_TEST_DIR}; then
  mkdir ${LLVM_TEST_DIR}
  tar -xf llvm-${VERSION}.src.tar.xz -C ${LLVM_TEST_DIR}
  mv ${LLVM_TEST_DIR}/llvm-${VERSION}.src ${LLVM_TEST_DIR}/src
  cp -rf ../src/modify/src/* ${LLVM_TEST_DIR}/src/.
  cp -rf ../Cpu0 ${LLVM_TEST_DIR}/src/lib/Target/.
  mkdir ${LLVM_TEST_DIR}/cmake_debug_build
  pushd ${LLVM_TEST_DIR}/cmake_debug_build
  if [ "$OS" == "Linux" ]; then
    cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
   -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=Cpu0 -G "Unix Makefiles" \
   ../src
    make -j$procs -l$procs
  else [ "$OS" == "Darwin" ];
    cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=Cpu0 -G "Xcode" ../src
    xcodebuild -project "LLVM.xcodeproj"
  fi
  popd
fi
