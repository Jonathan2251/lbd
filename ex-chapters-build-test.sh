#!/usr/bin/env bash
OS=`uname -s`
echo "OS =" ${OS}
PWD=`pwd`

if [ "$OS" == "Linux" ]; then
  TOOLDIR=~/llvm/test
else
  TOOLDIR=~/llvm/test
fi

if [ -e /proc/cpuinfo ]; then
    procs=`cat /proc/cpuinfo | grep processor | wc -l`
else
    procs=1
fi

rm -rf ${TOOLDIR}/cmake_debug_build/* ${TOOLDIR}/src/lib/Target/Cpu0/*
pushd ./lbdex
bash gen-chapters.sh
popd

# Chapter 2
cp -rf lbdex/chapters/Chapter2/* ${TOOLDIR}/src/lib/Target/Cpu0/.
pushd ${TOOLDIR}/cmake_debug_build
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_FLAGS=-std=c++11 -DCMAKE_BUILD_TYPE=Debug \
-DLLVM_TARGETS_TO_BUILD=Cpu0 -G "Unix Makefiles" ../src
result=`make -j$procs -l$procs`
echo "result: ${result}"
if [ ${result} ]; then
  exit 1;
fi
popd

allch="3_1 3_2 3_3 3_4 3_5 4_1 4_2 5_1 6_1 7_1 8_1 8_2 9_1 9_2 9_3 10_1 \
11_1 11_2 12_1"
# All other Chapters
for ch in $allch
do
  rm -rf ${TOOLDIR}/src/lib/Target/Cpu0/*
  cp -rf lbdex/chapters/Chapter$ch/* ${TOOLDIR}/src/lib/Target/Cpu0/.
  echo "cp -rf lbdex/Chapter$ch/* ${TOOLDIR}/src/lib/Target/Cpu0/."
  pushd ${TOOLDIR}/cmake_debug_build
  result=`make -j$procs -l$procs`
  echo "result: ${result}"
  popd
  if [ ${result} ]; then
    exit 1;
  fi
done

