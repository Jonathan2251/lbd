#!/usr/bin/env bash

if [ "$OS" == "Linux" ]; then
  TOOLDIR=~/llvm/test/cmake_debug_build/Debug/bin
else
  TOOLDIR=~/llvm/test/cmake_debug_build/bin
fi

rm -rf output
mkdir output

# Chapter 12
clang -target mips-unknown-linux-gnu -c lbdex/input/ch12_eh.cpp -emit-llvm \
-o ch12_eh.bc
${TOOLDIR}/llvm-dis ch12_eh.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch12_eh.ll
${TOOLDIR}/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=static -filetype=asm ch12_eh.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch12_eh.cpu0.s

clang -std=c++11 -target mips-unknown-linux-gnu -c \
lbdex/input/ch12_thread_var.cpp -emit-llvm -o ch12_thread_var.bc
${TOOLDIR}/llvm-dis ch12_thread_var.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch12_thread_var.ll
${TOOLDIR}/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch12_thread_var.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch12_thread_var.cpu0.pic.s
${TOOLDIR}/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=static -filetype=asm ch12_thread_var.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch12_thread_var.cpu0.static.s

rm -rf *.bc

