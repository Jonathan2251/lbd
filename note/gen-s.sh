#!/usr/bin/env bash

CFILES=`ls *.ll | sort -f`
for file in $CFILES
do
  ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm $file
#  ~/llvm/release/cmake_debug_build/bin/llc -march=mips -relocation-model=pic -filetype=asm $file
done

