#!/usr/bin/env bash

CFILES=`ls *.ll | sort -f`
for file in $CFILES
do
  ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm $file
#  ~/llvm/debug/build/bin/llc -march=mips -relocation-model=pic -filetype=asm $file
done

