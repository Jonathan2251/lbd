#!/usr/bin/env bash

CFILES=`ls *.c | sort -f`
for file in $CFILES
do
  clang -S $file -emit-llvm
done

