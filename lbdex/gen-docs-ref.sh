#!/usr/bin/env bash

pushd ./lbdex
bash ./gen-chapters.sh
# disable since removing output files of llvm-ir and asm from source/*.rst
#bash ./gen-ref-output.sh
popd

