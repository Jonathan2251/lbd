#!/usr/bin/env bash

pushd input
bash clean.sh
popd
pushd verilog
make clean
popd
rm -rf output chapters preprocess tmp.txt

