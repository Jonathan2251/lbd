#!/usr/bin/env bash

pushd input
bash clean.sh
popd
pushd verilog
bash clean.sh
popd
rm -rf chapters preprocess tmp.txt

