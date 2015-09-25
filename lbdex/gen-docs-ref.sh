#!/usr/bin/env bash

CURR_DIR=`pwd`
echo "aaaaaaaaa"
echo "curr_dir=$CURR_DIR"
pushd ./lbdex
bash ./gen-chapters.sh
bash ./gen-ref-output.sh
popd

