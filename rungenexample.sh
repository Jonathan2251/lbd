#!/usr/bin/env bash

bash ./clean.sh
rm -rf lbdex.tar.gz
CURR_DIR=$(pwd)
echo ${CURR_DIR}
tar -zcvf lbdex.tar.gz lbdex
pushd lbdex
bash ./gen-chapters.sh
popd

