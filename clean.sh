#!/usr/bin/env bash

pushd lbdex
bash clean.sh
popd
rm -rf build lbdex.tar.gz
rm -f `find . -name \*~`
rm -f `find . -name .DS_Store`