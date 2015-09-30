#!/usr/bin/env bash

pushd ./lbdex
bash ./gen-chapters.sh
bash ./gen-ref-output.sh
popd

