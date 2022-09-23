#!/usr/bin/env bash

pushd lbdex/input
echo "bash build-run_backend.sh cpu032I eb"
bash build-run_backend.sh cpu032I eb
if [ "$?" != "0" ]; then
  echo "FAIL"
  exit 1;
fi
pushd ../verilog
make
./cpu0Is
popd

echo "bash build-run_backend.sh cpu032I el"
bash build-run_backend.sh cpu032I el
if [ "$?" != "0" ]; then
  echo "FAIL"
  exit 1;
fi
pushd ../verilog
./cpu0Is
popd

echo "bash build-run_backend.sh cpu032II eb"
bash build-run_backend.sh cpu032II eb
if [ "$?" != "0" ]; then
  echo "FAIL"
  exit 1;
fi
pushd ../verilog
./cpu0IIs
popd

echo "bash build-run_backend.sh cpu032II el"
bash build-run_backend.sh cpu032II el
if [ "$?" != "0" ]; then
  echo "FAIL"
  exit 1;
fi
pushd ../verilog
./cpu0IIs
popd

echo "PASS"
popd
