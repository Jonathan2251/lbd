#!/usr/bin/env bash

export VERSIN=3.7.0

# Download address can be gotten from "Copy link location" of right clicking 
# mouse on firefox browser on llvm.org download page.
OS=`uname -s`
echo "OS =" ${OS}
if [ "$OS" == "Linux" ]; then
  wget http://llvm.org/releases/${VERSIN}/llvm-${VERSIN}.src.tar.xz
  wget http://llvm.org/releases/${VERSIN}/cfe-${VERSIN}.src.tar.xz
else [ "$OS" == "Darwin" ];
  curl -O http://llvm.org/releases/${VERSIN}/llvm-${VERSIN}.src.tar.xz
  curl -O http://llvm.org/releases/${VERSIN}/cfe-${VERSIN}.src.tar.xz
fi

