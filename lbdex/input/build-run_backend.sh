#!/usr/bin/env bash

source functions.sh

sh_name=build-run_backend.sh
argNum=$#
arg1=$1
arg2=$2

DEFFLAGS=""
if [ "$arg1" == cpu032II ] ; then
  DEFFLAGS=${DEFFLAGS}" -DCPU032II"
fi
echo ${DEFFLAGS}

prologue;

# ch8_5.cpp just for compile build test only, without running on verilog.
clang ${DEFFLAGS} -target mips-unknown-linux-gnu -c ch8_5.cpp \
-emit-llvm -o ch8_5.bc
${TOOLDIR}/llc -march=cpu0${endian} -mcpu=${CPU} -relocation-model=pic \
-filetype=obj ch8_5.bc -o ch8_5.cpu0.o

clang ${DEFFLAGS} -target mips-unknown-linux-gnu -c ch_run_backend.cpp \
-emit-llvm -o ch_run_backend.bc
${TOOLDIR}/llc -march=cpu0${endian} -mcpu=${CPU} -relocation-model=static \
-filetype=obj -enable-cpu0-tail-calls ch_run_backend.bc -o ch_run_backend.cpu0.o
${TOOLDIR}/llvm-objdump -d ch_run_backend.cpu0.o | tail -n +12| awk \
'{print "/* " $1 " */\t" $2 " " $3 " " $4 " " $5 "\t/* " $6"\t" $7" " $8" \
" $9" " $10 "\t*/"}' > ../verilog/cpu0.hex

if [ "$arg2" == le ] ; then
  echo "1   /* 0: big endian, 1: little endian */" > ../verilog/cpu0.config
else
  echo "0   /* 0: big endian, 1: little endian */" > ../verilog/cpu0.config
fi
cat ../verilog/cpu0.config
