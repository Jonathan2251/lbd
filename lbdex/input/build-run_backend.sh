#!/usr/bin/env bash

# for example:
# bash build-run_backend.sh cpu032I el
# bash build-run_backend.sh cpu032II eb

source functions.sh

sh_name=build-run_backend.sh
ARG_NUM=$#
CPU=$1
ENDIAN=$2

DEFFLAGS=""
if [ "$CPU" == cpu032II ] ; then
  DEFFLAGS=${DEFFLAGS}" -DCPU032II"
fi
echo ${DEFFLAGS}

prologue;

# ch8_2_select_global_pic.cpp just for compile build test only, without running 
# on verilog.
$CLANG ${DEFFLAGS} -target mips-unknown-linux-gnu -c ch8_2_select_global_pic.cpp \
-emit-llvm -o ch8_2_select_global_pic.bc
${TOOLDIR}/llc -march=cpu0${ENDIAN} -mcpu=${CPU} -relocation-model=pic \
-filetype=obj ch8_2_select_global_pic.bc -o ch8_2_select_global_pic.cpu0.o

$CLANG ${DEFFLAGS} -target mips-unknown-linux-gnu -c ch_run_backend.cpp \
-emit-llvm -o ch_run_backend.bc
echo "${TOOLDIR}/llc -march=cpu0${ENDIAN} -mcpu=${CPU} -relocation-model=static \
-filetype=obj -enable-cpu0-tail-calls ch_run_backend.bc -o ch_run_backend.cpu0.o"
${TOOLDIR}/llc -march=cpu0${ENDIAN} -mcpu=${CPU} -relocation-model=static \
-filetype=obj -enable-cpu0-tail-calls ch_run_backend.bc -o ch_run_backend.cpu0.o
# print must at the same line, otherwise it will spilt into 2 lines
${TOOLDIR}/llvm-objdump --section=.text -d ch_run_backend.cpu0.o | tail -n +8| awk \
'{print "/* " $1 " */\t" $2 " " $3 " " $4 " " $5 "\t/* " $6"\t" $7" " $8" " $9" " $10 "\t*/"}' \
 > ../verilog/cpu0.hex

ENDIAN=`${TOOLDIR}/llvm-readobj -h ch_run_backend.cpu0.o|grep "DataEncoding"|awk '{print $2}'`
isLittleEndian;

if [ $LE == "true" ] ; then
  echo "1   /* 0: big ENDIAN, 1: little ENDIAN */" > ../verilog/cpu0.config
else
  echo "0   /* 0: big ENDIAN, 1: little ENDIAN */" > ../verilog/cpu0.config
fi
cat ../verilog/cpu0.config
