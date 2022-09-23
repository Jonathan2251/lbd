#!/usr/bin/env bash

source functions.sh

sh_name=build-run_backend.sh
ARG_NUM=$#
CPU=$1
ENDIAN=$2

DEFFLAGS=""
if [ "$arg1" == cpu032II ] ; then
  DEFFLAGS=${DEFFLAGS}" -DCPU032II"
fi
echo ${DEFFLAGS}

prologue;

${CLANG} ${DEFFLAGS} -c ch_run_backend2.cpp \
-emit-llvm -o ch_run_backend2.bc
${TOOLDIR}/llc -march=cpu0${ENDIAN} -mcpu=${CPU} -relocation-model=static \
-filetype=obj ch_run_backend2.bc -o ch_run_backend2.cpu0.o
${TOOLDIR}/llvm-objdump -d ch_run_backend2.cpu0.o | tail -n +8| awk \
'{print "/* " $1 " */\t" $2 " " $3 " " $4 " " $5 "\t/* " $6"\t" $7" " $8" \
" $9" " $10 "\t*/"}' > ../verilog/cpu0.hex

ENDIAN=`${TOOLDIR}/llvm-readobj -h ch_run_backend2.cpu0.o|grep "DataEncoding"|awk '{print $2}'`
isLittleEndian;

if [ $LE == "true" ] ; then
  echo "1   /* 0: big endian, 1: little endian */" > ../verilog/cpu0.config
else
  echo "0   /* 0: big endian, 1: little endian */" > ../verilog/cpu0.config
fi
cat ../verilog/cpu0.config
