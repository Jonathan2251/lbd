#!/usr/bin/env bash

OS=`uname -s`
echo "OS =" ${OS}
PWD=`pwd`

if [ "$OS" == "Linux" ]; then
  TOOLDIR=~/llvm/test
else
  TOOLDIR=~/llvm/test
fi

if [ -e /proc/cpuinfo ]; then
    procs=`cat /proc/cpuinfo | grep processor | wc -l`
else
    procs=1
fi

rm -rf ${TOOLDIR}/cmake_debug_build/* ${TOOLDIR}/src/lib/Target/Cpu0/*
rm -rf output
mkdir output

cp -rf lbdex/Cpu0/* ${TOOLDIR}/src/lib/Target/Cpu0/.

# Chapter 3
pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH3_5" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_FLAGS=-std=c++11 -DCMAKE_BUILD_TYPE=Debug \
-DLLVM_TARGETS_TO_BUILD=Cpu0 -G "Unix Makefiles" ../src
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch3.cpp -emit-llvm \
-o ch3.bc
# Replace \t with "  ", Fold 80 characters
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch3.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch3.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic \
-filetype=asm ch3.bc -o - |awk '{gsub("\t","  ",$0); print;}' |fold -w 80 \
|awk '{gsub("\t","  ",$0); print;}' > output/ch3.pic.cpu0.s

clang -target mips-unknown-linux-gnu -c lbdex/input/ch3_2.cpp -emit-llvm \
-o ch3_2.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch3_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch3_2.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic \
-filetype=asm ch3_2.bc -o - |awk '{gsub("\t","  ",$0); print;}' |fold -w 80 \
|awk '{gsub("\t","  ",$0); print;}' > output/ch3_2.pic.cpu0.s

# Chapter 4
pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH4_1" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch4_1_1.cpp -emit-llvm \
-o ch4_1.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch4_1_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch4_1_1.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic \
-filetype=asm ch4_1_1.bc -o - |awk '{gsub("\t","  ",$0); print;}' |fold -w 80 \
|awk '{gsub("\t","  ",$0); print;}' > output/ch4_1_1.pic.cpu0.s

pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH4_2" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch4_2.cpp -emit-llvm \
-o ch4_2.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch4_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch4_2.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic \
-filetype=asm ch4_2.bc -o - |awk '{gsub("\t","  ",$0); print;}' |fold -w 80 \
|awk '{gsub("\t","  ",$0); print;}' > output/ch4_2.pic.cpu0.s

# Chapter 5
pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH5_1" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic \
-filetype=obj ch4_1.bc -o - |awk '{gsub("\t","  ",$0); print;}' |fold -w 80 \
|awk '{gsub("\t","  ",$0); print;}' > output/ch4_1.cpu0.o
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic \
-filetype=obj ch4_2.bc -o - |awk '{gsub("\t","  ",$0); print;}' |fold -w 80 \
|awk '{gsub("\t","  ",$0); print;}' > output/ch4_2.cpu0el.o

# Chapter 6
pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH6_1" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch6_1.cpp -emit-llvm \
-o ch6_1.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch6_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch6_1.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic \
-cpu0-use-small-section=true -filetype=asm ch6_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch6_1.pic.t.cpu0.s
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic \
-cpu0-use-small-section=false -filetype=asm ch6_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch6_1.pic.f.cpu0.s
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=static \
-cpu0-use-small-section=true -filetype=asm ch6_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch6_1.static.t.cpu0.s
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=static \
-cpu0-use-small-section=false -filetype=asm ch6_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}'|fold -w 80 > output/ch6_1.static.f.cpu0.s

# Chapter 7
pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH7_1" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch7_1.cpp -emit-llvm \
-o ch7_1.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch7_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_1.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch7_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_1.pic.cpu0.s

clang -target mips-unknown-linux-gnu -c lbdex/input/ch7_2.cpp -emit-llvm \
-o ch7_2.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch7_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_2.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch7_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 |awk > output/ch7_2.pic.cpu0.s

clang -target mips-unknown-linux-gnu -c lbdex/input/ch7_2_2.cpp -emit-llvm \
-o ch7_2_2.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch7_2_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_2_2.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch7_2_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 |awk > output/ch7_2_2.pic.cpu0.s

${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch7_3.ll -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_3.pic.cpu0.s

clang -target mips-unknown-linux-gnu -c lbdex/input/ch7_4.cpp -emit-llvm \
-o ch7_4.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch7_4.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_4.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch7_4.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_4.pic.cpu0.s

clang -target mips-unknown-linux-gnu -c lbdex/input/ch7_5.cpp -emit-llvm \
-o ch7_5.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch7_5.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_5.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch7_5.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_5.pic.cpu0.s

clang -target mips-unknown-linux-gnu -c lbdex/input/ch7_5_2.cpp -emit-llvm \
-o ch7_5_2.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch7_5_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_5_2.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch7_5_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_5_2.pic.cpu0.s

clang -target mips-unknown-linux-gnu -c lbdex/input/ch7_1_vector.cpp -emit-llvm \
-o ch7_1_vector.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch7_1_vector.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_1_vector.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch7_1_vector.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch7_1_vector.pic.cpu0.s

# Chapter 8
pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH8_2" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch8_2.cpp -emit-llvm \
-o ch8_2.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch8_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch8_2.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=static -filetype=asm -stats ch8_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch8_2.static.cpu0.s

clang -target mips-unknown-linux-gnu -c lbdex/input/ch8_2_longbranch.cpp -emit-llvm \
-o ch8_2_longbranch.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch8_2_longbranch.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch8_2_longbranch.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032II \
-relocation-model=pic -filetype=asm -force-cpu0-long-branch ch8_2_longbranch.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch8_2_longbranch.pic.cpu0.s

clang -O1 -target mips-unknown-linux-gnu -c lbdex/input/ch8_3.cpp -emit-llvm \
-o ch8_3.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch8_3.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch8_3.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch8_3.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch8_3.pic.cpu0.s

# Chapter 9
pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH9_2" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch9_1.cpp -emit-llvm \
-o ch9_1.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch9_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch9_1.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch9_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch9_1.pic.cpu0.s

pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH9_3" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch9_3_bswap.cpp -emit-llvm \
-o ch9_3_bswap.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch9_3_bswap.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch9_3_bswap.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch9_3_bswap.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch9_3_bswap.pic.cpu0.s
clang -target mips-unknown-linux-gnu -c lbdex/input/ch9_3_detect_exception.cpp -emit-llvm \
-o ch9_3_detect_exception.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch9_3_detect_exception.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch9_3_detect_exception.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch9_3_detect_exception.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch9_3_detect_exception.pic.cpu0.s
clang -target mips-unknown-linux-gnu -c lbdex/input/ch9_3_frame_return_addr.cpp -emit-llvm \
-o ch9_3_frame_return_addr.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch9_3_frame_return_addr.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch9_3_frame_return_addr.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=asm ch9_3_frame_return_addr.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch9_3_frame_return_addr.pic.cpu0.s

# Chapter 10
pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH10_1" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch8_1_1.cpp -emit-llvm \
-o ch8_1_1.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch8_1_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch8_1_1.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=obj -stats ch8_1_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch8_1_1.cpu0.o
${TOOLDIR}/cmake_debug_build/bin/llvm-objdump -d - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch8_1_1.cpu0.o.hex

# Chapter 11
pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH11_1" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch11_1.cpp -emit-llvm \
-o ch11_1.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch11_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch11_1.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=pic -filetype=obj -stats ch11_1.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch11_1.cpu0.o
${TOOLDIR}/cmake_debug_build/bin/llvm-objdump -d - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch11_1.cpu0.o.hex

pushd ${TOOLDIR}/cmake_debug_build
echo "#define CH       CH11_2" > ../src/lib/Target/Cpu0/Cpu0SetChapter.h
make -j$procs -l$procs
popd
clang -target mips-unknown-linux-gnu -c lbdex/input/ch11_2.cpp -emit-llvm \
-o ch11_2.bc
${TOOLDIR}/cmake_debug_build/bin/llvm-dis ch11_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch11_2.ll
${TOOLDIR}/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032I \
-relocation-model=static -filetype=obj -stats ch11_2.bc -o - |awk \
'{gsub("\t","  ",$0); print;}'|fold -w 80 > output/ch11_2.cpu0.o
${TOOLDIR}/cmake_debug_build/bin/llvm-objdump -d - |awk \
'{gsub("\t","  ",$0); print;}' |fold -w 80 > output/ch11_2.cpu0.o.hex

