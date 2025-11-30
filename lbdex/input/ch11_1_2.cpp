// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch11_1_2.cpp -emit-llvm -o ch11_1_2.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=obj ch11_1_2.bc -o ch11_1_2.cpu0.o

/// start
asm("slt $2, $2, $3");
asm("beq $2, $3, 20");
asm("ori $sw, $sw, 0x0020");
asm("andi $sw, $sw, 0xffdf");
