// clang -target mips-unknown-linux-gnu -c ch7_1_vector.cpp -emit-llvm -o ch7_1_vector.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llvm-dis ch7_1_vector.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch7_1_vector.bc -o -

/// start

typedef long   vector8long   __attribute__((__vector_size__(32)));
typedef long   vector8short   __attribute__((__vector_size__(16)));
typedef bool   vector8bool   __attribute__((__ext_vector_type__(8)));

volatile vector8long vsc;
volatile vector8long vbc;

void test_cmplt (void) {
  vbc = vbc < vsc;
}
