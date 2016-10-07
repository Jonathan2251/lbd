// clang -target mips-unknown-linux-gnu -c ch7_1_vector.cpp -emit-llvm -o ch7_1_vector.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llvm-dis ch7_1_vector.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm ch7_1_vector.bc -o -

/// start

typedef long   vector8long   __attribute__((__vector_size__(32)));
typedef long   vector8short   __attribute__((__vector_size__(16)));


int test_cmplt_short() {
  volatile vector8short a0 = {0, 1, 2, 3};
  volatile vector8short b0 = {2, 2, 2, 2};
  volatile vector8short c0;
  c0 = a0 < b0; // c0[0] = -2147483647=0x80000001, c0[1] = -2147483647=0x80000001, c0[2] = 0, c0[3] = 0
  
  return (int)(c0[0]+c0[1]+c0[2]+c0[3]); // 2
}

int test_cmplt_long() {
  volatile vector8long a0 = {2, 2, 2, 2, 1, 1, 1, 1};
  volatile vector8long b0 = {1, 1, 1, 1, 2, 2, 2, 2};
  volatile vector8long c0;
  c0 = a0 < b0; // c0[0..3] = {0, 0, ...}, c0[4..7] = {-2147483647=0x80000001, ...}
  
  return (c0[0]+c0[1]+c0[2]+c0[3]+c0[4]+c0[5]+c0[6]+c0[7]); //4
}
