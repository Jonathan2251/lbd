// clang -target mips-unknown-linux-gnu -c ch8_1_1.cpp -emit-llvm -o ch8_1_1.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch8_1_1.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm ch8_1_1.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -view-isel-dags -relocation-model=pic -filetype=asm ch8_1_1.bc -o -

/// start
int test_control1()
{
  unsigned int a = 0;
  int b = 1;
  int c = 2;
  int d = 3;
  int e = 4;
  int f = 5;
  int g = 6;
  int h = 7;
  int i = 8;
  int j = 9;
  
  if (a == 0) {
    a++; // a = 1
  }
  if (b != 0) {
    b++; // b = 2
  }
  if (c > 0) {
    c++; // c = 3
  }
  if (d >= 0) {
    d++; // d = 4
  }
  if (e < 0) {
    e++; // e = 4
  }
  if (f <= 0) {
    f++; // f = 5
  }
  if (g <= 1) {
    g++; // g = 6
  }
  if (h >= 1) {
    h++; // h = 8
  }
  if (i < h) {
    i++; // i = 8
  }
  if (a != b) {
    j++; // j = 10
  }
  
  return (a+b+c+d+e+f+g+h+i+j); // 1+2+3+4+4+5+6+8+8+10 = 51
}
