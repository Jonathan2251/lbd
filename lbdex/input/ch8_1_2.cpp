// clang -target mips-unknown-linux-gnu -c ch8_1_2.cpp -emit-llvm -o ch8_1_2.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch8_1_2.bc -o -

/// start
int main()
{
  int a = 5;
  int b = 0;
  int* p = &a;
  
  b = !(*p);
  if (b == 0) {
    a = a + b;
  } else if (b < 0) {
    a = a--;
  } else if (b > 0) {
    a = a++;
  } else if (b != 0) {
    a = a - b;
  }
  return a;
}
