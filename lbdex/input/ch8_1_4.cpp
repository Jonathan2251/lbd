// clang -target mips-unknown-linux-gnu -c ch8_1_4.cpp -emit-llvm -o ch8_1_4.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch8_1_4.bc -o -

/// start
int main()
{
  int a = 3;
  
  if (a != 0)
    a++;
  goto L1;
  a++;
L1:
  a--;
    
  return a;
}
