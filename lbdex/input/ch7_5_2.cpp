// clang -target mips-unknown-linux-gnu -c ch7_5_2.cpp -emit-llvm -o ch7_5_2.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch7_5_2.bc -o -

/// start
int main()
{
  int a[3]={0, 1, 2};
    
  return 0;
}
