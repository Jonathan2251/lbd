// clang -target mips-unknown-linux-gnu -c ch7_1_localarrayinit.cpp -emit-llvm -o ch7_1_localarrayinit.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch7_1_localarrayinit.bc -o -

/// start
int main()
{
  int a[3]={0, 1, 2};
    
  return 0;
}
