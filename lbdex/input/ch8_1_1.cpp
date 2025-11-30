// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch8_1_1.cpp -emit-llvm -o ch8_1_1.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch8_1_1.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm ch8_1_1.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -view-isel-dags -relocation-model=pic -filetype=asm ch8_1_1.bc -o -

/// start
int test_ifctrl()
{
  unsigned int a = 0;
  
  if (a == 0) {
    a++; // a = 1
  }
  
  return a;
}
