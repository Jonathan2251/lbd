// clang -target mips-unknown-linux-gnu -c ch8_2.cpp -emit-llvm -o ch8_2.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm -enable-cpu0-del-useless-jmp=false ch8_2.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm -stats ch8_2.bc -o -

/// start
int test_DelUselessJMP()
{
  int a = 1; int b = -2; int c = 3;
  
  if (a == 0) {
    a++;
  }
  if (b == 0) {
    a = a + 3;
    b++;
  } else if (b < 0) {
    a = a + b;
    b--;
  }
  if (c > 0) {
    a = a + c;
    c++;
  }
  
  return a;
}

