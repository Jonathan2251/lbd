// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch8_2_deluselessjmp.cpp -emit-llvm -o ch8_2_deluselessjmp.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm -enable-cpu0-del-useless-jmp=false ch8_2_deluselessjmp.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm -stats ch8_2_deluselessjmp.bc -o -

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

