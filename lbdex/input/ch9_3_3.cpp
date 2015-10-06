// clang -target mips-unknown-linux-gnu -c ch9_3_3.cpp -emit-llvm -o ch9_3_3.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc ch9_3_3.bc -o ch9_3_3.s
// clang++ ch9_3_3.s -o ch9_3_3.native
// ./ch9_3_3.native
// lldb -- ch9_3_3.native
// b main
// s
// ...
// print $rsp		; print %rsp, choose $ instead of % in assembly code

// mips-linux-gnu-g++ -g ch9_3_3.cpp -o ch9_3_3 -static
// qemu-mips ch9_3_3
// mips-linux-gnu-g++ -S ch9_3_3.cpp
// cat ch9_3_3.s

/// start
#include <stdarg.h>

extern int main();
int start()
{
  int(*pf)() = &main;

  int a = main();
  a = pf();
	
  return a;
}
