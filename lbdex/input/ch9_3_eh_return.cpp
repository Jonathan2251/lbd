// clang -target mips-unknown-linux-gnu -c ch9_3_eh_return.cpp -emit-llvm -o ch9_3_eh_return.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_3_eh_return.bc -o -

/// start
int exception_handler() {
  return 3;
}

__attribute__ ((weak)) 
int test_ehreturn() {
  void* handler=(void*)(&exception_handler);
  __builtin_eh_return(0, handler); // no warning, eh_return never returns.
}

