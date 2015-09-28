// clang -O0 -target mips-unknown-linux-gnu -c ch9_caller_callee_save_registers.cpp -emit-llvm -o ch9_caller_callee_save_registers.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -O0 -march=cpu0 -relocation-model=static -filetype=asm ch9_caller_callee_save_registers.bc -o -
// ~/llvm/release/cmake_debug_build/Debug/bin/llc -O0 -march=mips -relocation-model=static -filetype=asm ch9_caller_callee_save_registers.bc -o -

/// start
extern int add1(int x);

int caller(int a1)
{ 
  int t1 = a1;
  int result = add1(t1);  
  result = result - t1;
  
  return result;
}