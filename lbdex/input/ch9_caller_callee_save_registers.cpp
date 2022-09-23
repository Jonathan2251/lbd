// clang -O0 -target mips-unknown-linux-gnu -c ch9_caller_callee_save_registers.cpp -emit-llvm -o ch9_caller_callee_save_registers.bc
// ~/llvm/test/build/bin/llc -O0 -march=cpu0 -relocation-model=static -filetype=asm ch9_caller_callee_save_registers.bc -o -
// ~/llvm/debug/build/bin/llc -O0 -march=mips -relocation-model=static -filetype=asm ch9_caller_callee_save_registers.bc -o -

/// start
extern int add1(int x);

int callee()
{ 
  int t1 = 3;
  int result = add1(t1);  
  result = result - t1;
  
  return result;
}
