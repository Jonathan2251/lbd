// clang -target mips-unknown-linux-gnu -c ch9_3_stacksave.cpp -emit-llvm -o ch9_3_stacksave.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_3_stacksave.bc -o -

/// start
int test_stacksaverestore(unsigned x) {
  // CHECK: call i8* @llvm.stacksave()
  char s1[x];
  s1[x] = 5;
  
  return s1[x];
  // CHECK: call void @llvm.stackrestore(i8*
}