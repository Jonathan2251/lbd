// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch9_3_frame_return_addr.cpp -emit-llvm -o ch9_3_frame_return_addr.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch9_3_frame_return_addr.bc -o -

/// start
int display_frameaddress() {
  return (int)__builtin_frame_address(0);
}

extern int fn();

int display_returnaddress() {
  int a = (int)__builtin_return_address(0);
  fn();
  return a;
}
