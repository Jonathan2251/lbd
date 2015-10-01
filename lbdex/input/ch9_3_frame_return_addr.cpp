// clang -target mips-unknown-linux-gnu -c ch9_3_frame_return_addr.cpp -emit-llvm -o ch9_3_frame_return_addr.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch9_3_frame_return_addr.bc -o -

/// start
int test_framereturnaddress() {
  int frameaddr = (int)__builtin_frame_address(0);
  int returnaddr = (int)__builtin_return_address(0);

  if (frameaddr == (int)__builtin_frame_address(0) && 
      returnaddr == (int)__builtin_return_address(0))
    return 0;
  else
    return 1;
}

