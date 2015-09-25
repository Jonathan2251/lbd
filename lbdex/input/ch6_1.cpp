// clang -target mips-unknown-linux-gnu -c ch6_1.cpp -emit-llvm -o ch6_1.bc
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=false -filetype=asm ch6_1.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=true -filetype=asm ch6_1.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch6_1.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=obj ch6_1.bc -o ch6_1.cpu0.o
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=static -filetype=obj ch6_1.bc -o ch6_1.cpu0.static.o

// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=true -filetype=asm ch6_1.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=false -filetype=asm ch6_1.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=true -filetype=asm ch6_1.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=false -filetype=asm ch6_1.bc -o -
// /Applications/Xcode.app/Contents/Developer/usr/bin/lldb -- /Users/Jonathan/llvm/test/cmake_debug_build/bin/Debug/llc -march=cpu0 -filetype=asm ch6_1.bc -o ch6_1.cpu0.s 

/// start
int gStart = 3;
int gI = 100;
int test_global()
{
  int c = 0;

  c = gI;

  return c;
}
