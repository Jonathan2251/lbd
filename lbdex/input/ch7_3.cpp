// clang -target mips-unknown-linux-gnu -c ch7_3.cpp -emit-llvm -o ch7_3.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch7_3.bc -o -

/// start
bool test_load_bool()
{
  int a = 1;

  if (a < 0)
    return false;

  return true;
}

