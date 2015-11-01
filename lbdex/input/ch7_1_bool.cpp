// clang -target mips-unknown-linux-gnu -c ch7_1_bool.cpp -emit-llvm -o ch7_1_bool.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch7_1_bool.bc -o -

/// start
bool test_load_bool()
{
  int a = 1;

  if (a < 0)
    return false;

  return true;
}

