// clang -O1 -target mips-unknown-linux-gnu -c ch_arm-tailcall.cpp -emit-llvm -o ch_arm-tailcall.bc
// ~/llvm/test/cmake_debug_build/bin/llc -march=arm -relocation-model=static -filetype=asm ch_arm-tailcall.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=mips -relocation-model=static -filetype=asm -enable-mips-tail-calls ch_arm-tailcall.bc -o -

// ~/llvm/test/cmake_debug_build/bin/llvm-dis ch_arm-tailcall.bc -o -

// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=static -filetype=asm -enable-cpu0-tail-calls ch_arm-tailcall.bc -o -

/// start
#if 1
int factorial(int x)
{
  if (x > 0)
    return x*factorial(x-1);
  else
    return 1;
}

int test_tailcall(int a)
{
  int b = 3;

  return b+factorial(a);
}

#else
int test_tailcall(int a);

int factorial(int x)
{
  if (x > 0)
    return x*test_tailcall(x-1);
  else
    return 1;
}

int test_tailcall(int a)
{
#if 0
  int b = 3;

  return b+factorial(a);
#else
  return factorial(a);
#endif
}
#endif
