// ~/llvm/debug/build/bin/clang -O1 -c ch8_3_2.cpp -emit-llvm -o ch8_3_2.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm ch8_3_2.bc -o -

/// start

// The following files will generate IR select when compile with clang -O1 but 
// ~/llvm/debug/build/bin/clang -O0 won't generate IR select.
volatile int a = 1;
volatile int b = 2;

int test_movx_3()
{
  int c = 0;

  if (a < b)
    return 1;
  else
    return 2;
}

int test_movx_4()
{
  int c = 0;

  if (a)
    c = 1;
  else
    c = 3;

  return c;
}

