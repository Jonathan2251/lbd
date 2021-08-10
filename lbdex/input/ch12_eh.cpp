// clang -c ch12_eh.cpp -emit-llvm -o ch12_eh.bc
// ~/llvm/test/build/bin/llvm-dis ch12_eh.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch12_eh.bc -o -

/// start
class Ex1 {};
void throw_exception(int a, int b) {
  Ex1 ex1;

  if (a > b) {
    throw ex1;
  }
}

int test_try_catch() {
  try {
    throw_exception(2, 1);
  }
  catch(...) {
    return 1;
  }
  return 0;
}

