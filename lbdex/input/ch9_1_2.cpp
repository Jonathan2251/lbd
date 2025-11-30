// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch9_1_2.cpp -emit-llvm -o ch9_1_2.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch9_1_2.bc -o -

/// start
int main()
{
  char str[81] = "Hello world";
  char s[6] = "Hello";
  
  return 0;
}
