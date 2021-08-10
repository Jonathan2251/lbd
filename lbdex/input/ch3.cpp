// clang -target mips-unknown-linux-gnu -c ch3.cpp -emit-llvm -o ch3.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc -o ch3.cpu0.s
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=obj ch3.bc -o ch3.cpu0.o

/// start
int main()
{
  return 0;
}

