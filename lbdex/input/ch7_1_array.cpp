// clang -target mips-unknown-linux-gnu -c ch7_1_array.cpp -emit-llvm -o ch7_1_array.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch7_1_array.bc -o -

/// start

// Need libc to do array init. ref. http://www.dbp-consulting.com/tutorials/debugging/linuxProgramStartup.html
int ta[][4] =
{
{0x00000000, 0x00000001, 0x00000000}
};

int test_array()
{
  return ta[0][1];
}
