// clang -target mips-unknown-linux-gnu -c ch7_2.cpp -emit-llvm -o ch7_2.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch7_2.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch7_2.bc -o -

/// start
struct Date
{
  short year;
  char month;
  char day;
  char hour;
  char minute;
  char second;
};

unsigned char b[4] = {'a', 'b', 'c', '\0'};

int test_char()
{
  unsigned char a = b[1];
  char c = (char)b[1];
  Date date1 = {2012, (char)11, (char)25, (char)9, (char)40, (char)15};
  char m = date1.month;
  char s = date1.second;

  return 0;
}

