// clang -target mips-unknown-linux-gnu -c ch7_5.cpp -emit-llvm -o ch7_5.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch7_5.bc -o -

/// start
struct Date
{
  int year;
  int month;
  int day;
};

Date date = {2012, 10, 12};
int a[3] = {2012, 10, 12};

int test_struct()
{
  int day = date.day;
  int i = a[1];

  return (i+day); // 10+12=22
}

