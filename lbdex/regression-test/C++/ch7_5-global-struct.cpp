// clang -target mips-unknown-linux-gnu -c ch7_5-global-struct.cpp -emit-llvm -o ch7_5-global-struct.bc
// /Users/Jonathan/llvm/test/cmake_debug_build/bin/Debug/llvm-dis ch7_5-global-struct.bc -o ch7_5-global-struct.ll

/// start
struct Date
{
  int year;
  int month;
  int day;
};

Date date = {2012, 10, 12};

int test_struct()
{
  int day = date.day;

  return day; // 2012
}

