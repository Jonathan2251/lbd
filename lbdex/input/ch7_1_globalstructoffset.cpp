// clang -target mips-unknown-linux-gnu -c ch7_1_globalstructoffset.cpp -emit-llvm -o ch7_1_globalstructoffset.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch7_1_globalstructoffset.bc -o -

// No need to verify in ch_nolld.cpp since test_func_arg_struct() of ch9_2_1.cpp include the test already

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

