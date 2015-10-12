
/// start

#include "print.cpp"

#include "ch8_2_longbranch.cpp"
#include "ch9_4.cpp"

int test_nolld2()
{
  bool pass = true;
  int a = 0;

  a = test_longbranch();
  print_integer(a);	// a = 1
  if (a != 1) pass = false;
  a = test_alloc();  // 31
  print_integer(a);  // a = 1
  if (a != 31) pass = false;
  return pass;
}

