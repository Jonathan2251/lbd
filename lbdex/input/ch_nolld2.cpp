
/// start

#include "print.cpp"

#include "ch9_3_alloc.cpp"

int test_nolld2()
{
  bool pass = true;
  int a = 0;

  a = test_alloc();
  print_integer(a);  // a = 31
  if (a != 31) pass = false;
  return pass;
}

