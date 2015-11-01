
/// start
#define TEST_ROXV
#define RUN_ON_VERILOG

#include "print.cpp"

#include "ch4_1_math.cpp"
#include "ch4_1_rotate.cpp"
#include "ch4_1_mult2.cpp"
#include "ch4_1_mod.cpp"
#include "ch4_1_div.cpp"
#include "ch4_2_logic.cpp"
#include "ch7_1.cpp"
#include "ch7_2_2.cpp"
#include "ch7_3.cpp"
#include "ch7_4.cpp"
#include "ch7_1_vector.cpp"
#include "ch8_1_1.cpp"
#include "ch8_2.cpp"
#include "ch8_3.cpp"
#include "ch9_1_4.cpp"
#include "ch9_2_3_tailcall.cpp"
#include "ch9_3.cpp"
#include "ch9_3_stacksave.cpp"
#include "ch9_3_bswap.cpp"
#include "ch9_4.cpp"
#include "ch11_2.cpp"

// Test build only for the following files since it needs lld linker support.
#include "ch6_1.cpp"
#include "ch9_2_1.cpp"
#include "ch9_2_2.cpp"
#include "ch9_3_2.cpp"
#include "ch12_inherit.cpp"

void test_asm_build()
{
  #include "ch11_1.cpp"
#ifdef CPU032II
  #include "ch11_1_2.cpp"
#endif
}

int test_rotate()
{
  int a = test_rotate_left1(); // rolv 4, 30 = 1
  int b = test_rotate_left(); // rol 8, 30  = 2
  int c = test_rotate_right(); // rorv 1, 30 = 4
  
  return (a+b+c);
}

int test_nolld()
{
  bool pass = true;
  int a = 0;

  a = test_math();
  print_integer(a);  // a = 74
  if (a != 74) pass = false;
  a = test_rotate();
  print_integer(a);  // a = 7
  if (a != 7) pass = false;
  a = test_mult();
  print_integer(a);  // a = 0
  if (a != 0) pass = false;
  a = test_mod();
  print_integer(a);  // a = 0
  if (a != 0) pass = false;
  a = test_div();
  print_integer(a);  // a = 253
  if (a != 253) pass = false;
  a = test_local_pointer();
  print_integer(a);  // a = 3
  if (a != 3) pass = false;
  a = (int)test_load_bool();
  print_integer(a);  // a = 1
  if (a != 1) pass = false;
  a = test_andorxornot();
  print_integer(a); // a = 14
  if (a != 14) pass = false;
  a = test_setxx();
  print_integer(a); // a = 3
  if (a != 3) pass = false;
  a = test_signed_char();
  print_integer(a); // a = -126
  if (a != -126) pass = false;
  a = test_unsigned_char();
  print_integer(a); // a = 130
  if (a != 130) pass = false;
  a = test_signed_short();
  print_integer(a); // a = -32766
  if (a != -32766) pass = false;
  a = test_unsigned_short();
  print_integer(a); // a = 32770
  if (a != 32770) pass = false;
  long long b = test_longlong();
  print_integer((int)(b >> 32)); // 393307
  if ((int)(b >> 32) != 393307) pass = false;
  print_integer((int)b); // 16777222
  if ((int)(b) != 16777222) pass = false;
  a = test_cmplt_short();
  print_integer(a); // a = 2
  if (a != 2) pass = false;
  a = test_cmplt_long();
  print_integer(a); // a = 4
  if (a != 4) pass = false;
  a = test_control1();
  print_integer(a);	// a = 51
  if (a != 51) pass = false;
  a = test_DelUselessJMP();
  print_integer(a); // a = 2
  if (a != 2) pass = false;
  a = test_movx_1();
  print_integer(a); // a = 3
  if (a != 3) pass = false;
  a = test_movx_2();
  print_integer(a); // a = 1
  if (a != 1) pass = false;
  print_integer(2147483647); // test mod % (mult) from itoa.cpp
  print_integer(-2147483648); // test mod % (multu) from itoa.cpp
  a = test_madd();
  print_integer(a); // a = 7
  if (a != 7) pass = false;
  a = test_tailcall(5);
  print_integer(a); // a = 120
  if (a != 120) pass = false;
  a = test_vararg();
  print_integer(a); // a = 15
  if (a != 15) pass = false;
  a = test_stacksaverestore(100);
  print_integer(a); // a = 5
  if (a != 5) pass = false;
  a = test_bswap();
  print_integer(a); // a = 0
  if (a != 0) pass = false;
  a = test_alloc();
  print_integer(a); // a = 31
  if (a != 31) pass = false;
  a = test_inlineasm();
  print_integer(a); // a = 49
  if (a != 49) pass = false;

  return pass;
}

/* result:
74
15
253
3
1
14
3
-126
130
-32766
32770
393307
16777222
51
2
3
1
2147483647
-2147483648
7
120
15
5
49
...
RET to PC < 0, finished!
*/
