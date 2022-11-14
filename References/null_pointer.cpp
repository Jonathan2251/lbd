// with #define FORGET_SET then both clang -O0 and -O3 are output "hello"
// without then clang -O0 output segmentation fault

// For undef is only happened in compiler optimization.

#include <stdio.h>

#define FORGET_SET

static void (*FP)() = 0;
static void impl() {
  printf("hello\n");
}
void set() {
  FP = impl;
}
void call() {
  FP();
}
int main() {
#ifndef FORGET_SET
  set();
#endif
  call();
}
