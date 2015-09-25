#include "a.h"

void ISR() {
  asm("ISR:");
  return;
}

int foo4(void) {
  return 5;
}

int main() {
  return foo1();
}

