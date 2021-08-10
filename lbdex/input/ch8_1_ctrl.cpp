// clang -target mips-unknown-linux-gnu -c ch8_1_ctrl.cpp -emit-llvm -o ch8_1_ctrl.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch8_1_ctrl.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm ch8_1_ctrl.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -view-isel-dags -relocation-model=pic -filetype=asm ch8_1_ctrl.bc -o -

/// start
int test_control1()
{
  unsigned int a = 0;
  int b = 39;
  int i = 3;
  
  for (i = 0; i <= 3; i++) {
    a = a + i; // a = 6
  }
  if (b == 0) {
    b--;
  }
  else if (b > 0) {
    b++; // b = 40
    goto label_1;
  }

label_1:
  for (i = 3; i >= 2; i--) {
    a--; // a = 4
  }
  
  while (i < 100) {
    a++;
    i++;
    if (a < 10)
      continue;
    else
      break;
  }

  switch (a) {
  case 10:
    a++; // a = 11
    break;
  default:
    a = 0;
  }
  
  return (a+b); // 11+40 = 51
}
