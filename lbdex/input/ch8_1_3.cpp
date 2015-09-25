// clang -target mips-unknown-linux-gnu -c ch8_1_3.cpp -emit-llvm -o ch8_1_3.bc
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch8_1_3.bc -o -

/// start
int main()
{
  int a;
  int b = 5;
  int i = 0;
  
  for (i = 0; i == 3; i++) {
    a = a + i;
  }
  for (i = 0; i != 3; i++) {
    a = a + i;
  }
  for (i = 0; i > 3; i++) {
    a = a + i;
  }
  for (i = 0; i > 3; i++) {
    a = a + i;
  }
  for (i = 0; i == b; i++) {
    a++;
  }
  for (i = 0; i != b; i++) {
    a++;
  }
  for (i = 0; i < b; i++) {
    a++;
  }
  for (i = 7; i > b; i--) {
    a--;
  }
  for (i = 0; i <= b; i++) {
    a++;
  }
label_1:
  for (i = 7; i >= b; i--) {
    a--;
  }
  
  while (i < 7) {
    a++;
    i++;
    if (a >= 4)
      continue;
    else if (a == 3) {
      break;
    }
  }
  if (a == 3)
    goto label_1;

  switch (a) {
  case 1:
    a = a+1;
    break;
  case 2:
    a = a+2;
    break;
  default:
    a = a+8;
  }
  
  return a;
}
