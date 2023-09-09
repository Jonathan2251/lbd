int cfg_ex(int a, int b, int n)
{
  for (int i = 0; i <= n; i++) {
    if (a < b) {
      a = a + i;
      b = b - 1;
    }
    if (b == 0) {
      goto label_1;
    }
  }

label_1:
  switch (a) {
  case 10:
    a = a*a-b+2;
    a++;
    break;
  }
  
  return (a+b);
}
