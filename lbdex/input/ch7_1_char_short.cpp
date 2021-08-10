// clang -target mips-unknown-linux-gnu -c ch7_1_char_short.cpp -emit-llvm -o ch7_1_char_short.bc
// ~/llvm/test/build/bin/llvm-dis ch7_1_char_short.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch7_1_char_short.bc -o -

/// start
int test_signed_char()
{
  char a = 0x80;
  int i = (signed int)a;
  i = i + 2; // i = (-128+2) = -126

  return i;
}

int test_unsigned_char()
{
  unsigned char c = 0x80;
  unsigned int ui = (unsigned int)c;
  ui = ui + 2; // i = (128+2) = 130

  return (int)ui;
}

int test_signed_short()
{
  short a = 0x8000;
  int i = (signed int)a;
  i = i + 2; // i = (-32768+2) = -32766

  return i;
}

int test_unsigned_short()
{
  unsigned short c = 0x8000;
  unsigned int ui = (unsigned int)c;
  ui = ui + 2; // i = (32768+2) = 32770
  c = (unsigned short)ui;

  return (int)ui;
}

