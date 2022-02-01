// clang -target mips-unknown-linux-gnu -S ch7_1_longlong.cpp -emit-llvm
// ~/llvm/test/build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm ch7_1_longlong.ll

/// start
long long test_longlong()
{
  long long a = 0x300000002;
  long long b = 0x100000001;
  int a1 = 0x3001000;
  int b1 = 0x2001000;
  
  long long c = a + b;   // c = 0x00000004,00000003
  long long d = a - b;   // d = 0x00000002,00000001
  long long e = a * b;   // e = 0x00000005,00000002
  long long f = (long long)a1 * (long long)b1; // f = 0x00060050,01000000

  long long g = ((-7 * 8) + 1) >> 4; // g = -55/16=-3.4375=-4

  return (c+d+e+f+g); // (0x0006005b,01000002) = (393307,16777218)
}

