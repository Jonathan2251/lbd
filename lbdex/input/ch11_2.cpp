// clang -target mips-unknown-linux-gnu -c ch11_2.cpp -emit-llvm -o ch11_2.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch11_2.bc -o -

/// start
extern "C" int printf(const char *format, ...);
int inlineasm_addu(void)
{
  int foo = 10;
  const int bar = 15;

//  call i32 asm sideeffect "addu $0,$1,$2", "=r,r,r"(i32 %1, i32 %2) #1, !srcloc !1
  __asm__ __volatile__("addu %0,%1,%2"
                       :"=r"(foo) // 5
                       :"r"(foo), "r"(bar)
                       );

  return foo;
}

int inlineasm_longlong(void)
{
  int a, b;
  const long long bar = 0x0000000500000006;
  int* p = (int*)&bar;
//  int* q = (p+1); // Do not set q here.

//  call i32 asm sideeffect "ld $0,$1", "=r,*m"(i32* %2) #2, !srcloc !2
  __asm__ __volatile__("ld %0,%1"
                       :"=r"(a) // 0x700070007000700b
                       :"m"(*p)
                       );
  int* q = (p+1); // Set q just before inline asm refer to avoid register clobbered. 
//  call i32 asm sideeffect "ld $0,$1", "=r,*m"(i32* %6) #2, !srcloc !3
  __asm__ __volatile__("ld %0,%1"
                       :"=r"(b) // 11
                       :"m"(*q)
//              Or use :"m"(*(p+1)) to avoid register clobbered. 
                       );

  return (a+b);
}

int inlineasm_constraint(void)
{
  int foo = 10;
  const int n_5 = -5;
  const int n5 = 5;
  const int n0 = 0;
  const unsigned int un5 = 5;
  const int n65536 = 0x10000;
  const int n_65531 = -65531;

//   call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,I"(i32 %1, i32 15) #1, !srcloc !2
  __asm__ __volatile__("addiu %0,%1,%2"
                       :"=r"(foo) // 15
                       :"r"(foo), "I"(n_5)
                       );

  __asm__ __volatile__("addiu %0,%1,%2"
                       :"=r"(foo) // 15
                       :"r"(foo), "J"(n0)
                       );

  __asm__ __volatile__("addiu %0,%1,%2"
                       :"=r"(foo) // 10
                       :"r"(foo), "K"(n5)
                       );

  __asm__ __volatile__("ori %0,%1,%2"
                       :"=r"(foo) // 10
                       :"r"(foo), "L"(n65536) // 0x10000 = 65536
                       );

  __asm__ __volatile__("addiu %0,%1,%2"
                       :"=r"(foo) // 15
                       :"r"(foo), "N"(n_65531)
                       );

  __asm__ __volatile__("addiu %0,%1,%2"
                       :"=r"(foo) // 10
                       :"r"(foo), "O"(n_5)
                       );

  __asm__ __volatile__("addiu %0,%1,%2"
                       :"=r"(foo) // 15
                       :"r"(foo), "P"(un5)
                       );

  return foo;
}

int inlineasm_arg(int u, int v)
{
  int w;

  __asm__ __volatile__("subu %0,%1,%2"
                       :"=r"(w)
                       :"r"(u), "r"(v)
                       );

  return w;
}

int g[3] = {1,2,3};

int inlineasm_global()
{
  int c, d;
  __asm__ __volatile__("ld %0,%1"
                       :"=r"(c) // c=3
                       :"m"(g[2])
                       );

  __asm__ __volatile__("addiu %0,%1,1"
                       :"=r"(d) // d=4
                       :"r"(c)
                       );

  return d;
}

#ifdef TESTSOFTFLOATLIB
// test_float() will call soft float library
int inlineasm_float()
{
  float a = 2.2;
  float b = 3.3;
  
  int c = (int)(a + b);

  int d;
  __asm__ __volatile__("addiu %0,%1,1"
                       :"=r"(d)
                       :"r"(c)
                       );

  return d;
}
#endif

int test_inlineasm()
{
  int a, b, c, d, e, f;

  a = inlineasm_addu(); // 25
  b = inlineasm_longlong(); // 11
  c = inlineasm_constraint(); // 15
  d = inlineasm_arg(1, 10); // -9
  e = inlineasm_arg(6, 3); // 3
  __asm__ __volatile__("addiu %0,%1,1"
                       :"=r"(f) // e=4
                       :"r"(e)
                       );

  return (a+b+c+d+e+f); // 25+11+15-9+3+4=49
}

