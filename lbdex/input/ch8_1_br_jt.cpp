// ~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -c ch8_1_br_jt.cpp -emit-llvm -o ch8_1_br_jt.bc
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch8_1_br_jt.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch8_1_br_jt.bc -o -
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic -filetype=obj ch8_1_br_jt.bc -o ch8_1_br_jt.cpu0.o
// ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=static -filetype=obj ch8_1_br_jt.bc -o ch8_1_br_jt.cpu0.o

/// start

// This program test IR br_jt and JumpTable. It will generate JTI symbol 
// reference (global variable) as the following. So, need lld support.
//	lui	$3, %hi($JTI0_0)
//	ori	$3, $3, %lo($JTI0_0)

typedef unsigned char Byte;
typedef unsigned char *Address;
typedef enum {FALSE=0, TRUE=1} Boolean;
unsigned char sBuffer[4] = {0x35, 0x35, 0x00, 0x00};

int test_ctrl2()
{
  Boolean Result = FALSE;
  Byte Comparator = sBuffer[1];
  Byte ByteToCompare = sBuffer[0];

  switch (sBuffer[0])
  {
    case 0x30:
      if (ByteToCompare == Comparator)
        Result = TRUE;
      break;
    case 0x31:
      if (ByteToCompare != Comparator)
        Result = TRUE;
      break;
    case 0x32:
      if (ByteToCompare > Comparator)
        Result = TRUE;
      break;
    case 0x33:
      if (ByteToCompare < Comparator)
        Result = TRUE;
      break;
    case 0x34:
      if (ByteToCompare >= Comparator)
        Result = TRUE;
      break;
    case 0x35:
      if (ByteToCompare <= Comparator)
        Result = TRUE;
      break;
  }
  return (int)Result;
}

