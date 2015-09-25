#ifndef _DEBUG_H_
#define _DEBUG_H_

#define STOP \
  asm("lui $t9, 0xffff"); \
  asm("addiu $t9, $zero, 0xffff"); \
  asm("ret $t9");

#define ENABLE_TRACE \
  asm("ori $sw, $sw, 0x0020");

#define DISABLE_TRACE \
  asm("lui $at, 0xffff");       \
  asm("ori $at, $at, 0xffdf");  \
  asm("and $sw, $sw, $at"); // clear `D

#define SET_OVERFLOW \
  asm("ori $sw, $sw, 0x008");

#define CLEAR_OVERFLOW \
  asm("lui $7, 0xffff");        \
  asm("ori $7, $7, 0xfff7");    \
  asm("and $sw, $sw, $7"); // clear `V

#define SET_SOFTWARE_INT \
  asm("ori $sw, $sw, 0x4000");

#define CLEAR_SOFTWARE_INT \
  asm("lui $7, 0xffff");        \
  asm("ori $7, $7, 0xbfff");    \
  asm("and $sw, $sw, $7");

#define SAVE_REGISTERS          \
  asm("lui $at, 7");            \
  asm("ori $at, $at, 0xff00");  \
  asm("st $2,   0($at)");       \
  asm("st $3,   4($at)");       \
  asm("st $4,   8($at)");       \
  asm("st $5,  12($at)");       \
  asm("st $t9, 16($at)");       \
  asm("st $7,  20($at)");       \
  asm("st $8,  24($at)");       \
  asm("st $9,  28($at)");       \
  asm("st $10, 32($at)");       \
  asm("st $gp, 36($at)");       \
  asm("st $12, 40($at)");       \
  asm("st $13, 44($at)");

#define RESTORE_REGISTERS       \
  asm("lui $at,  7");           \
  asm("ori $at,  $at, 0xff00"); \
  asm("ld  $2,   0($at)");      \
  asm("ld  $3,   4($at)");      \
  asm("ld  $4,   8($at)");      \
  asm("ld  $5,  12($at)");      \
  asm("ld  $t9, 16($at)");      \
  asm("ld  $7,  20($at)");      \
  asm("ld  $8,  24($at)");      \
  asm("ld  $9,  28($at)");      \
  asm("ld  $10, 32($at)");      \
  asm("ld  $gp, 36($at)");      \
  asm("ld  $12, 40($at)");      \
  asm("ld  $13, 44($at)");

#define OVERFLOW     0x8
#define INT          0x2000
#define SOFTWARE_INT 0x4000
#define INT1         0x8000
#define INT2         0x10000

extern void int_sim();

#endif

