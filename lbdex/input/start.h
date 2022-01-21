
/// start

#ifndef _START_H_
#define _START_H_

// defined in cpu0.v
#define IOADDR 0xff000000

// $sp begin at mem less than IOADDR
#if 1
#define INIT_SP \
  asm("addiu $sp, $zero, 0x0000"); \
  asm("lui $sp, 0xff");
#else
#define INIT_SP \
  asm("lui $sp, 0x7"); \
  asm("addiu $sp, $sp, 0xfffc");
#endif

#define SET_SW \
asm("andi $sw, $zero, 0"); \
asm("ori  $sw, $sw, 0x1e00"); // enable all interrupts

#define initRegs() \
  asm("addiu $1,  $zero, 0"); \
  asm("addiu $2,  $zero, 0"); \
  asm("addiu $3,  $zero, 0"); \
  asm("addiu $4,  $zero, 0"); \
  asm("addiu $5,  $zero, 0"); \
  asm("addiu $t9, $zero, 0"); \
  asm("addiu $7,  $zero, 0"); \
  asm("addiu $8,  $zero, 0"); \
  asm("addiu $9,  $zero, 0"); \
  asm("addiu $10, $zero, 0"); \
  SET_SW;                     \
  asm("addiu $fp, $zero, 0");

#endif
