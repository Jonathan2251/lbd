/// start
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

