
/// start

#include "start.h"

// boot:
  asm("boot:");
//  asm("_start:");
  asm("jmp 12"); // RESET: jmp RESET_START;
  asm("jmp 4");  // ERROR: jmp ERR_HANDLE;
  asm("jmp 4");  // IRQ: jmp IRQ_HANDLE;
  asm("jmp -4"); // ERR_HANDLE: jmp ERR_HANDLE; (loop forever)

// RESET_START:
  initRegs();
  asm("addiu $gp, $ZERO, 0");
  asm("addiu $lr, $ZERO, -1");
  
  asm("addiu $sp, $zero, 0x6ffc");
  asm("mfc0 $3, $pc");
  asm("addiu $3, $3, 0x8"); // Assume main() entry point is at the next next 
                             // instruction.
  asm("ret $3");
  asm("nop");
