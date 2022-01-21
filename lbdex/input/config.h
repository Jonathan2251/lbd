#ifndef _CONFIG_H_
#define _CONFIG_H_

// defined in cpu0.v
#define IOADDR 0xff000000

// $sp begin at mem less than MEMSIZE of cpu0.v
#define INIT_SP \
  asm("addiu $sp, $zero, 0x0000"); \
  asm("lui $sp, 0xff");

// default test part of 1 in 10 for udivmoddi4_test.c
//#define TEST_FULL
//#define TEST_HALF

// TEST_FULL in udivmoddi4_test.c
#ifdef TEST_FULL
  #define PART_2_IN_10
  #define PART_HALF
#else
  #ifdef TEST_HALF
    #define PART_HALF
  #endif
#endif

#endif
