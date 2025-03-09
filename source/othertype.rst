.. _sec-othertypesupport:

Other data type
=================

.. contents::
   :local:
   :depth: 4

Until now, we have only handled int and long types of 32-bit size.
This chapter introduces other types, such as pointers and types that are not 
32-bit, including bool, char, short int, and long long.
 
Local Variable Pointer
-----------------------

To support pointers to local variables, add the following code fragment to
Cpu0InstrInfo.td and Cpu0InstPrinter.cpp:

.. rubric:: lbdex/chapters/Chapter7_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH7_1 1
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@EffectiveAddress
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@def LEA_ADDiu {
    :end-before: //@def LEA_ADDiu }
  
.. rubric:: lbdex/chapters/Chapter3_2/InstPrinter/Cpu0InstPrinter.h
.. literalinclude:: ../lbdex/Cpu0/InstPrinter/Cpu0InstPrinter.h
    :start-after: //#if CH >= CH7_1
    :end-before: //#endif

.. rubric:: lbdex/chapters/Chapter3_2/InstPrinter/Cpu0InstPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/InstPrinter/Cpu0InstPrinter.cpp
    :start-after: //#if CH >= CH7_1
    :end-before: //#endif

As noted in Cpu0InstPrinter.cpp, the printMemOperandEA function was added in an 
earlier Chapter 3.2 because the DAG data node mem_ea in Cpu0InstrInfo.td cannot 
be disabled by ch7_1_localpointer; only the opcode node can be disabled.

Run ch7_1_localpointer.cpp with the Chapter7_1/ directory, which supports 
pointers to local variables. The expected result is as follows:


.. rubric:: lbdex/input/ch7_1_localpointer.cpp
.. literalinclude:: ../lbdex/input/ch7_1_localpointer.cpp
    :start-after: /// start

.. code-block:: console

  118-165-66-82:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch7_1_localpointer.cpp -emit-llvm -o ch7_1_localpointer.bc
  118-165-66-82:input Jonathan$ llvm-dis ch7_1_localpointer.bc -o -
  ...
  ; Function Attrs: nounwind
  define i32 @_Z18test_local_pointerv() #0 {
    %b = alloca i32, align 4
    %p = alloca i32*, align 4
    store i32 3, i32* %b, align 4
    store i32* %b, i32** %p, align 4
    %1 = load i32** %p, align 4
    %2 = load i32* %1, align 4
    ret i32 %2
  }
  ...

  118-165-66-82:input Jonathan$ /Users/Jonathan/llvm/test/build/bin/llc
  -march=cpu0 -relocation-model=pic -filetype=asm 
  ch7_1_localpointer.bc -o -
    ...
	  addiu	$sp, $sp, -8
	  addiu	$2, $zero, 3
	  st	$2, 4($fp)
	  addiu	$2, $fp, 4     // b address is 4($sp)
	  st	$2, 0($fp)
	  ld	$2, 4($fp)
	  addiu	$sp, $sp, 8
	  ret	$lr
    ...


char, short int and bool
--------------------------

To support signed and unsigned char and short int, add the following
code to Chapter7_1/:

.. rubric:: lbdex/chapters/Chapter7_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH7_1 2
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH7_1 3
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH7_1 5
    :end-before: //#endif

Run Chapter7_1/ with ch7_1_char_in_struct.cpp to obtain the following result.

.. rubric:: lbdex/input/ch7_1_char_in_struct.cpp
.. literalinclude:: ../lbdex/input/ch7_1_char_in_struct.cpp
    :start-after: /// start

.. code-block:: console
  
  118-165-64-245:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llvm-dis ch7_1_char_in_struct.bc -o -
  define i32 @_Z9test_charv() #0 {
    %a = alloca i8, align 1
    %c = alloca i8, align 1
    %date1 = alloca %struct.Date, align 2
    %m = alloca i8, align 1
    %s = alloca i8, align 1
    %1 = load i8* getelementptr inbounds ([4 x i8]* @b, i32 0, i32 1), align 1
    store i8 %1, i8* %a, align 1
    %2 = load i8* getelementptr inbounds ([4 x i8]* @b, i32 0, i32 1), align 1
    store i8 %2, i8* %c, align 1
    %3 = bitcast %struct.Date* %date1 to i8*
    call void @llvm.memcpy.p0i8.p0i8.i32(i8* %3, i8* bitcast ({ i16, i8, i8, i8, 
    i8, i8, i8 }* @_ZZ9test_charvE5date1 to i8*), i32 8, i32 2, i1 false)
    %4 = getelementptr inbounds %struct.Date* %date1, i32 0, i32 1
    %5 = load i8* %4, align 1
    store i8 %5, i8* %m, align 1
    %6 = getelementptr inbounds %struct.Date* %date1, i32 0, i32 5
    %7 = load i8* %6, align 1
    store i8 %7, i8* %s, align 1
    ret i32 0
  }

  118-165-64-245:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch7_1_char_in_struct.cpp -emit-llvm -o ch7_1_char_in_struct.bc
  118-165-64-245:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm 
  ch7_1_char_in_struct.bc -o -
    ...
  # BB#0:                                 # %entry
    addiu $sp, $sp, -24
    lui $2, %got_hi(b)
    addu  $2, $2, $gp
    ld  $2, %got_lo(b)($2)
    lbu $3, 1($2)
    sb  $3, 20($fp)
    lbu $2, 1($2)
    sb  $2, 16($fp)
    ld  $2, %got($_ZZ9test_charvE5date1)($gp)
    addiu $2, $2, %lo($_ZZ9test_charvE5date1)
    lhu $3, 4($2)
    shl $3, $3, 16
    lhu $4, 6($2)
    or  $3, $3, $4
    st  $3, 12($fp) // store hour, minute and second on 12($sp)
    lhu $3, 2($2)
    lhu $2, 0($2)
    shl $2, $2, 16
    or  $2, $2, $3
    st  $2, 8($fp)    // store year, month and day on 8($sp)  
    lbu $2, 10($fp)   // m = date1.month;
    sb  $2, 4($fp)
    lbu $2, 14($fp)   // s = date1.second;
    sb  $2, 0($fp)
    addiu $sp, $sp, 24
    ret $lr
    .set  macro
    .set  reorder
    .end  _Z9test_charv
  $tmp1:
    .size _Z9test_charv, ($tmp1)-_Z9test_charv
  
    .type b,@object               # @b
    .data
    .globl  b
  b:
    .asciz   "abc"
    .size b, 4
  
    .type $_ZZ9test_charvE5date1,@object # @_ZZ9test_charvE5date1
    .section  .rodata.cst8,"aM",@progbits,8
    .align  1
  $_ZZ9test_charvE5date1:
    .2byte  2012                    # 0x7dc
    .byte 11                      # 0xb
    .byte 25                      # 0x19
    .byte 9                       # 0x9
    .byte 40                      # 0x28
    .byte 15                      # 0xf
    .space  1
    .size $_ZZ9test_charvE5date1, 8

Run Chapter7_1/ with ch7_1_char_short.cpp to obtain the following result.

.. rubric:: lbdex/input/ch7_1_char_short.cpp
.. literalinclude:: ../lbdex/input/ch7_1_char_short.cpp
    :start-after: /// start

.. code-block:: console
  
  1-160-136-236:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llvm-dis ch7_1_char_short.bc -o -
    ...
  define i32 @_Z16test_signed_charv() #0 {
    ...
    %1 = load i8* %a, align 1
    %2 = sext i8 %1 to i32
    ...
  }
  
  ; Function Attrs: nounwind
  define i32 @_Z18test_unsigned_charv() #0 {
    ...
    %1 = load i8* %c, align 1
    %2 = zext i8 %1 to i32
    ...
  }
  
  ; Function Attrs: nounwind
  define i32 @_Z17test_signed_shortv() #0 {
    ...
    %1 = load i16* %a, align 2
    %2 = sext i16 %1 to i32
    ...
  }
  
  ; Function Attrs: nounwind
  define i32 @_Z19test_unsigned_shortv() #0 {
    ...
    %1 = load i16* %c, align 2
    %2 = zext i16 %1 to i32
    ...
  }
  
  attributes #0 = { nounwind }
  
  1-160-136-236:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch7_1_char_short.bc -o -
    ...
    .globl  _Z16test_signed_charv
    ...
    lb  $2, 4($sp)
    ...
    .end  _Z16test_signed_charv
  
    .globl  _Z18test_unsigned_charv
    ...
    lbu $2, 4($sp)
    ...
    .end  _Z18test_unsigned_charv
    
    .globl  _Z17test_signed_shortv
    ...
    lh  $2, 4($sp)
    ...
    .end  _Z17test_signed_shortv
    
    .globl  _Z19test_unsigned_shortv
    ...
    lhu $2, 4($sp)
    ...
    .end  _Z19test_unsigned_shortv
    ...

As shown, lb/lh instructions are used for signed byte/short types, while 
lbu/lhu are used for unsigned byte/short types.
To efficiently support C type-casting and type-conversion features,
Cpu0 provides the lb instruction, which converts a char to an int with a single 
instruction.
The instructions lbu, lh, lhu, sb, and sh are applied to both signed and 
unsigned byte and short conversions.
Their differences were explained in Chapter 2.

To support loading the bool type, add the following code:

.. rubric:: lbdex/chapters/Chapter7_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH7_1 //1
    :end-before: #endif

.. code-block:: c++

    ...
  }

The purpose of setBooleanContents() is as follows, but its details are not well 
understood.
Without it, ch7_1_bool2.ll still works as shown below.

The IR input file ch7_1_bool2.ll is used for testing, as the C++ version
requires flow control, which is not supported at this point.
The file ch_run_backend.cpp includes a test fragment for bool, as shown below.

.. rubric:: include/llvm/Target/TargetLowering.h
.. code-block:: c++

    enum BooleanContent { // How the target represents true/false values.
      UndefinedBooleanContent,    // Only bit 0 counts, the rest can hold garbage.
      ZeroOrOneBooleanContent,        // All bits zero except for bit 0.
      ZeroOrNegativeOneBooleanContent // All bits equal to bit 0.
    };
  ...
  protected:
    /// setBooleanContents - Specify how the target extends the result of a
    /// boolean value from i1 to a wider type.  See getBooleanContents.
    void setBooleanContents(BooleanContent Ty) { BooleanContents = Ty; }
    /// setBooleanVectorContents - Specify how the target extends the result
    /// of a vector boolean value from a vector of i1 to a wider type.  See
    /// getBooleanContents.
    void setBooleanVectorContents(BooleanContent Ty) {
      BooleanVectorContents = Ty;
    }

.. rubric:: lbdex/input/ch7_1_bool2.ll
.. literalinclude:: ../lbdex/input/ch7_1_bool2.ll
    :start-after: /// start

.. code-block:: console

    118-165-64-245:input Jonathan$ /Users/Jonathan/llvm/test/build/
    bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch7_1_bool2.ll -o -

    .section .mdebug.abi32
    .previous
    .file "ch7_1_bool2.ll"
    .text
    .globl  verify_load_bool
    .align  2
    .type verify_load_bool,@function
    .ent  verify_load_bool        # @verify_load_bool
  verify_load_bool:
    .cfi_startproc
    .frame  $sp,8,$lr
    .mask   0x00000000,0
    .set  noreorder
    .set  nomacro
  # BB#0:                                 # %entry
    addiu $sp, $sp, -8
  $tmp1:
    .cfi_def_cfa_offset 8
    addiu $2, $zero, 1
    sb  $2, 7($sp)
    addiu $sp, $sp, 8
    ret $lr
    .set  macro
    .set  reorder
    .end  verify_load_bool
  $tmp2:
    .size verify_load_bool, ($tmp2)-verify_load_bool
    .cfi_endproc


The ch7_1_bool.cpp file provides a bool test version for C.
You can run it in Chapter8_1/ to obtain results similar to ch7_1_bool2.ll.

.. rubric:: lbdex/input/ch7_1_bool.cpp
.. literalinclude:: ../lbdex/input/ch7_1_bool.cpp
    :start-after: /// start

Summary Table

.. table:: The C, IR, and DAG translation for char, short and bool translation (ch7_1_char_short.cpp and ch7_1_bool2.ll).

  ==================================  =================================  ====================================
  C                                   .bc                                Optimized legalized selection DAG
  ==================================  =================================  ====================================
  char a =0x80;                       %1 = load i8* %a, align 1          - 
  int i = (signed int)a;              %2 = sext i8 %1 to i32             load ..., <..., sext from i8>
  unsigned char c = 0x80;             %1 = load i8* %c, align 1          -
  unsigned int ui = (unsigned int)c;  %2 = zext i8 %1 to i32             load ..., <..., zext from i8>
  short a =0x8000;                    %1 = load i16* %a, align 2         -
  int i = (signed int)a;              %2 = sext i16 %1 to i32            load ..., <..., sext from i16>
  unsigned short c = 0x8000;          %1 = load i16* %c, align 2         -
  unsigned int ui = (unsigned int)c;  %2 = zext i16 %1 to i32            load ..., <..., zext from i16>
  c = (unsigned short)ui;             %6 = trunc i32 %5 to i16           -
  -                                   store i16 %6, i16* %c, align 2     store ...,<..., trunc to i16>
  return true;                        store i1 1, i1* %retval, align 1   store ...,<..., trunc to i8>
  ==================================  =================================  ====================================


.. table:: The backend translation for char, short and bool translation (ch7_1_char_short.cpp and ch7_1_bool2.ll).

  ====================================  =======  ============================================
  Optimized legalized selection DAG     Cpu0     pattern in Cpu0InstrInfo.td
  ====================================  =======  ============================================
  load ..., <..., sext from i8>         lb       LB  : LoadM32<0x03, "lb",  sextloadi8>;
  load ..., <..., zext from i8>         lbu      LBu : LoadM32<0x04, "lbu", zextloadi8>;
  load ..., <..., sext from i16>        lh       LH  : LoadM32<0x06, "lh",  sextloadi16_a>;
  load ..., <..., zext from i16>        lhu      LHu : LoadM32<0x07, "lhu", zextloadi16_a>;
  store ...,<..., trunc to i16>         sh       SH  : StoreM32<0x08, "sh", truncstorei16_a>;
  store ...,<..., trunc to i8>          sb       SB  : StoreM32<0x05, "sb", truncstorei8>;
  ====================================  =======  ============================================
  

long long
----------

Like MIPS, the long type in Cpu0 is 32-bit, while long long is 64-bit in C.
To support long long, add the following code to Chapter7_1/:

.. rubric:: lbdex/chapters/Chapter7_1/Cpu0SEISelDAGToDAG.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelDAGToDAG.cpp
    :start-after: #if CH >= CH7_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelDAGToDAG.cpp
    :start-after: //@selectNode
    :end-before: #if CH >= CH7_1 //2
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelDAGToDAG.cpp
    :start-after: #if CH >= CH7_1 //2
    :end-before: #endif

.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter7_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@class Cpu0TargetLowering
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH7_1 //2
    :end-before: #endif

.. code-block:: c++

      ...
    }

.. rubric:: lbdex/chapters/Chapter7_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH7_1 //2
    :end-before: #endif

.. code-block:: c++

    ...
  }

The additional code in Cpu0ISelLowering.cpp handles shift operations for long 
long (64-bit).
Using the << and >> operators on 64-bit variables generates DAG SHL_PARTS,
SRA_PARTS, and SRL_PARTS, which manage 32-bit operands during LLVM DAG 
translation.

At this point, ch9_7.cpp, which includes 64-bit shift operations, cannot be 
executed.
It will be verified in the later chapter Function Call.

Run Chapter7_1/ with ch7_1_longlong.cpp to obtain the following result.

.. rubric:: lbdex/input/ch7_1_longlong.cpp
.. literalinclude:: ../lbdex/input/ch7_1_longlong.cpp
    :start-after: /// start

.. code-block:: console

  1-160-134-62:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch7_1_longlong.cpp -emit-llvm -o ch7_1_longlong.bc
  1-160-134-62:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm 
  ch7_1_longlong.bc -o -
    ...
  # BB#0:
	  addiu	$sp, $sp, -72
	  st	$8, 68($fp)             # 4-byte Folded Spill
	  addiu	$2, $zero, 2
	  st	$2, 60($fp)
	  addiu	$2, $zero, 3
	  st	$2, 56($fp)
	  addiu	$2, $zero, 1
	  st	$2, 52($fp)
	  st	$2, 48($fp)
	  lui	$2, 768
	  ori	$2, $2, 4096
	  st	$2, 44($fp)
	  lui	$2, 512
	  ori	$2, $2, 4096
	  st	$2, 40($fp)
	  ld	$2, 52($fp)
	  ld	$3, 60($fp)
	  addu	$3, $3, $2
	  ld	$4, 56($fp)
	  ld	$5, 48($fp)
	  st	$3, 36($fp)
	  cmp	$sw, $3, $2
	  andi	$2, $sw, 1
	  addu	$2, $2, $5
	  addu	$2, $4, $2
	  st	$2, 32($fp)
	  ld	$2, 52($fp)
	  ld	$3, 60($fp)
	  subu	$4, $3, $2
	  ld	$5, 56($fp)
	  ld	$t9, 48($fp)
	  st	$4, 28($fp)
	  cmp	$sw, $3, $2
	  andi	$2, $sw, 1
	  addu	$2, $2, $t9
	  subu	$2, $5, $2
	  st	$2, 24($fp)
	  ld	$2, 52($fp)
	  ld	$3, 60($fp)
	  multu	$3, $2
	  mflo	$4
	  mfhi	$5
	  ld	$t9, 56($fp)
	  ld	$7, 48($fp)
	  st	$4, 20($fp)
	  mul	$3, $3, $7
	  addu	$3, $5, $3
	  mul	$2, $t9, $2
	  addu	$2, $3, $2
	  st	$2, 16($fp)
	  ld	$2, 40($fp)
	  ld	$3, 44($fp)
	  mult	$3, $2
	  mflo	$2
	  mfhi	$4
	  st	$2, 12($fp)
	  st	$4, 8($fp)
	  ld	$5, 28($fp)
	  ld	$3, 36($fp)
	  addu	$t9, $3, $5
	  ld	$7, 20($fp)
	  addu	$8, $t9, $7
	  addu	$3, $8, $2
	  cmp	$sw, $3, $2
	  andi	$2, $sw, 1
	  addu	$2, $2, $4
	  cmp	$sw, $t9, $5
	  st	$sw, 4($fp)             # 4-byte Folded Spill
	  cmp	$sw, $8, $7
	  andi	$4, $sw, 1
	  ld	$5, 16($fp)
	  addu	$4, $4, $5
	  ld	$sw, 4($fp)             # 4-byte Folded Reload
	  andi	$5, $sw, 1
	  ld	$t9, 24($fp)
	  addu	$5, $5, $t9
	  ld	$t9, 32($fp)
	  addu	$5, $t9, $5
	  addu	$4, $5, $4
	  addu	$2, $4, $2
	  ld	$8, 68($fp)             # 4-byte Folded Reload
	  addiu	$sp, $sp, 72
	  ret	$lr
    ...

float and double
-----------------

At this point, Cpu0 only supports integer instructions.
For floating-point operations, the Cpu0 backend calls library functions
to convert integers to floats, as follows:

.. rubric:: lbdex/input/ch7_1_fmul.c
.. literalinclude:: ../lbdex/input/ch7_1_fmul.c

Floating-point function calls for Cpu0 will be supported in the Function Call 
chapter.
Due to hardware cost constraints, many CPUs do not include floating-point 
hardware instructions.
Instead, they rely on library functions.
MIPS separates floating-point operations into a dedicated co-processor for 
applications that require floating-point arithmetic.

To support the floating-point library (part of compiler-rt) 
[#lbt-compiler-rt]_, the following code is added to support clz and clo 
instructions.
Although these instructions are implemented in compiler-rt, they
are integer operations that improve floating-point application performance.

.. rubric:: lbdex/chapters/Chapter7_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH7_1 4
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH7_1 6
    :end-before: //@def LEA_ADDiu {

Array and struct support
-------------------------

LLVM uses getelementptr to represent array and struct types in C.
For details, refer to [#]_.

For ch7_1_globalstructoffset.cpp, the LLVM IR is as follows:

.. rubric:: lbdex/input/ch7_1_globalstructoffset.cpp
.. literalinclude:: ../lbdex/input/ch7_1_globalstructoffset.cpp
    :start-after: /// start

.. code-block:: console

  // ch7_1_globalstructoffset.ll
  ; ModuleID = 'ch7_1_globalstructoffset.bc'
  ...
  %struct.Date = type { i32, i32, i32 }

  @date = global %struct.Date { i32 2012, i32 10, i32 12 }, align 4
  @a = global [3 x i32] [i32 2012, i32 10, i32 12], align 4

  ; Function Attrs: nounwind
  define i32 @_Z11test_structv() #0 {
    %day = alloca i32, align 4
    %i = alloca i32, align 4
    %1 = load i32* getelementptr inbounds (%struct.Date* @date, i32 0, i32 2), align 4
    store i32 %1, i32* %day, align 4
    %2 = load i32* getelementptr inbounds ([3 x i32]* @a, i32 0, i32 1), align 4
    store i32 %2, i32* %i, align 4
    %3 = load i32* %i, align 4
    %4 = load i32* %day, align 4
    %5 = add nsw i32 %3, %4
    ret i32 %5
  }
    
Run Chapter6_1/ with ch7_1_globalstructoffset.bc on static mode will get the 
incorrect asm file as follows,

.. code-block:: console

  1-160-134-62:input Jonathan$ /Users/Jonathan/llvm/test/build/bin/
  llc -march=cpu0 -relocation-model=static -filetype=asm 
  ch7_1_globalstructoffset.bc -o -
    ...
    lui $2, %hi(date)
    ori $2, $2, %lo(date)
    ld  $2, 0($2)   // the correct one is   ld  $2, 8($2)
    ...

For day = date.day, the correct instruction is 
**ld $2, 8($2), not ld $2, 0($2),** 
since date.day has an offset of 8 bytes (the date struct contains year and 
month before day).
Use the debug option in llc to analyze this:

.. code-block:: console

  jonathantekiimac:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -debug -relocation-model=static 
  -filetype=asm ch6_2.bc -o ch6_2.cpu0.static.s
  ...
  === main
  Initial selection DAG: BB#0 'main:entry'
  SelectionDAG has 20 nodes:
    0x7f7f5b02d210: i32 = undef [ORD=1]
    
        0x7f7f5ac10590: ch = EntryToken [ORD=1]
    
        0x7f7f5b02d010: i32 = Constant<0> [ORD=1]
    
        0x7f7f5b02d110: i32 = FrameIndex<0> [ORD=1]
    
        0x7f7f5b02d210: <multiple use>
      0x7f7f5b02d310: ch = store 0x7f7f5ac10590, 0x7f7f5b02d010, 0x7f7f5b02d110, 
      0x7f7f5b02d210<ST4[%retval]> [ORD=1]
    
        0x7f7f5b02d410: i32 = GlobalAddress<%struct.Date* @date> 0 [ORD=2]
    
        0x7f7f5b02d510: i32 = Constant<8> [ORD=2]
    
      0x7f7f5b02d610: i32 = add 0x7f7f5b02d410, 0x7f7f5b02d510 [ORD=2]
    
      0x7f7f5b02d210: <multiple use>
    0x7f7f5b02d710: i32,ch = load 0x7f7f5b02d310, 0x7f7f5b02d610, 0x7f7f5b02d210
    <LD4[getelementptr inbounds (%struct.Date* @date, i32 0, i32 2)]> [ORD=3]
    
    0x7f7f5b02db10: i64 = Constant<4>
    
        0x7f7f5b02d710: <multiple use>
        0x7f7f5b02d710: <multiple use>
        0x7f7f5b02d810: i32 = FrameIndex<1> [ORD=4]
  
        0x7f7f5b02d210: <multiple use>
      0x7f7f5b02d910: ch = store 0x7f7f5b02d710:1, 0x7f7f5b02d710, 0x7f7f5b02d810,
       0x7f7f5b02d210<ST4[%day]> [ORD=4]
  
        0x7f7f5b02da10: i32 = GlobalAddress<[3 x i32]* @a> 0 [ORD=5]
    
        0x7f7f5b02dc10: i32 = Constant<4> [ORD=5]
    
      0x7f7f5b02dd10: i32 = add 0x7f7f5b02da10, 0x7f7f5b02dc10 [ORD=5]
    
      0x7f7f5b02d210: <multiple use>
    0x7f7f5b02de10: i32,ch = load 0x7f7f5b02d910, 0x7f7f5b02dd10, 0x7f7f5b02d210
    <LD4[getelementptr inbounds ([3 x i32]* @a, i32 0, i32 1)]> [ORD=6]
    
  ...
    
    
  Replacing.3 0x7f7f5b02dd10: i32 = add 0x7f7f5b02da10, 0x7f7f5b02dc10 [ORD=5]
    
  With: 0x7f7f5b030010: i32 = GlobalAddress<[3 x i32]* @a> + 4
    
    
  Replacing.3 0x7f7f5b02d610: i32 = add 0x7f7f5b02d410, 0x7f7f5b02d510 [ORD=2]
    
  With: 0x7f7f5b02db10: i32 = GlobalAddress<%struct.Date* @date> + 8
    
  Optimized lowered selection DAG: BB#0 'main:entry'
  SelectionDAG has 15 nodes:
    0x7f7f5b02d210: i32 = undef [ORD=1]
    
        0x7f7f5ac10590: ch = EntryToken [ORD=1]
    
        0x7f7f5b02d010: i32 = Constant<0> [ORD=1]
    
        0x7f7f5b02d110: i32 = FrameIndex<0> [ORD=1]
    
        0x7f7f5b02d210: <multiple use>
      0x7f7f5b02d310: ch = store 0x7f7f5ac10590, 0x7f7f5b02d010, 0x7f7f5b02d110, 
      0x7f7f5b02d210<ST4[%retval]> [ORD=1]
    
      0x7f7f5b02db10: i32 = GlobalAddress<%struct.Date* @date> + 8
    
      0x7f7f5b02d210: <multiple use>
    0x7f7f5b02d710: i32,ch = load 0x7f7f5b02d310, 0x7f7f5b02db10, 0x7f7f5b02d210
    <LD4[getelementptr inbounds (%struct.Date* @date, i32 0, i32 2)]> [ORD=3]
    
        0x7f7f5b02d710: <multiple use>
        0x7f7f5b02d710: <multiple use>
        0x7f7f5b02d810: i32 = FrameIndex<1> [ORD=4]
    
        0x7f7f5b02d210: <multiple use>
      0x7f7f5b02d910: ch = store 0x7f7f5b02d710:1, 0x7f7f5b02d710, 0x7f7f5b02d810,
       0x7f7f5b02d210<ST4[%day]> [ORD=4]
    
      0x7f7f5b030010: i32 = GlobalAddress<[3 x i32]* @a> + 4
    
      0x7f7f5b02d210: <multiple use>
    0x7f7f5b02de10: i32,ch = load 0x7f7f5b02d910, 0x7f7f5b030010, 0x7f7f5b02d210
    <LD4[getelementptr inbounds ([3 x i32]* @a, i32 0, i32 1)]> [ORD=6]
    
  ...


The output reveals the DAG translation process.
As shown, the DAG node for date.day
(add GlobalAddress<[3 x i32]* @a> 0, Constant<8>) with three nodes is replaced 
by a single node GlobalAddress<%struct.Date* @date> + 8.
The same applies to a[1].

This replacement occurs because TargetLowering.cpp::isOffsetFoldingLegal(...)
returns true in ``llc -static`` static addressing mode.
In Cpu0, the **ld** instruction format is **ld $r1, offset($r2)**,
meaning it loads the value at address($r2) + offset into $r1.
To correct this, override isOffsetFoldingLegal(...) as follows:

.. rubric:: lib/CodeGen/SelectionDAG/TargetLowering.cpp

.. code-block:: c++

  bool
  TargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
    // Assume that everything is safe in static mode.
    if (getTargetMachine().getRelocationModel() == Reloc::Static)
      return true;
    
    // In dynamic-no-pic mode, assume that known defined values are safe.
    if (getTargetMachine().getRelocationModel() == Reloc::DynamicNoPIC &&
       GA &&
       !GA->getGlobal()->isDeclaration() &&
       !GA->getGlobal()->isWeakForLinker())
    return true;
    
    // Otherwise assume nothing is safe.
    return false;
  }
    
.. rubric:: lbdex/chapters/Chapter7_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH7_1 //4
    :end-before: #endif

Additionally, add the following code to Cpu0ISelDAGToDAG.cpp:

When SelectAddr(...) in Cpu0ISelDAGToDAG.cpp is called,
Addr represents the DAG node for date.day:

.. rubric:: lbdex/chapters/Chapter7_1/Cpu0ISelDAGToDAG.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: //@SelectAddr {
    :end-before: //@SelectAddr }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: #if CH >= CH7_1 //1
    :end-before: #endif

.. code-block:: c++
  
    ...
  }

Recall that we have translated the DAG list for ``date.day``  
``(add GlobalAddress<[3 x i32]* @a> 0, Constant<8>)`` into  

``(add (add Cpu0ISD::Hi (Cpu0II::MO_ABS_HI), Cpu0ISD::Lo(Cpu0II::MO_ABS_LO)),``  
``Constant<8>)``  

by the following code in ``Cpu0ISelLowering.h``.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@getAddrNonPIC
    :end-before: #endif // #if CH >= CH6_1

add (add Cpu0ISD::Hi (Cpu0II::MO_ABS_HI), Cpu0ISD::Lo(Cpu0II::MO_ABS_LO)), Constant<8>

Since Addr.getOpcode() = ISD:ADD,
Addr.getOperand(0) = (add Cpu0ISD::Hi (Cpu0II::MO_ABS_HI), Cpu0ISD::Lo(Cpu0II::MO_ABS_LO)),
and Addr.getOperand(1).getOpcode() = ISD::Constant,
we set Base to (add Cpu0ISD::Hi (Cpu0II::MO_ABS_HI), Cpu0ISD::Lo(Cpu0II::MO_ABS_LO))
and Offset to Constant<8>.
This ensures ld $r1, 8($r2) is correctly generated in the Instruction Selection stage.

Run Chapter7_1/ with ch7_1_globalstructoffset.cpp to obtain the correct instruction.


.. code-block:: console

    ...
	  lui	$2, %hi(date)
	  ori	$2, $2, %lo(date)
	  ld	$2, 8($2)   // correct
    ...

The ch7_1_localarrayinit.cpp is for local variable initialization test. 
The result as follows,

.. rubric:: lbdex/input/ch7_1_localarrayinit.cpp
.. literalinclude:: ../lbdex/input/ch7_1_localarrayinit.cpp
    :start-after: /// start

.. code-block:: console

  118-165-79-206:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch7_1_localarrayinit.cpp -emit-llvm -o ch7_1_localarrayinit.bc
  118-165-79-206:input Jonathan$ llvm-dis ch7_1_localarrayinit.bc -o -
  ...
  
.. code-block:: llvm

  define i32 @main() nounwind ssp {
  entry:
    %retval = alloca i32, align 4
    %a = alloca [3 x i32], align 4
    store i32 0, i32* %retval
    %0 = bitcast [3 x i32]* %a to i8*
    call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* bitcast ([3 x i32]* 
      @_ZZ4mainE1a to i8*), i32 12, i32 4, i1 false)
    ret i32 0
  }
  ; Function Attrs: nounwind
  declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) #1

.. code-block:: console

  118-165-79-206:input Jonathan$ ~/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch7_1_localarrayinit.bc -o -
	  ...
  # BB#0:                                 # %entry
	  addiu	$sp, $sp, -16
	  addiu	$2, $zero, 0
	  st	$2, 12($fp)
	  ld	$2, %got($_ZZ4mainE1a)($gp)
	  ori	$2, $2, %lo($_ZZ4mainE1a)
	  ld	$3, 8($2)
	  st	$3, 8($fp)
	  ld	$3, 4($2)
	  st	$3, 4($fp)
	  ld	$2, 0($2)
	  st	$2, 0($fp)
	  addiu	$sp, $sp, 16
	  ret	$lr
	  ...
	  .type	$_ZZ4mainE1a,@object    # @_ZZ4mainE1a
	  .section	.rodata,"a",@progbits
	  .align	2
  $_ZZ4mainE1a:
	  .4byte	0                       # 0x0
	  .4byte	1                       # 0x1
	  .4byte	2                       # 0x2
	  .size	$_ZZ4mainE1a, 12


Vector type (SIMD) support
---------------------------

Vector types are used when multiple primitive data are operated in parallel 
using a single instruction (SIMD) [#vector]_. Mips supports the 
following llvm IRs "icmp slt" and "sext" for vector type, Cpu0 supports them
either.


Vector types enable multiple primitive data operations in parallel
using a single instruction (SIMD) [#vector]_.
MIPS supports **icmp slt** and **sext** LLVM IRs for vector types, which Cpu0 
also supports.

.. rubric:: lbdex/input/ch7_1_vector.cpp
.. literalinclude:: ../lbdex/input/ch7_1_vector.cpp
    :start-after: /// start

.. code-block:: console

  118-165-79-206:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch7_1_vector.cpp -emit-llvm -o ch7_1_vector.bc
  118-165-79-206:input Jonathan$ ~/llvm/test/build/bin/
  llvm-dis ch7_1_vector.bc -o -  
  ...
  
.. code-block:: llvm

  ; Function Attrs: nounwind
  define i32 @_Z16test_cmplt_shortv() #0 {
    %a0 = alloca <4 x i32>, align 16
    %b0 = alloca <4 x i32>, align 16
    %c0 = alloca <4 x i32>, align 16
    store volatile <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <4 x i32>* %a0, align 16
    store volatile <4 x i32> <i32 2, i32 2, i32 2, i32 2>, <4 x i32>* %b0, align 16
    %1 = load volatile <4 x i32>, <4 x i32>* %a0, align 16
    %2 = load volatile <4 x i32>, <4 x i32>* %b0, align 16
    %3 = icmp slt <4 x i32> %1, %2
    %4 = sext <4 x i1> %3 to <4 x i32>
    store volatile <4 x i32> %4, <4 x i32>* %c0, align 16
    %5 = load volatile <4 x i32>, <4 x i32>* %c0, align 16
    %6 = extractelement <4 x i32> %5, i32 0
    %7 = load volatile <4 x i32>, <4 x i32>* %c0, align 16
    %8 = extractelement <4 x i32> %7, i32 1
    %9 = add nsw i32 %6, %8
    %10 = load volatile <4 x i32>, <4 x i32>* %c0, align 16
    %11 = extractelement <4 x i32> %10, i32 2
    %12 = add nsw i32 %9, %11
    %13 = load volatile <4 x i32>, <4 x i32>* %c0, align 16
    %14 = extractelement <4 x i32> %13, i32 3
    %15 = add nsw i32 %12, %14
    ret i32 %15
  }

.. code-block:: console

  118-165-79-206:input Jonathan$ ~/llvm/test/build/bin/llc 
    -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm ch7_1_vector.bc 
    -o -
    .text
    .section .mdebug.abiO32
    .previous
    .file "ch7_1_vector.bc"
    .globl  _Z16test_cmplt_shortv
    .p2align  2
    .type _Z16test_cmplt_shortv,@function
    .ent  _Z16test_cmplt_shortv   # @_Z16test_cmplt_shortv
  _Z16test_cmplt_shortv:
    .frame  $fp,48,$lr
    .mask   0x00000000,0
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -48
    addiu $2, $zero, 3
    st  $2, 44($sp)
    addiu $2, $zero, 1
    st  $2, 36($sp)
    addiu $2, $zero, 0
    st  $2, 32($sp)
    addiu $2, $zero, 2
    st  $2, 40($sp)
    st  $2, 28($sp)
    st  $2, 24($sp)
    st  $2, 20($sp)
    st  $2, 16($sp)
    ld  $2, 32($sp)
    ld  $3, 44($sp)
    ld  $4, 40($sp)
    ld  $5, 36($sp)
    ld  $t9, 20($sp)
    slt $5, $5, $t9
    ld  $t9, 24($sp)
    slt $4, $4, $t9
    ld  $t9, 28($sp)
    slt $3, $3, $t9
    shl $3, $3, 31
    sra $3, $3, 31
    ld  $t9, 16($sp)
    st  $3, 12($sp)
    shl $3, $4, 31
    sra $3, $3, 31
    st  $3, 8($sp)
    shl $3, $5, 31
    sra $3, $3, 31
    st  $3, 4($sp)
    slt $2, $2, $t9
    shl $2, $2, 31
    sra $2, $2, 31
    st  $2, 0($sp)
    ld  $2, 12($sp)
    ld  $2, 8($sp)
    ld  $2, 4($sp)
    ld  $2, 0($sp)
    ld  $3, 4($sp)
    addu  $2, $2, $3
    ld  $3, 12($sp)
    ld  $3, 8($sp)
    ld  $3, 0($sp)
    ld  $3, 8($sp)
    addu  $2, $2, $3
    ld  $3, 12($sp)
    ld  $3, 4($sp)
    ld  $3, 0($sp)
    ld  $3, 12($sp)
    addu  $2, $2, $3
    ld  $3, 8($sp)
    ld  $3, 4($sp)
    ld  $3, 0($sp)
    addiu $sp, $sp, 48
    ret $lr
    .set  macro
    .set  reorder
    .end  _Z16test_cmplt_shortv
  $func_end0:
    .size _Z16test_cmplt_shortv, ($func_end0)-_Z16test_cmplt_shortv
  
  
    .ident  "Apple LLVM version 7.0.0 (clang-700.1.76)"
    .section  ".note.GNU-stack","",@progbits

  
Since test_longlong_shift2() in ch7_1_vector.cpp requires storeRegToStack()
in Cpu0SEInstInfo.cpp, it cannot be verified at this point.

.. rubric:: lbdex/chapters/Chapter7_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH7_1 //0.5
    :end-before: #endif
    
.. rubric:: lbdex/chapters/Chapter7_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH7_1 //3.5
    :end-before: #endif


.. [#] http://llvm.org/docs/LangRef.html#getelementptr-instruction

.. [#lbt-compiler-rt] http://jonathan2251.github.io/lbt/lib.html#compiler-rt

.. [#vector] http://llvm.org/docs/LangRef.html#vector-type
