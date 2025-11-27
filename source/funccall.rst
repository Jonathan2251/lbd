.. _sec-funccall:

Function call
==============

.. contents::
   :local:
   :depth: 4

This chapter introduces support for subroutine and function calls in backend 
translation. A significant amount of code is required to support function 
calls, and it is organized using LLVM-supplied interfaces for clarity.

The chapter begins by introducing the MIPS stack frame structure, as many 
parts of the ABI are borrowed from it. Although each CPU has its own ABI, 
most RISC CPU ABIs share similar characteristics.

Section "4.5 DAG Lowering" of *tricore_llvm.pdf* provides insight into the 
lowering process. Section "4.5.1 Calling Conventions" in the same document is 
also a helpful reference for further understanding.

If you have difficulty understanding the stack frame illustrated in the first 
three sections of this chapter, you may consult the following resources: 
Appendix B, "Procedure Call Convention," in *Computer Organization and Design, 
1st Edition* [#computer_arch_interface]_; "Run Time Memory" in a compiler 
textbook; or "Function Call Sequence" and "Stack Frame" in the MIPS ABI 
[#abi]_.

MIPS Stack Frame
----------------

The first step in designing Cpu0 function calls is deciding how to pass 
arguments. There are two options:

1. Pass all arguments on the stack.
2. Pass arguments using registers reserved for function arguments, and place 
   any remaining arguments on the stack once the registers are full.

For example, MIPS passes the first four arguments in registers `$a0`, `$a1`, 
`$a2`, and `$a3`. Any additional arguments are passed on the stack. 
:numref:`funccall-f1` illustrates the MIPS stack frame.

.. _funccall-f1:
.. figure:: ../Fig/funccall/1.png
    :height: 531 px
    :width: 688 px
    :scale: 50 %
    :align: center

    Mips stack frame
    
Run ``llc -march=mips`` on ``ch9_1.bc``, and you will get the following result. 
See the comments marked with **"//"**.

.. rubric:: lbdex/input/ch9_1.cpp
.. literalinclude:: ../lbdex/input/ch9_1.cpp
    :start-after: /// start

.. code-block:: console

  118-165-78-230:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_1.cpp -emit-llvm -o ch9_1.bc
  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=mips -relocation-model=pic -filetype=asm ch9_1.bc -o 
  ch9_1.mips.s
  118-165-78-230:input Jonathan$ cat ch9_1.mips.s 
    .section .mdebug.abi32
    .previous
    .file "ch9_1.bc"
    .text
    .globl  _Z5sum_iiiiiii
    .align  2
    .type _Z5sum_iiiiiii,@function
    .set  nomips16                # @_Z5sum_iiiiiii
    .ent  _Z5sum_iiiiiii
  _Z5sum_iiiiiii:
    .cfi_startproc
    .frame  $sp,32,$ra
    .mask   0x00000000,0
    .fmask  0x00000000,0
    .set  noreorder
    .set  nomacro
    .set  noat
  # BB#0:
    addiu $sp, $sp, -32
  $tmp1:
    .cfi_def_cfa_offset 32
    sw  $4, 28($sp)
    sw  $5, 24($sp)
    sw  $t9, 20($sp)
    sw  $7, 16($sp)
    lw  $1, 48($sp) // load argument 5
    sw  $1, 12($sp)
    lw  $1, 52($sp) // load argument 6
    sw  $1, 8($sp)
    lw  $2, 24($sp)
    lw  $3, 28($sp)
    addu  $2, $3, $2
    lw  $3, 20($sp)
    addu  $2, $2, $3
    lw  $3, 16($sp)
    addu  $2, $2, $3
    lw  $3, 12($sp)
    addu  $2, $2, $3
    addu  $2, $2, $1
    sw  $2, 4($sp)
    jr  $ra
    addiu $sp, $sp, 32
    .set  at
    .set  macro
    .set  reorder
    .end  _Z5sum_iiiiiii
  $tmp2:
    .size _Z5sum_iiiiiii, ($tmp2)-_Z5sum_iiiiiii
    .cfi_endproc
  
    .globl  main
    .align  2
    .type main,@function
    .set  nomips16                # @main
    .ent  main
  main:
    .cfi_startproc
    .frame  $sp,40,$ra
    .mask   0x80000000,-4
    .fmask  0x00000000,0
    .set  noreorder
    .set  nomacro
    .set  noat
  # BB#0:
    lui $2, %hi(_gp_disp)
    ori $2, $2, %lo(_gp_disp)
    addiu $sp, $sp, -40
  $tmp5:
    .cfi_def_cfa_offset 40
    sw  $ra, 36($sp)            # 4-byte Folded Spill
  $tmp6:
    .cfi_offset 31, -4
    addu  $gp, $2, $25
    sw  $zero, 32($sp)
    addiu $1, $zero, 6
    sw  $1, 20($sp) // Save argument 6 to 20($sp)
    addiu $1, $zero, 5
    sw  $1, 16($sp) // Save argument 5 to 16($sp)
    lw  $25, %call16(_Z5sum_iiiiiii)($gp)
    addiu $4, $zero, 1    // Pass argument 1 to $4 (=$a0)
    addiu $5, $zero, 2    // Pass argument 2 to $5 (=$a1)
    addiu $t9, $zero, 3
    jalr  $25
    addiu $7, $zero, 4
    sw  $2, 28($sp)
    lw  $ra, 36($sp)            # 4-byte Folded Reload
    jr  $ra
    addiu $sp, $sp, 40
    .set  at
    .set  macro
    .set  reorder
    .end  main
  $tmp7:
    .size main, ($tmp7)-main
    .cfi_endproc


From the MIPS assembly code generated above, we can see that the first four 
arguments are saved in registers `$a0` to `$a3`, and the last two arguments 
are saved at memory locations `16($sp)` and `20($sp)`.

:numref:`funccall-f2` shows the location of the arguments in the example code 
`ch9_1.cpp`.

In the `sum_i()` function, argument 5 is loaded from `48($sp)` because it was 
stored at `16($sp)` in the `main()` function. Since the stack size of 
`sum_i()` is 32, the address of the incoming argument 5 is calculated as 
`16 + 32 = 48($sp)`.

.. _funccall-f2:
.. figure:: ../Fig/funccall/2.png
    :height: 577 px
    :width: 740 px
    :scale: 50 %
    :align: center

    Mips arguments location in stack frame


The document *007-2418-003.pdf* referenced in [#mipsasm]_ is the MIPS assembly 
language manual. The MIPS Application Binary Interface, referenced in [#abi]_, 
includes the diagram shown in :numref:`funccall-f1`.

Load Incoming Arguments from Stack Frame
----------------------------------------

As discussed in the previous section, supporting function calls requires 
implementing an argument-passing mechanism using the stack frame.

Before proceeding with the implementation, let’s run the old version of the 
code in `Chapter8_2/` with `ch9_1.cpp` and observe what happens.

.. code-block:: console

  118-165-79-31:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm 
  ch9_1.bc -o ch9_1.cpu0.s
  Assertion failed: (InVals.size() == Ins.size() && "LowerFormalArguments didn't 
  emit the correct number of values!"), function LowerArguments, file /Users/
  Jonathan/llvm/test/llvm/lib/CodeGen/SelectionDAG/
  SelectionDAGBuilder.cpp, ...
  ...
  0.  Program arguments: /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_1.bc -o 
  ch9_1.cpu0.s 
  1.  Running pass 'Function Pass Manager' on module 'ch9_1.bc'.
  2.  Running pass 'CPU0 DAG->DAG Pattern Instruction Selection' on function 
  '@_Z5sum_iiiiiii'
  Illegal instruction: 4

Since `Chapter8_2/` defines `LowerFormalArguments()` with an empty body, we 
receive the error messages shown above.

Before implementing `LowerFormalArguments()`, we must first decide how to pass 
arguments in a function call.

For demonstration purposes, Cpu0 passes the first two arguments in registers by 
default, which corresponds to the setting ``llc -cpu0-s32-calls=false``.

When using ``llc -cpu0-s32-calls=true``, Cpu0 passes all its arguments on the 
stack.

The function `LowerFormalArguments()` is responsible for creating the incoming 
arguments. We define it as follows:

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@class Cpu0TargetLowering
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@CH3_4 1 {
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_1 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_1 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_1 //4
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_1 //5
    :end-before: #endif

.. code-block:: c++

    };
    ...

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_1 //6
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_1 //7
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_1 //8
    :end-before: #endif

.. code-block:: c++

    ...
  }


.. rubric:: lbdex/chapters/Chapter9_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //4
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //5
    :end-before: #if CH >= CH9_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //5
    :end-before: #if CH >= CH9_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #else // CH >= CH9_2
    :end-before: #endif

.. code-block:: c++

  }

.. literalinclude:: ../lbdex/chapters/Chapter9_1/Cpu0ISelLowering.cpp
    :start-after: //@            Formal Arguments Calling Convention Implementation
    :end-before: //@              Return Value Calling Convention Implementation
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //7
    :end-before: #endif // #if CH >= CH9_1
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //8
    :end-before: #endif // #if CH >= CH9_1
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //9
    :end-before: #endif // #if CH >= CH9_1
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //10
    :end-before: #endif

As reviewed in the section "Global variable" [#secglobal]_, we handled global 
variable translation by first creating the IR DAG in `LowerGlobalAddress()`, 
and then completing instruction selection based on the corresponding machine 
instruction DAGs in `Cpu0InstrInfo.td`.

`LowerGlobalAddress()` is called when `llc` encounters a global variable 
access. Similarly, `LowerFormalArguments()` is called when entering a function. 

Before entering the **“for loop”**, it gathers incoming argument information 
using `CCInfo(CallConv, ..., ArgLocs, ...)`.

In `ch9_1.cpp`, the function `sum_i(...)` has 6 arguments. Thus, 
`ArgLocs.size()` is 6, with each argument's information stored in `ArgLocs[i]`.

- If `VA.isRegLoc()` returns true, the argument is passed via register.
- If `VA.isMemLoc()` returns true, the argument is passed via memory stack.

For register-passed arguments, the register is marked as "live-in", and the 
value is copied directly from the register.

For stack-passed arguments, a stack offset is created for the frame index 
object. A load node is then created using this offset and added to the `InVals` 
vector.

When using ``llc -cpu0-s32-calls=false``, the first two arguments are passed in 
registers, and the remaining arguments are passed in the stack frame.

When using ``llc -cpu0-s32-calls=true``, all arguments are passed in the stack 
frame.

Before handling arguments, `analyzeFormalArguments()` is called. Inside it, 
`fixedArgFn()` is used to return the function pointer to either 
`CC_Cpu0O32()` or `CC_Cpu0S32()`.

`ArgFlags.isByVal()` will be true for "struct pointer byval" arguments, such as 
`%struct.S* byval` in `tailcall.ll`.

With ``llc -cpu0-s32-calls=false``, the stack offset begins at 8 (to allow space 
in case argument registers are spilled). With ``llc -cpu0-s32-calls=true``, the 
stack offset begins at 0.

For example, when running `ch9_1.cpp` with ``llc -cpu0-s32-calls=true`` 
(memory stack only), `LowerFormalArguments()` will be called twice:

- First, for `sum_i()`, it will create six load DAGs for the six incoming 
  arguments.
- Second, for `main()`, no load DAG is created, as there are no incoming 
  arguments.

In addition to `LowerFormalArguments()`, we use 
`loadRegFromStackSlot()` (defined in an earlier chapter) to generate the 
machine instruction **“ld $r, offset($sp)”**, which loads arguments from the 
stack frame.

`GetMemOperand(..., FI, ...)` returns the memory location of the frame index 
variable, representing the offset.

For the input `ch9_incoming.cpp` shown below, `LowerFormalArguments()` will 
generate the red-boxed DAG nodes illustrated in :numref:`funccall-f-incoming-arg1` 
and :numref:`funccall-f-incoming-arg2`, corresponding to 
``llc -cpu0-s32-calls=true`` and ``llc -cpu0-s32-calls=false``, respectively.

The root node at the bottom is created by:

.. rubric:: lbdex/input/ch9_incoming.cpp
.. literalinclude:: ../lbdex/input/ch9_incoming.cpp
    :start-after: /// start
    
.. code-block:: console
  
  JonathantekiiMac:input Jonathan$ clang -O3 -target mips-unknown-linux-gnu -c 
  ch9_incoming.cpp -emit-llvm -o ch9_incoming.bc
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llvm-dis ch9_incoming.bc -o -
  ...
  define i32 @_Z5sum_iiii(i32 %x1, i32 %x2, i32 %x3) #0 {
    %1 = add nsw i32 %x2, %x1
    %2 = add nsw i32 %1, %x3
    ret i32 %2
  }

.. _funccall-f-incoming-arg1:
.. graphviz:: ../Fig/funccall/incoming-arg-S32.gv
   :caption: Incoming arguments DAG created for ch9_incoming.cpp with -cpu0-s32-calls=true

.. _funccall-f-incoming-arg2:
.. graphviz:: ../Fig/funccall/incoming-arg-O32.gv
   :caption: Incoming arguments DAG created for ch9_incoming.cpp with -cpu0-s32-calls=false


In addition to the calling convention and `LowerFormalArguments()`, 
`Chapter9_1/` adds support for instruction selection and printing of the Cpu0 
instructions **swi** (software interrupt), **jsub**, and **jalr** (function call).
    
.. rubric:: lbdex/chapters/Chapter9_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //4
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //5
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //6
    :end-before: //@class TailCall
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //7
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //8
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //10
    :end-before: //@Pat<Cpu0TailCall>

.. code-block:: c++

  }
    
.. rubric:: lbdex/chapters/Chapter9_1/Cpu0MCInstLower.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: //@LowerSymbolOperand {
    :end-before: default:                   llvm_unreachable("Invalid target flag!");
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH9_1 //1
    :end-before: #endif

.. code-block:: c++

    ...
    }
    switch (MOTy) {
  . ...

.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH9_1 //2
    :end-before: #endif

.. code-block:: c++

    ...
    }
    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: //@LowerOperand {
    :end-before: default: llvm_unreachable("unknown operand type");
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH9_1 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: //@1
    :end-before: #endif

.. code-block:: c++

    ...
    }
    ...
  }

.. rubric:: lbdex/chapters/Chapter9_1/MCTargetDesc/Cpu0AsmBackend.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0AsmBackend.cpp
    :start-after: //@adjustFixupValue {
    :end-before: default:
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0AsmBackend.cpp
    :start-after: #if CH >= CH9_1
    :end-before: #endif

.. code-block:: c++

    ...
    }
    ...
  }

.. rubric:: lbdex/chapters/Chapter9_1/MCTargetDesc/Cpu0ELFObjectWriter.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0ELFObjectWriter.cpp
    :start-after: //@GetRelocType {
    :end-before: default:
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0ELFObjectWriter.cpp
    :start-after: #if CH >= CH9_1
    :end-before: #endif

.. code-block:: c++

    ...
    }
    ...
  }

.. rubric:: lbdex/chapters/Chapter9_1/MCTargetDesc/Cpu0FixupKinds.h
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0FixupKinds.h
    :start-after: //@Fixups {
    :end-before: //@ Pure upper 32 bit fixup resulting in - R_CPU0_32.
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0FixupKinds.h
    :start-after: #if CH >= CH9_1
    :end-before: #endif

.. code-block:: c++

      ...
  . }

.. rubric:: lbdex/chapters/Chapter9_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: //@getJumpTargetOpValue {
    :end-before: #if CH >= CH8_1 //3
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: #if CH >= CH9_1 //1
    :end-before: #else
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: #endif //#if CH >= CH9_1 //1
    :end-before: else

.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: //@getExprOpValue {
    :end-before: //@getExprOpValue body {
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: //@switch {
    :end-before: //@switch }
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: #if CH >= CH9_1 //2
    :end-before: #endif

.. code-block:: c++

    ...
    }
  ...
  }
 
.. rubric:: lbdex/chapters/Chapter9_1/Cpu0MachineFunction.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: //@1 {
    :end-before: #if CH >= CH3_4 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH9_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH9_1 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH9_1 //3
    :end-before: #if CH >= CH9_3
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #endif //#if CH >= CH9_3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH9_1 //4
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH9_1 //5
    :end-before: #endif

.. code-block:: c++

    ...
  };

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0SEFrameLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.h
    :start-after: #if CH >= CH9_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0SEFrameLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #if CH >= CH9_1 //1
    :end-before: #endif

Both `JSUB` and `JALR`, defined in `Cpu0InstrInfo.td` as shown above, use the 
`Cpu0JmpLink` node. They are distinguishable by their operand types: `JSUB` 
uses an `imm` (immediate) operand, while `JALR` uses a register operand.

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH9_1 //10
    :end-before: //@Pat<Cpu0TailCall>

The code instructs TableGen to generate pattern-matching logic that first 
matches the `"imm"` operand for the `"tglobaladdr"` pattern. If that match 
fails, it then attempts to match the `"texternalsym"` pattern.

A user-defined function belongs to the `"tglobaladdr"` category. For example, 
the function `sum_i(...)` defined in `ch9_1.cpp` falls under `"tglobaladdr"`. 

On the other hand, functions implicitly used by LLVM, such as `memcpy`, belong 
to `"texternalsym"`. The `memcpy` function is typically generated when defining 
a long string. The file `ch9_1_2.cpp` is an example that triggers a call to 
`memcpy`. This will be shown in the next section with the `Chapter9_2` example 
code.

The file `Cpu0GenDAGISel.inc` contains the pattern-matching information for 
`JSUB` and `JALR`, which is generated by TableGen as follows:

.. code-block:: c++

            /*SwitchOpcode*/ 74,  TARGET_VAL(Cpu0ISD::JmpLink),// ->734
  /*660*/     OPC_RecordNode,   // #0 = 'Cpu0JmpLink' chained node
  /*661*/     OPC_CaptureGlueInput,
  /*662*/     OPC_RecordChild1, // #1 = $target
  /*663*/     OPC_Scope, 57, /*->722*/ // 2 children in Scope
  /*665*/       OPC_MoveChild, 1,
  /*667*/       OPC_SwitchOpcode /*3 cases */, 22,  TARGET_VAL(ISD::Constant),
  // ->693
  /*671*/         OPC_MoveParent,
  /*672*/         OPC_EmitMergeInputChains1_0,
  /*673*/         OPC_EmitConvertToTarget, 1,
  /*675*/         OPC_Scope, 7, /*->684*/ // 2 children in Scope
  /*684*/         /*Scope*/ 7, /*->692*/
  /*685*/           OPC_MorphNodeTo, TARGET_VAL(Cpu0::JSUB), 0|OPFL_Chain|
  OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                        0/*#VTs*/, 1/*#Ops*/, 2, 
                    // Src: (Cpu0JmpLink (imm:iPTR):$target) - Complexity = 6
                    // Dst: (JSUB (imm:iPTR):$target)
  /*692*/         0, /*End of Scope*/
                /*SwitchOpcode*/ 11,  TARGET_VAL(ISD::TargetGlobalAddress),// ->707
  /*696*/         OPC_CheckType, MVT::i32,
  /*698*/         OPC_MoveParent,
  /*699*/         OPC_EmitMergeInputChains1_0,
  /*700*/         OPC_MorphNodeTo, TARGET_VAL(Cpu0::JSUB), 0|OPFL_Chain|
  OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                      0/*#VTs*/, 1/*#Ops*/, 1, 
                  // Src: (Cpu0JmpLink (tglobaladdr:i32):$dst) - Complexity = 6
                  // Dst: (JSUB (tglobaladdr:i32):$dst)
                /*SwitchOpcode*/ 11,  TARGET_VAL(ISD::TargetExternalSymbol),// ->721
  /*710*/         OPC_CheckType, MVT::i32,
  /*712*/         OPC_MoveParent,
  /*713*/         OPC_EmitMergeInputChains1_0,
  /*714*/         OPC_MorphNodeTo, TARGET_VAL(Cpu0::JSUB), 0|OPFL_Chain|
  OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                      0/*#VTs*/, 1/*#Ops*/, 1, 
                  // Src: (Cpu0JmpLink (texternalsym:i32):$dst) - Complexity = 6
                  // Dst: (JSUB (texternalsym:i32):$dst)
                0, // EndSwitchOpcode
  /*722*/     /*Scope*/ 10, /*->733*/
  /*723*/       OPC_CheckChild1Type, MVT::i32,
  /*725*/       OPC_EmitMergeInputChains1_0,
  /*726*/       OPC_MorphNodeTo, TARGET_VAL(Cpu0::JALR), 0|OPFL_Chain|
  OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                    0/*#VTs*/, 1/*#Ops*/, 1, 
                // Src: (Cpu0JmpLink CPURegs:i32:$rb) - Complexity = 3
                // Dst: (JALR CPURegs:i32:$rb)
  /*733*/     0, /*End of Scope*/


After applying the above changes, you can run `Chapter9_1/` with `ch9_1.cpp` 
and observe the results as shown below:

.. code-block:: console

  118-165-79-83:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm 
  ch9_1.bc -o ch9_1.cpu0.s
  Assertion failed: ((CLI.IsTailCall || InVals.size() == CLI.Ins.size()) && 
  "LowerCall didn't emit the correct number of values!"), function LowerCallTo, 
  file /Users/Jonathan/llvm/test/llvm/lib/CodeGen/SelectionDAG/SelectionDAGBuilder.
  cpp, ...
  ...
  0.  Program arguments: /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_1.bc -o 
  ch9_1.cpu0.s 
  1.  Running pass 'Function Pass Manager' on module 'ch9_1.bc'.
  2.  Running pass 'CPU0 DAG->DAG Pattern Instruction Selection' on function 
  '@main'
  Illegal instruction: 4
  
Now, the LowerFormalArguments() has the correct number, but LowerCall() has not  
the correct number of values!


Store Outgoing Arguments to Stack Frame
---------------------------------------

:numref:`funccall-f2` illustrates two steps involved in argument passing:

1. Storing outgoing arguments in the caller function.
2. Loading incoming arguments in the callee function.

In the previous section, we implemented `LowerFormalArguments()` to handle 
**"loading incoming arguments"** in the callee function.

Now, we will implement the part responsible for **"storing outgoing arguments"** 
in the caller function.

This task is handled by the `LowerCall()` function. Its implementation is shown 
below:

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0MachineFunction.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH9_2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0MachineFunction.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.cpp
    :start-after: #if CH >= CH9_2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_2 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@CH3_4 1 {
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_2 //2
    :end-before: #endif

.. code-block:: c++

  .  };

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_2 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_2 //4
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_2 //5
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_2 //6
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_2 //7
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_2 //1
    :end-before: #endif // #if CH >= CH9_2 //1
.. literalinclude:: ../lbdex/chapters/Chapter9_2/Cpu0ISelLowering.cpp
    :start-after: //@LowerCall {
    :end-before: //@LowerCall }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_2 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_2 //4
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_2 //5
    :end-before: #endif
.. literalinclude:: ../lbdex/chapters/Chapter9_2/Cpu0ISelLowering.cpp
    :start-after: //@#if CH >= CH9_2 //6 {
    :end-before: //@#if CH >= CH9_2 //6 }

Just like loading incoming arguments from the stack frame, we call 
`CCInfo(CallConv, ..., ArgLocs, ...)` to obtain outgoing argument information 
before entering the **“for loop”**.

The loop structure is almost identical to that in `LowerFormalArguments()`, 
except that `LowerCall()` creates a "store DAG vector" instead of a "load DAG 
vector".

After the **“for loop”**, it generates the instruction 
**`ld $t9, %call16(_Z5sum_iiiiiii)($gp)`** followed by `jalr $t9` to call the 
subroutine (where `$6` is `$t9`) in PIC (Position Independent Code) mode.

As with loading incoming arguments, we need to implement 
`storeRegToStackSlot()` in an earlier chapter to handle storing outgoing 
arguments.

Pseudo Hook Instructions ADJCALLSTACKDOWN and ADJCALLSTACKUP
************************************************************

`DAG.getCALLSEQ_START()` and `DAG.getCALLSEQ_END()` are invoked before and 
after the **“for loop”**, respectively. These insert `CALLSEQ_START` and 
`CALLSEQ_END`, which are later translated into the pseudo machine instructions 
`ADJCALLSTACKDOWN` and `ADJCALLSTACKUP`.

These pseudo instructions are defined in `Cpu0InstrInfo.td` as shown below:

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_2 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_2 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_2 //3
    :end-before: #endif

With the definition below, `eliminateCallFramePseudoInstr()` will be called 
when LLVM encounters the pseudo instructions `ADJCALLSTACKDOWN` and 
`ADJCALLSTACKUP`. 

This function simply discards these two pseudo instructions. LLVM will then 
automatically adjust the stack offset as needed.

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0InstrInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.cpp
    :start-after: //@Cpu0InstrInfo {
    :end-before: #if CH >= CH9_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.cpp
    :start-after: #if CH >= CH9_2
    :end-before: #endif
  
.. rubric:: lbdex/chapters/Chapter9_2/Cpu0FrameLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0FrameLowering.h
    :start-after: #if CH >= CH9_2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0FrameLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0FrameLowering.cpp
    :start-after: #if CH >= CH9_2
    :end-before: #if CH >= CH9_3 // dynamic alloc
.. literalinclude:: ../lbdex/Cpu0/Cpu0FrameLowering.cpp
    :start-after: #endif // dynamic alloc
    :end-before: #endif // #if CH >= CH9_2


Read LowerCall() with Graphviz's Help
**************************************

The complete DAGs created for outgoing arguments are shown in 
:numref:`funccall-f-outgoing-arg` for `ch9_outgoing.cpp` with `cpu032I`.

The `LowerCall()` function (excluding the call to `LowerCallResult()`) will 
generate the DAG nodes shown in :numref:`funccall-f-outgoing-arg-lowercal` for 
`ch9_outgoing.cpp` with `cpu032I`.

The corresponding code for the DAG nodes `Store` and `TargetGlobalAddress` is 
listed in the figures. Users can match other DAG nodes to the `LowerCall()` 
function code accordingly.

By using the Graphviz tool with the `llc` option `-view-dag-combine1-dags`, you 
can design a small input in C or LLVM IR, then inspect the DAGs to better 
understand the behavior of `LowerCall()` and `LowerFormalArguments()`.

In the later sub-sections, "Variable Arguments" and "Dynamic Stack Allocation 
Support", you can create input examples that demonstrate these features. You 
can then use the DAGs to confirm your understanding of the logic in these two 
functions.

For more information about Graphviz, refer to the section 
"Display LLVM IR Nodes with Graphviz" in Chapter 4, *Arithmetic and Logic 
Instructions*.

The DAG diagrams can be generated using the `llc` option as shown below:


.. rubric:: lbdex/input/ch9_outgoing.cpp
.. literalinclude:: ../lbdex/input/ch9_outgoing.cpp
    :start-after: /// start
    
.. code-block:: console
  
  JonathantekiiMac:input Jonathan$ clang -O3 -target mips-unknown-linux-gnu -c 
  ch9_outgoing.cpp -emit-llvm -o ch9_outgoing.bc
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llvm-dis ch9_outgoing.bc -o -
  ...
  define i32 @_Z10call_sum_iv() #0 {
    %1 = tail call i32 @_Z5sum_ii(i32 1)
    ret i32 %1
  }
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -mcpu=cpu032I -view-dag-combine1-dags -relocation-
  model=static -filetype=asm ch9_outgoing.bc -o -
  	.text
  	.section .mdebug.abiS32
  	.previous
  	.file	"ch9_outgoing.bc"
  Writing '/var/folders/rf/8bgdgt9d6vgf5sn8h8_zycd00000gn/T/dag._Z10call_sum_iv-
  0dfaf1.dot'...  done. 
  Running 'Graphviz' program...

.. _funccall-f-outgoing-arg:
.. graphviz:: ../Fig/funccall/outgoing-arg-S32.gv
   :caption: Outgoing arguments DAG (A) created for ch9_outgoing.cpp with -cpu0-s32-calls=true
  
.. _funccall-f-outgoing-arg-lowercal:
.. graphviz:: ../Fig/funccall/outgoing-arg-LowerCall.gv
   :caption: Outgoing arguments DAG (B) created by LowerCall() for ch9_outgoing.cpp with -cpu0-s32-calls=true

As mentioned in the previous section, the option ``llc -cpu0-s32-calls=true`` 
uses the S32 calling convention, which passes all arguments in registers. 

In contrast, the option ``llc -cpu0-s32-calls=false`` uses the O32 convention, 
which passes the first two arguments in registers and the remaining arguments 
on the stack.

The resulting behavior is shown as follows:

.. code-block:: console

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=true 
  -relocation-model=pic -filetype=asm ch9_1.bc -o -
	  .text
	  .section .mdebug.abiS32
	  .previous
	  .file	"ch9_1.bc"
	  .globl	_Z5sum_iiiiiii
	  .align	2
	  .type	_Z5sum_iiiiiii,@function
	  .ent	_Z5sum_iiiiiii          # @_Z5sum_iiiiiii
  _Z5sum_iiiiiii:
	  .frame	$fp,32,$lr
	  .mask 	0x00000000,0
	  .set	noreorder
	  .cpload	$t9
	  .set	nomacro
  # BB#0:
	  addiu	$sp, $sp, -32
	  ld	$2, 52($sp)
	  ld	$3, 48($sp)
	  ld	$4, 44($sp)
	  ld	$5, 40($sp)
	  ld	$t9, 36($sp)
	  ld	$7, 32($sp)
	  st	$7, 28($sp)
	  st	$t9, 24($sp)
	  st	$5, 20($sp)
	  st	$4, 16($sp)
	  st	$3, 12($sp)
	  lui	$3, %got_hi(gI)
	  addu	$3, $3, $gp
	  st	$2, 8($sp)
	  ld	$3, %got_lo(gI)($3)
	  ld	$3, 0($3)
	  ld	$4, 28($sp)
	  addu	$3, $3, $4
	  ld	$4, 24($sp)
	  addu	$3, $3, $4
	  ld	$4, 20($sp)
	  addu	$3, $3, $4
	  ld	$4, 16($sp)
	  addu	$3, $3, $4
	  ld	$4, 12($sp)
	  addu	$3, $3, $4
	  addu	$2, $3, $2
	  st	$2, 4($sp)
	  addiu	$sp, $sp, 32
	  ret	$lr
	  nop
	  .set	macro
	  .set	reorder
	  .end	_Z5sum_iiiiiii
  $tmp0:
	  .size	_Z5sum_iiiiiii, ($tmp0)-_Z5sum_iiiiiii

	  .globl	main
	  .align	2
	  .type	main,@function
	  .ent	main                    # @main
  main:
	  .frame	$fp,40,$lr
	  .mask 	0x00004000,-4
	  .set	noreorder
	  .cpload	$t9
	  .set	nomacro
  # BB#0:
	  addiu	$sp, $sp, -40
	  st	$lr, 36($sp)            # 4-byte Folded Spill
	  addiu	$2, $zero, 0
	  st	$2, 32($sp)
	  addiu	$2, $zero, 6
	  st	$2, 20($sp)
	  addiu	$2, $zero, 5
	  st	$2, 16($sp)
	  addiu	$2, $zero, 4
	  st	$2, 12($sp)
	  addiu	$2, $zero, 3
	  st	$2, 8($sp)
	  addiu	$2, $zero, 2
	  st	$2, 4($sp)
	  addiu	$2, $zero, 1
	  st	$2, 0($sp)
	  ld	$t9, %call16(_Z5sum_iiiiiii)($gp)
	  jalr	$t9
	  nop
	  st	$2, 28($sp)
	  ld	$lr, 36($sp)            # 4-byte Folded Reload
	  addiu	$sp, $sp, 40
	  ret	$lr
	  nop
	  .set	macro
	  .set	reorder
	  .end	main
  $tmp1:
	  .size	main, ($tmp1)-main

	  .type	gI,@object              # @gI
	  .data
	  .globl	gI
	  .align	2
  gI:
	  .4byte	100                     # 0x64
	  .size	gI, 4

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -mcpu=cpu032II -cpu0-s32-calls=false 
  -relocation-model=pic -filetype=asm ch9_1.bc -o -
    ...
	  .globl	main
	  .align	2
	  .type	main,@function
	  .ent	main                    # @main
  main:
	  .frame	$fp,40,$lr
	  .mask 	0x00004000,-4
	  .set	noreorder
	  .cpload	$t9
	  .set	nomacro
  # BB#0:
	  addiu	$sp, $sp, -40
	  st	$lr, 36($sp)            # 4-byte Folded Spill
	  addiu	$2, $zero, 0
	  st	$2, 32($sp)
	  addiu	$2, $zero, 6
	  st	$2, 20($sp)
	  addiu	$2, $zero, 5
	  st	$2, 16($sp)
	  addiu	$2, $zero, 4
	  st	$2, 12($sp)
	  addiu	$2, $zero, 3
	  st	$2, 8($sp)
	  ld	$t9, %call16(_Z5sum_iiiiiii)($gp)
	  addiu	$4, $zero, 1
	  addiu	$5, $zero, 2
	  jalr	$t9
	  nop
	  st	$2, 28($sp)
	  ld	$lr, 36($sp)            # 4-byte Folded Reload
	  addiu	$sp, $sp, 40
	  ret	$lr
	  nop
	  .set	macro
	  .set	reorder
	  .end	main


Long and Short String Initialization
************************************

In the previous section, we mentioned the `JSUB texternalsym` pattern.

Run `Chapter9_2` with `ch9_1_2.cpp` to observe the following results:

For a long string, LLVM generates a call to `memcpy()` to initialize the 
string—for example, `char str[81] = "Hello world"`.

For a short string, the `call memcpy` is optimized and translated into a 
direct `store` with a constant value during the optimization stages.

.. rubric:: lbdex/input/ch9_1_2.cpp
.. literalinclude:: ../lbdex/input/ch9_1_2.cpp
    :start-after: /// start

.. code-block:: console

  JonathantekiiMac:input Jonathan$ llvm-dis ch9_1_2.bc -o -
  ; ModuleID = 'ch9_1_2.bc'
  ...
  @_ZZ4mainE3str = private unnamed_addr constant [81 x i8] c"Hello world\00\00\00\
  00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00
  \00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0
  0\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1
  @_ZZ4mainE1s = private unnamed_addr constant [6 x i8] c"Hello\00", align 1
  
  ; Function Attrs: nounwind
  define i32 @main() #0 {
  entry:
    %retval = alloca i32, align 4
    %str = alloca [81 x i8], align 1
    store i32 0, i32* %retval
    %0 = bitcast [81 x i8]* %str to i8*
    call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* getelementptr inbounds 
    ([81 x i8]* @_ZZ4mainE3str, i32 0, i32 0), i32 81, i32 1, i1 false)
    %1 = bitcast [6 x i8]* %s to i8*
    call void @llvm.memcpy.p0i8.p0i8.i32(i8* %1, i8* getelementptr inbounds 
    ([6 x i8]* @_ZZ4mainE1s, i32 0, i32 0), i32 6, i32 1, i1 false)
  
    ret i32 0
  }
  
  JonathantekiiMac:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_1_2.cpp -emit-llvm -o ch9_1_2.bc
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build
  /bin/llc -march=cpu0 -mcpu=cpu032II -cpu0-s32-calls=true
  -relocation-model=static -filetype=asm ch9_1_2.bc -o -
    .section .mdebug.abi32
    ...
	  lui	$2, %hi($_ZZ4mainE3str)
	  ori	$2, $2, %lo($_ZZ4mainE3str)
	  st	$2, 4($sp)
	  addiu	$2, $sp, 24
	  st	$2, 0($sp)
	  jsub	memcpy
	  nop
	  lui	$2, %hi($_ZZ4mainE1s)
	  ori	$2, $2, %lo($_ZZ4mainE1s)
	  lbu	$3, 4($2)
	  shl	$3, $3, 8
	  lbu	$4, 5($2)
	  or	$3, $3, $4
	  sh	$3, 20($sp)
	  lbu	$3, 2($2)
	  shl	$3, $3, 8
	  lbu	$4, 3($2)
	  or	$3, $3, $4
	  lbu	$4, 1($2)
	  lbu	$2, 0($2)
	  shl	$2, $2, 8
	  or	$2, $2, $4
	  shl	$2, $2, 16
	  or	$2, $2, $3
	  st	$2, 16($sp)
    ...
	.type	$_ZZ4mainE3str,@object  # @_ZZ4mainE3str
	.section	.rodata,"a",@progbits
  $_ZZ4mainE3str:
	  .asciz	"Hello world\000\000\000\000\000\000\000\000\000\000\000\000\000\000
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000
    \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
	  .size	$_ZZ4mainE3str, 81

	  .type	$_ZZ4mainE1s,@object    # @_ZZ4mainE1s
	  .section	.rodata.str1.1,"aMS",@progbits,1
  $_ZZ4mainE1s:
	  .asciz	"Hello"
	  .size	$_ZZ4mainE1s, 6


The `call memcpy` for a short string is optimized by LLVM before the 
"DAG-to-DAG Pattern Instruction Selection" stage. 

It is translated into a `store` with a constant value, as shown below:

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build
  /bin/llc -march=cpu0 -mcpu=cpu032II -cpu0-s32-calls=true 
  -relocation-model=static -filetype=asm ch9_1_2.bc -debug -o -
  
  Initial selection DAG: BB#0 'main:entry'
  SelectionDAG has 35 nodes:
    ...
          0x7fd909030810: <multiple use>
          0x7fd909030c10: i32 = Constant<1214606444>  // 1214606444=0x48656c6c="Hell"
  
          0x7fd909030910: <multiple use>
          0x7fd90902d810: <multiple use>
        0x7fd909030d10: ch = store 0x7fd909030810, 0x7fd909030c10, 0x7fd909030910, 
        0x7fd90902d810<ST4[%1]>
  
          0x7fd909030810: <multiple use>
          0x7fd909030e10: i16 = Constant<28416>      // 28416=0x6f00="o\0"
  
          ...
  
          0x7fd90902d810: <multiple use>
        0x7fd909031210: ch = store 0x7fd909030810, 0x7fd909030e10, 0x7fd909031010, 
        0x7fd90902d810<ST2[%1+4](align=4)>
    ...

The incoming arguments refer to the *formal arguments* as defined in compiler 
and programming language literature. The outgoing arguments refer to the 
*actual arguments* passed during a function call.

Summary as Table: Callee incoming arguments and caller outgoing arguments.

.. table:: Callee incoming arguments and caller outgoing arguments

  ========================  ===========================================    ===============================
  Description               Callee                                         Caller   
  ========================  ===========================================    ===============================
  Charged Function          LowerFormalArguments()                         LowerCall()
  Charged Function Created  Create load vectors for incoming arguments     Create store vectors for outgoing arguments
  ========================  ===========================================    ===============================


Structure Type Support
-----------------------

Ordinary Struct Type
********************

The following code in `Chapter9_1/` and `Chapter3_4/` supports ordinary 
structure types in function calls.

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@LowerFormalArguments {
    :end-before: #if CH >= CH3_4
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Ordinary struct type: 1 {
    :end-before: //@Ordinary struct type: 1 }

.. code-block:: c++

  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH3_1 //LowerReturn
    :end-before: #if CH >= CH3_4 //in LowerReturn
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Ordinary struct type: 2 {
    :end-before: //@Ordinary struct type: 2 }

.. code-block:: c++

  }


In addition to the code above, we defined the calling convention in an earlier 
chapter as follows:

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0CallingConv.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0CallingConv.td
    :start-after: //#if CH >= CH3_4 1
    :end-before: //#endif

This means that for the return value, we store it in registers `V0`, `V1`, `A0`, 
and `A1` if the size of the return value does not exceed four registers. 

If it exceeds four registers, Cpu0 will store the value in memory and return a 
pointer to that memory in a register.

For demonstration, let's run `Chapter9_2/` with `ch9_1_struct.cpp` and explain 
using this example.

.. rubric:: lbdex/input/ch9_1_struct.cpp
.. literalinclude:: ../lbdex/input/ch9_1_struct.cpp
    :start-after: /// start

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm 
  ch9_1_struct.bc -o -
    .section .mdebug.abi32
    .previous
    .file "ch9_1_struct.bc"
    .text
    .globl  _Z7getDatev
    .align  2
    .type _Z7getDatev,@function
    .ent  _Z7getDatev             # @_Z7getDatev
  _Z7getDatev:
    .cfi_startproc
    .frame  $sp,0,$lr
    .mask   0x00000000,0
    .set  noreorder
    .cpload $t9
    .set  nomacro
  # BB#0:
	  lui	$2, %got_hi(gDate)
	  addu	$2, $2, $gp
	  ld	$3, %got_lo(gDate)($2)
	  ld	$2, 0($sp)
    ld  $4, 20($3)        // save gDate contents to 212..192($sp)
    st  $4, 20($2)
    ld  $4, 16($3)
    st  $4, 16($2)
    ld  $4, 12($3)
    st  $4, 12($2)
    ld  $4, 8($3)
    st  $4, 8($2)
    ld  $4, 4($3)
    st  $4, 4($2)
    ld  $3, 0($3)
    st  $3, 0($2)
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z7getDatev
  $tmp0:
    .size _Z7getDatev, ($tmp0)-_Z7getDatev
    .cfi_endproc
    ...
    .globl  _Z20test_func_arg_structv
    .align  2
    .type _Z20test_func_arg_structv,@function
    .ent  _Z20test_func_arg_structv                    # @main
  _Z20test_func_arg_structv:
    .cfi_startproc
    .frame  $sp,248,$lr
    .mask   0x00004180,-4
    .set  noreorder
    .cpload $t9
    .set  nomacro
    # BB#0:
	  addiu	$sp, $sp, -200
	  st	$lr, 196($sp)           # 4-byte Folded Spill
	  st	$8, 192($sp)            # 4-byte Folded Spill
	  ld	$2, %got($_ZZ20test_func_arg_structvE5time1)($gp)
	  ori	$2, $2, %lo($_ZZ20test_func_arg_structvE5time1)
	  ld	$3, 8($2)
	  st	$3, 184($sp)
	  ld	$3, 4($2)
	  st	$3, 180($sp)
	  ld	$2, 0($2)
	  st	$2, 176($sp)
	  addiu	$8, $sp, 152
	  st	$8, 0($sp)
	  ld	$t9, %call16(_Z7getDatev)($gp) // copy gDate contents to date1, 176..152($sp)
	  jalr	$t9
	  nop
	  ld	$gp, 176($sp)
	  ld	$2, 172($sp)
	  st	$2, 124($sp)
	  ld	$2, 168($sp)
	  st	$2, 120($sp)
	  ld	$2, 164($sp)
	  st	$2, 116($sp)
	  ld	$2, 160($sp)
	  st	$2, 112($sp)
	  ld	$2, 156($sp)
	  st	$2, 108($sp)
	  ld	$2, 152($sp)
	  st	$2, 104($sp)
    ...


The `ch9_1_constructor.cpp` includes an implementation of the C++ class `Date`.

This can also be translated by the Cpu0 backend, since the frontend (Clang, in 
this case) translates C++ classes into equivalent C language constructs.

If you comment out the `if hasStructRetAttr()` part in both of the functions 
mentioned above, the output Cpu0 code for `ch9_1_struct.cpp` will use register 
`$3` instead of `$2` as the return register, as shown below:

.. code-block:: console

	  .text
	  .section .mdebug.abiS32
	  .previous
	  .file	"ch9_1_struct.bc"
	  .globl	_Z7getDatev
	  .align	2
	  .type	_Z7getDatev,@function
	  .ent	_Z7getDatev             # @_Z7getDatev
  _Z7getDatev:
	  .frame	$fp,0,$lr
	  .mask 	0x00000000,0
	  .set	noreorder
	  .cpload	$t9
	  .set	nomacro
  # BB#0:
	  lui	$2, %got_hi(gDate)
	  addu	$2, $2, $gp
	  ld	$2, %got_lo(gDate)($2)
	  ld	$3, 0($sp)
	  ld	$4, 20($2)
	  st	$4, 20($3)
	  ld	$4, 16($2)
	  st	$4, 16($3)
	  ld	$4, 12($2)
	  st	$4, 12($3)
	  ld	$4, 8($2)
	  st	$4, 8($3)
	  ld	$4, 4($2)
	  st	$4, 4($3)
	  ld	$2, 0($2)
	  st	$2, 0($3)
	  ret	$lr
	  nop
    ...

According to the MIPS ABI, the address for returning a struct variable must be 
placed in register `$2`.

Byval Struct Type
*****************

The following code in `Chapter9_1/` and `Chapter9_2/` supports the `byval` 
structure type in function calls.

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_1 //11
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@LowerFormalArguments {
    :end-before: #if CH >= CH3_4
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@2 {
    :end-before: //@2 }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@byval pass {
    :end-before: //@byval pass }

.. code-block:: c++

      ...
  . }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Ordinary struct type: 1 {
    :end-before: //@Ordinary struct type: 1 }

.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_2 //7
    :end-before: #endif // #if CH >= CH9_2 //7
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@LowerCall {
    :end-before: #if CH >= CH9_2 //2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@1 {
    :end-before: //@1 }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@ByVal Arg {
    :end-before: //@ByVal Arg }

.. code-block:: c++

      ...
    }
    ...
  }

In `LowerCall()`, `Flags.isByVal()` will be `true` if the function call in the 
caller contains a **byval** struct type, as shown below:

.. rubric:: lbdex/input/tailcall.ll
.. code-block:: llvm

  define internal fastcc i32 @caller9_1() nounwind noinline {
  entry:
    ...
    %call = tail call i32 @callee9(%struct.S* byval @gs1) nounwind
    ret i32 %call
  }

In `LowerFormalArguments()`, `Flags.isByVal()` will be `true` when it encounters 
a **byval** parameter in the callee function, as shown below:

.. rubric:: lbdex/input/tailcall.ll
.. code-block:: llvm

  define i32 @caller12(%struct.S* nocapture byval %a0) nounwind {
  entry:
    ...
  }

At this point, I don't know how to make Clang generate `byval` IR using the 
C language.

Function Call Optimization
--------------------------

Tail Call Optimization
**********************

Tail call optimization is applied in certain function call situations. In some 
cases, the caller and callee can share the same memory stack.

When applied to recursive function calls, this optimization often reduces the 
stack space requirement from linear, or O(n), to constant, or O(1) 
[#wikitailcall]_.

LLVM IR supports `tailcall` as described here [#tailcallopt]_.

The `tailcall` instructions appearing in `Cpu0ISelLowering.cpp` and 
`Cpu0InstrInfo.td` are used to implement tail call optimization.

.. rubric:: lbdex/input/ch9_2_tailcall.cpp
.. literalinclude:: ../lbdex/input/ch9_2_tailcall.cpp
    :start-after: /// start

Run `Chapter9_2/` with `ch9_2_tailcall.cpp` to get the following result.

.. code-block:: console

  JonathantekiiMac:input Jonathan$ clang -O1 -target mips-unknown-linux-gnu -c 
  ch9_2_tailcall.cpp -emit-llvm -o ch9_2_tailcall.bc
  JonathantekiiMac:input Jonathan$ ~/llvm/test/build/bin/
  llvm-dis ch9_2_tailcall.bc -o -
  ...
  ; Function Attrs: nounwind readnone
  define i32 @_Z9factoriali(i32 %x) #0 {
    %1 = icmp sgt i32 %x, 0
    br i1 %1, label %tailrecurse, label %tailrecurse._crit_edge

  tailrecurse:                                      ; preds = %tailrecurse, %0
    %x.tr2 = phi i32 [ %2, %tailrecurse ], [ %x, %0 ]
    %accumulator.tr1 = phi i32 [ %3, %tailrecurse ], [ 1, %0 ]
    %2 = add nsw i32 %x.tr2, -1
    %3 = mul nsw i32 %x.tr2, %accumulator.tr1
    %4 = icmp sgt i32 %2, 0
    br i1 %4, label %tailrecurse, label %tailrecurse._crit_edge

  tailrecurse._crit_edge:                           ; preds = %tailrecurse, %0
    %accumulator.tr.lcssa = phi i32 [ 1, %0 ], [ %3, %tailrecurse ]
    ret i32 %accumulator.tr.lcssa
  }

  ; Function Attrs: nounwind readnone
  define i32 @_Z13test_tailcalli(i32 %a) #0 {
    %1 = tail call i32 @_Z9factoriali(i32 %a)
    ret i32 %1
  }
  ...
  JonathantekiiMac:input Jonathan$ ~/llvm/test/build/bin/
  llc -march=cpu0 -mcpu=cpu032II -relocation-model=static -filetype=asm 
  -enable-cpu0-tail-calls ch9_2_tailcall.bc -stats -o -
	  .text
	  .section .mdebug.abi32
	  .previous
	  .file	"ch9_2_tailcall.bc"
	  .globl	_Z9factoriali
	  .align	2
	  .type	_Z9factoriali,@function
	  .ent	_Z9factoriali           # @_Z9factoriali
  _Z9factoriali:
	  .frame	$sp,0,$lr
	  .mask 	0x00000000,0
	  .set	noreorder
	  .set	nomacro
  # BB#0:
	  addiu	$2, $zero, 1
	  slt	$3, $4, $2
	  bne	$3, $zero, $BB0_2
	  nop
  $BB0_1:                                 # %tailrecurse
                                          # =>This Inner Loop Header: Depth=1
	  mul	$2, $4, $2
	  addiu	$4, $4, -1
	  addiu	$3, $zero, 0
	  slt	$3, $3, $4
	  bne	$3, $zero, $BB0_1
	  nop
  $BB0_2:                                 # %tailrecurse._crit_edge
	  ret	$lr
	  nop
	  .set	macro
	  .set	reorder
	  .end	_Z9factoriali
  $tmp0:
	  .size	_Z9factoriali, ($tmp0)-_Z9factoriali

	  .globl	_Z13test_tailcalli
	  .align	2
	  .type	_Z13test_tailcalli,@function
	  .ent	_Z13test_tailcalli      # @_Z13test_tailcalli
  _Z13test_tailcalli:
	  .frame	$sp,0,$lr
	  .mask 	0x00000000,0
	  .set	noreorder
	  .set	nomacro
  # BB#0:
	  jmp	_Z9factoriali
	  nop
	  .set	macro
	  .set	reorder
	  .end	_Z13test_tailcalli
  $tmp1:
	  .size	_Z13test_tailcalli, ($tmp1)-_Z13test_tailcalli


  ===-------------------------------------------------------------------------===
                            ... Statistics Collected ...
  ===-------------------------------------------------------------------------===

   ...
   1 cpu0-lower        - Number of tail calls
   ...

The tail call optimization shares the caller's and callee's stack, and it is 
applied in `cpu032II` only for this example (it uses `jmp _Z9factoriali` 
instead of `jsub _Z9factoriali`).

However, `cpu032I` (which passes all arguments on the stack) does not satisfy 
the condition `NextStackOffset <= FI.getIncomingArgSize()` in 
`isEligibleForTailCallOptimization()`, and thus returns `false` for the 
function, as shown below:

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0SEISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelLowering.cpp
    :start-after: #if CH >= CH9_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@LowerCall {
    :end-before: #if CH >= CH9_2 //2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@TailCall 1 {
    :end-before: //@TailCall 1 }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@TailCall 2 {
    :end-before: //@TailCall 2 }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@TailCall 3 {
    :end-before: //@TailCall 3 }

.. code-block:: c++

    ...
  }

Since tail call optimization translates the call into a `jmp` instruction 
directly instead of `jsub`, the `callseq_start`, `callseq_end`, and the DAG 
nodes created in `LowerCallResult()` and `LowerReturn()` are unnecessary. It 
creates DAGs for `ch9_2_tailcall.cpp` as shown in 
:numref:`funccall-f-outgoing-arg-tailcall`.

.. _funccall-f-outgoing-arg-tailcall:
.. graphviz:: ../Fig/funccall/outgoing-arg-tailcall.gv
   :caption: Outgoing arguments DAGs created for ch9_2_tailcall.cpp 

Finally, the DAGs translation of the tail call is listed in the following table.

.. table:: the DAGs translation of tail call

  =============================  =================  =============
  Stage                          DAG                Function
  =============================  =================  =============
  Backend lowering               Cpu0ISD::TailCall  LowerCall()
  Instruction selection          TAILCALL           note 1
  Instruction Print              JMP                note 2
  =============================  =================  =============

note 1: by Cpu0InstrInfo.td as follows,

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@Pat<Cpu0TailCall>
    :end-before: }


note 2: by Cpu0InstrInfo.td and emitPseudoExpansionLowering() of 
Cpu0AsmPrinter.cpp as follows,

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@class TailCall
    :end-before: } // let Predicates = [Ch9_1]
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH9_1 //9
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0AsmPrinter.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.h
    :start-after: #if CH >= CH9_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0AsmPrinter.cpp
.. literalinclude:: ../lbdex/chapters/Chapter9_1/Cpu0AsmPrinter.cpp
    :start-after: //@EmitInstruction {
    :end-before: //@EmitInstruction }

The function `emitPseudoExpansionLowering()` is generated by TableGen and is 
located in `Cpu0GenMCPseudoLowering.inc`.


Recursion optimization
**********************

As mentioned in the last section, cpu032I cannot perform tail call optimization 
in `ch9_2_tailcall.cpp` due to the limitation that the argument size condition 
is not satisfied. 

However, when running with the ``clang -O3`` optimization option, it can achieve 
the same or even better performance than tail call optimization, as shown below:

.. code-block:: console

  JonathantekiiMac:input Jonathan$ clang -O1 -target mips-unknown-linux-gnu -c 
  ch9_2_tailcall.cpp -emit-llvm -o ch9_2_tailcall.bc
  JonathantekiiMac:input Jonathan$ ~/llvm/test/build/bin/
  llvm-dis ch9_2_tailcall.bc -o -
  ...
  ; Function Attrs: nounwind readnone
  define i32 @_Z9factoriali(i32 %x) #0 {
    %1 = icmp sgt i32 %x, 0
    br i1 %1, label %tailrecurse.preheader, label %tailrecurse._crit_edge

  tailrecurse.preheader:                            ; preds = %0
    br label %tailrecurse

  tailrecurse:                                      ; preds = %tailrecurse, 
  %tailrecurse.preheader
    %x.tr2 = phi i32 [ %2, %tailrecurse ], [ %x, %tailrecurse.preheader ]
    %accumulator.tr1 = phi i32 [ %3, %tailrecurse ], [ 1, %tailrecurse.preheader ]
    %2 = add nsw i32 %x.tr2, -1
    %3 = mul nsw i32 %x.tr2, %accumulator.tr1
    %4 = icmp sgt i32 %2, 0
    br i1 %4, label %tailrecurse, label %tailrecurse._crit_edge.loopexit

  tailrecurse._crit_edge.loopexit:                  ; preds = %tailrecurse
    %.lcssa = phi i32 [ %3, %tailrecurse ]
    br label %tailrecurse._crit_edge

  tailrecurse._crit_edge:                           ; preds = %tailrecurse._crit
    _edge.loopexit, %0
    %accumulator.tr.lcssa = phi i32 [ 1, %0 ], [ %.lcssa, %tailrecurse._crit_edge
    .loopexit ]
    ret i32 %accumulator.tr.lcssa
  }

  ; Function Attrs: nounwind readnone
  define i32 @_Z13test_tailcalli(i32 %a) #0 {
    %1 = icmp sgt i32 %a, 0
    br i1 %1, label %tailrecurse.i.preheader, label %_Z9factoriali.exit

  tailrecurse.i.preheader:                          ; preds = %0
    br label %tailrecurse.i

  tailrecurse.i:                                    ; preds = %tailrecurse.i, 
    %tailrecurse.i.preheader
    %x.tr2.i = phi i32 [ %2, %tailrecurse.i ], [ %a, %tailrecurse.i.preheader ]
    %accumulator.tr1.i = phi i32 [ %3, %tailrecurse.i ], [ 1, %tailrecurse.i.
    preheader ]
    %2 = add nsw i32 %x.tr2.i, -1
    %3 = mul nsw i32 %accumulator.tr1.i, %x.tr2.i
    %4 = icmp sgt i32 %2, 0
    br i1 %4, label %tailrecurse.i, label %_Z9factoriali.exit.loopexit

  _Z9factoriali.exit.loopexit:                      ; preds = %tailrecurse.i
    %.lcssa = phi i32 [ %3, %tailrecurse.i ]
    br label %_Z9factoriali.exit

  _Z9factoriali.exit:                               ; preds = %_Z9factoriali.
    exit.loopexit, %0
    %accumulator.tr.lcssa.i = phi i32 [ 1, %0 ], [ %.lcssa, %_Z9factoriali.
    exit.loopexit ]
    ret i32 %accumulator.tr.lcssa.i
  }
  ...
  JonathantekiiMac:input Jonathan$ ~/llvm/test/build/bin/
  llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm 
  ch9_2_tailcall.bc -o -
	  .text
	  .section .mdebug.abiS32
	  .previous
	  .file	"ch9_2_tailcall.bc"
	  .globl	_Z9factoriali
	  .align	2
	  .type	_Z9factoriali,@function
	  .ent	_Z9factoriali           # @_Z9factoriali
  _Z9factoriali:
	  .frame	$sp,0,$lr
	  .mask 	0x00000000,0
	  .set	noreorder
	  .set	nomacro
  # BB#0:
	  addiu	$2, $zero, 1
	  ld	$3, 0($sp)
	  cmp	$sw, $3, $2
	  jlt	$sw, $BB0_2
	  nop
  $BB0_1:                                 # %tailrecurse
                                          # =>This Inner Loop Header: Depth=1
	  mul	$2, $3, $2
	  addiu	$3, $3, -1
	  addiu	$4, $zero, 0
	  cmp	$sw, $3, $4
	  jgt	$sw, $BB0_1
	  nop
  $BB0_2:                                 # %tailrecurse._crit_edge
	  ret	$lr
	  nop
	  .set	macro
	  .set	reorder
	  .end	_Z9factoriali
  $tmp0:
	  .size	_Z9factoriali, ($tmp0)-_Z9factoriali

	  .globl	_Z13test_tailcalli
	  .align	2
	  .type	_Z13test_tailcalli,@function
	  .ent	_Z13test_tailcalli      # @_Z13test_tailcalli
  _Z13test_tailcalli:
	  .frame	$sp,0,$lr
	  .mask 	0x00000000,0
	  .set	noreorder
	  .set	nomacro
  # BB#0:
	  addiu	$2, $zero, 1
	  ld	$3, 0($sp)
	  cmp	$sw, $3, $2
	  jlt	$sw, $BB1_2
	  nop
  $BB1_1:                                 # %tailrecurse.i
                                          # =>This Inner Loop Header: Depth=1
	  mul	$2, $2, $3
	  addiu	$3, $3, -1
	  addiu	$4, $zero, 0
	  cmp	$sw, $3, $4
	  jgt	$sw, $BB1_1
	  nop
  $BB1_2:                                 # %_Z9factoriali.exit
	  ret	$lr
	  nop
	  .set	macro
	  .set	reorder
	  .end	_Z13test_tailcalli
  $tmp1:
	  .size	_Z13test_tailcalli, ($tmp1)-_Z13test_tailcalli

According to the above LLVM IR, the ``clang -O3`` option replaces recursion 
with a loop by inlining the callee recursion function. This is a frontend 
optimization achieved through cross-function analysis.

Cpu0 doesn't support `fastcc` [#callconv]_, but it can pass the `fastcc` 
keyword in the IR. MIPS supports `fastcc` by using as many registers as 
possible without strictly following the ABI specification.

Other Features Supported
-------------------------

This section supports features for the "$gp register caller saved register in 
PIC addressing mode," "variable number of arguments," and "dynamic stack 
allocation."

Run `Chapter9_2/` with `ch9_3_vararg.cpp` to get the following error:

.. rubric:: lbdex/input/ch9_3_vararg.cpp
.. literalinclude:: ../lbdex/input/ch9_3_vararg.cpp
    :start-after: /// start

.. code-block:: console

  118-165-78-230:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_3_vararg.cpp -emit-llvm -o ch9_3_vararg.bc
  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch9_3_vararg.bc -o -
  ...
  LLVM ERROR: Cannot select: 0x7f8b6902fd10: ch = vastart 0x7f8b6902fa10, 
  0x7f8b6902fb10, 0x7f8b6902fc10 [ORD=9] [ID=22]
    0x7f8b6902fb10: i32 = FrameIndex<5> [ORD=7] [ID=9]
  In function: _Z5sum_iiz


.. rubric:: lbdex/input/ch9_3_alloc.cpp
.. literalinclude:: ../lbdex/input/ch9_3_alloc.cpp
    :start-after: /// start

Run `Chapter9_2` with `ch9_3_alloc.cpp` to get the following error.

.. code-block:: console

  118-165-72-242:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_3_alloc.cpp -emit-llvm -o ch9_3_alloc.bc
  118-165-72-242:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=false 
  -relocation-model=pic -filetype=asm ch9_3_alloc.bc -o -
  ...
  LLVM ERROR: Cannot select: 0x7ffd8b02ff10: i32,ch = dynamic_stackalloc 
  0x7ffd8b02f910:1, 0x7ffd8b02fe10, 0x7ffd8b02c010 [ORD=12] [ID=48]
    0x7ffd8b02fe10: i32 = and 0x7ffd8b02fc10, 0x7ffd8b02fd10 [ORD=12] [ID=47]
      0x7ffd8b02fc10: i32 = add 0x7ffd8b02fa10, 0x7ffd8b02fb10 [ORD=12] [ID=46]
        0x7ffd8b02fa10: i32 = shl 0x7ffd8b02f910, 0x7ffd8b02f510 [ID=45]
          0x7ffd8b02f910: i32,ch = load 0x7ffd8b02ee10, 0x7ffd8b02e310, 
          0x7ffd8b02b310<LD4[%1]> [ID=44]
            0x7ffd8b02e310: i32 = FrameIndex<1> [ORD=3] [ID=10]
            0x7ffd8b02b310: i32 = undef [ORD=1] [ID=2]
          0x7ffd8b02f510: i32 = Constant<2> [ID=25]
        0x7ffd8b02fb10: i32 = Constant<7> [ORD=12] [ID=16]
      0x7ffd8b02fd10: i32 = Constant<-8> [ORD=12] [ID=17]
    0x7ffd8b02c010: i32 = Constant<0> [ORD=12] [ID=8]
  In function: _Z5sum_iiiiiii


.. _global-var-pic-func:

Gloabal Variables Accessing In PIC Addressing Mode
**************************************************

In order to support Global Variables accessing in PIC mode, 
The $gp Register Caller Saved Register in PIC Addressing Mode have to be solved.

According to the original Cpu0 website information, it only supports **“jsub”** 
for 24-bit address range access. We added **“jalr”** to Cpu0 and expanded it to 
32-bit addressing. We made this change for two reasons: 

1. Cpu0 can be expanded to 32-bit address space by simply adding this instruction.
2. Cpu0 and this book are designed as a tutorial for better understanding.

We reserve **“jalr”** for PIC mode, which is used for dynamic linking functions, 
to demonstrate:

1. How the caller handles the caller-saved register `$gp` when calling a function.
2. How code in the shared library function uses `$gp` to access the global 
   variable address.
3. Why using **jalr** for dynamic linking functions is easier to implement and 
   faster. As we discussed in the 
   :ref:`section PIC Mode in Chapter Global Variables <pic-mode-global>` this 
   solution is popular in real 
   applications and deserves to be incorporated into the official Cpu0 design in 
   compiler books.

In the chapter on "Global Variables," we mentioned two link types: static link 
and dynamic link. The option `-relocation-model=static` is for static link functions, 
while `-relocation-model=pic` is for dynamic link functions. An example of a dynamic 
link function is calling functions from a shared library.

Shared libraries consist of many dynamic link functions that are typically loaded 
at runtime. Since shared libraries can be loaded at different memory addresses, 
the address of a global variable cannot be determined at link time. However, the 
distance between the global variable address and the start address of the shared 
library function can be calculated once it has been loaded.

Let's run `Chapter9_3/` with `ch9_gprestore.cpp` to get the following result. 
We will add comments in the result for explanation.

.. rubric:: lbdex/input/ch9_gprestore.cpp
.. literalinclude:: ../lbdex/input/ch9_gprestore.cpp
    :start-after: /// start

.. code-block:: console

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -mcpu=cpu032II-cpu0-s32-calls=true
  -relocation-model=pic -filetype=asm ch9_gprestore.bc -o -
  ...
    .cpload $t9
    .set  nomacro
  # BB#0:                                 # %entry
    addiu $sp, $sp, -24
  $tmp0:
    .cfi_def_cfa_offset 24
    st  $lr, 12($sp)            # 4-byte Folded Spill
    st  $fp, 16($sp)              # 4-byte Folded Spill
  $tmp1:
    .cfi_offset 14, -4
  $tmp2:
    .cfi_offset 12, -8
    .cprestore  8    // save $gp to 8($sp)
    ld  $t9, %call16(_Z5sum_ii)($gp)
    addiu $4, $zero, 1
    jalr  $t9
    nop
    ld  $gp, 8($sp)  // restore $gp from 8($sp)
    add $8, $zero, $2
    ld  $t9, %call16(_Z5sum_ii)($gp)
    addiu $4, $zero, 2
    jalr  $t9
    nop
    ld  $gp, 8($sp)  // restore $gp from 8($sp)
    addu  $2, $2, $8
    ld  $8, 8($sp)              # 4-byte Folded Reload
    ld  $lr, 12($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 16
    ret $lr
    nop

As mentioned in the code comment, **“.cprestore 8”** is a pseudo instruction 
for saving **$gp** to **8($sp)**, while the instruction **“ld $gp, 8($sp)”** 
restores the $gp. Refer to Table 8-1 of the "MIPSpro TM Assembly Language 
Programmer’s Guide" [#mipsasm]_ for more details.

In other words, $gp is a caller-saved register, so the `main()` function 
needs to save and restore $gp before and after calling the shared library 
`_Z5sum_ii()` function.

In LLVM MIPS 3.5, the `.cprestore` instruction was removed in PIC mode, 
meaning $gp is no longer treated as a caller-saved register in PIC. However, 
it is still present in Cpu0, and this feature can be removed by not defining 
it in `Cpu0Config.h`.

The `#ifdef ENABLE_GPRESTORE` part of the code in Cpu0 can be removed, but 
it comes with the cost of reserving the $gp register as a specific register 
that cannot be allocated for program variables in PIC mode. As explained in 
earlier chapters on "Global Variables," PIC is not a critical function, and 
its performance advantage can be considered negligible in dynamic linking. 
Therefore, we keep this feature in Cpu0.

Reserving $gp as a specific register in PIC mode will save a lot of code 
during programming. When reserving $gp, the `.cprestore` can be disabled 
using the option `"-cpu0-reserve-gp"`.

The `.cpload` instruction is still needed even when reserving $gp (since 
programmers may implement boot code functions with a mix of C and assembly). 
In this case, the programmer can set the $gp value through `.cpload`.

If enabling `-cpu0-no-cpload`, and undefining `ENABLE_GPRESTORE` or enabling 
`-cpu0-reserve-gp`, the `.cpload` and `$gp` save/restore instructions will 
not be issued, as shown in the following.

.. code-block:: console

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -mcpu=cpu032II-cpu0-s32-calls=true 
  -relocation-model=pic -filetype=asm ch9_gprestore.bc -cpu0-no-cpload
  -cpu0-reserve-gp -o -
  ...
  # BB#0:
    addiu $sp, $sp, -24
  $tmp0:
    .cfi_def_cfa_offset 24
    st  $lr, 20($sp)            # 4-byte Folded Spill
    st  $fp, 16($sp)            # 4-byte Folded Spill
  $tmp1:
    .cfi_offset 14, -4
  $tmp2:
    .cfi_offset 12, -8
    move   $fp, $sp
  $tmp3:
    .cfi_def_cfa_register 12
    ld  $t9, %call16(_Z5sum_ii)($gp)
    addiu $4, $zero, 1
    jalr  $t9
    nop
    st  $2, 12($fp)
    addiu $4, $zero, 2
    ld  $t9, %call16(_Z5sum_ii)($gp)
    jalr  $t9
    nop
    ld  $3, 12($fp)
    addu  $2, $3, $2
    st  $2, 12($fp)
    move   $sp, $fp
    ld  $fp, 16($sp)            # 4-byte Folded Reload
    ld  $lr, 20($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 24
    ret $lr
    nop


LLVM Mips 3.1 emits the directives ``.cpload`` and ``.cprestore``, and Cpu0 
inherits this behavior from that version. However, newer versions of LLVM Mips 
replace ``.cpload`` with actual instructions and remove ``.cprestore`` entirely. 
In these versions, the ``$gp`` register is treated as a reserved register in PIC 
(position-independent code) mode.

According to the MIPS assembly documentation I referenced, ``$gp`` is considered 
a "caller-saved register." Cpu0 follows this convention and provides an option to 
reserve the ``$gp`` register accordingly.

.. code-block:: console

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=mips -relocation-model=pic -filetype=asm ch9_gprestore.bc 
  -o -
  ...
  # BB#0:                                 # %entry
    lui $2, %hi(_gp_disp)
    ori $2, $2, %lo(_gp_disp)
    addiu $sp, $sp, -32
  $tmp0:
    .cfi_def_cfa_offset 32
    sw  $ra, 28($sp)            # 4-byte Folded Spill
    sw  $fp, 24($sp)            # 4-byte Folded Spill
    sw  $16, 20($sp)            # 4-byte Folded Spill
  $tmp1:
    .cfi_offset 31, -4
  $tmp2:
    .cfi_offset 30, -8
  $tmp3:
    .cfi_offset 16, -12
    move   $fp, $sp
  $tmp4:
    .cfi_def_cfa_register 30
    addu  $16, $2, $25
    lw  $25, %call16(_Z5sum_ii)($16)
    addiu $4, $zero, 1
    jalr  $25
    move   $gp, $16
    sw  $2, 16($fp)
    lw  $25, %call16(_Z5sum_ii)($16)
    jalr  $25
    addiu $4, $zero, 2
    lw  $1, 16($fp)
    addu  $2, $1, $2
    sw  $2, 16($fp)
    move   $sp, $fp
    lw  $16, 20($sp)            # 4-byte Folded Reload
    lw  $fp, 24($sp)            # 4-byte Folded Reload
    lw  $ra, 28($sp)            # 4-byte Folded Reload
    jr  $ra
    addiu $sp, $sp, 32

The following code, added in Chapter9_3/, emits ``.cprestore`` or the 
corresponding machine instructions before the first PIC function call.

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@LowerCall {
    :end-before: #if CH >= CH9_2 //2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //6
    :end-before: #endif //#if CH >= CH9_3 //6

.. code-block:: c++

  ...
  }

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0MachineFunction.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH9_3
    :end-before: #endif //#if CH >= CH9_3

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0SEFrameLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: //@emitPrologue {
    :end-before: #if CH >= CH3_5 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: //@ENABLE_GPRESTORE {
    :end-before: //@ENABLE_GPRESTORE }
	
.. code-block:: c++

  }

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0RegisterInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: //@eliminateFrameIndex {
    :end-before: #if CH >= CH3_5
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: #if CH >= CH9_3 //1
    :end-before: #endif //#if CH >= CH9_3 //1
	
.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@def CPRESTORE {
    :end-before: } // let Predicates = [Ch9_2]
  
.. rubric:: lbdex/chapters/Chapter9_3/Cpu0AsmPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH9_3 //1
    :end-before: #endif //#if CH >= CH9_3 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH9_3 //2
    :end-before: #endif //#if CH >= CH9_3 //2
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: //@EmitInstruction {
    :end-before: //@EmitInstruction body {
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH9_3 //3
    :end-before: #endif //#if CH >= CH9_3 //3
	
.. code-block:: c++

    ...
  }
  

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0MCInstLower.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.h
    :start-after: #if CH >= CH9_3
    :end-before: #endif //#if CH >= CH9_3

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0MCInstLower.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH9_3
    :end-before: #endif //#if CH >= CH9_3


The added code in ``Cpu0AsmPrinter.cpp``, as shown above, will call 
``LowerCPRESTORE()`` when the user runs the program with 
``llc -filetype=obj``.

The added code in ``Cpu0MCInstLower.cpp``, as shown above, handles the 
machine instructions for ``.cprestore``.

.. code-block:: console

  118-165-76-131:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -relocation-model=pic -filetype=
  obj ch9_1.bc -o ch9_1.cpu0.o
  118-165-76-131:input Jonathan$ hexdump  ch9_1.cpu0.o
  ...
  // .cprestore machine instruction “ 01 ad 00 18”
  00000d0 01 ad 00 18 09 20 00 00 01 2d 00 40 09 20 00 06
  ...
  
  118-165-67-25:input Jonathan$ cat ch9_1.cpu0.s
  ...
    .ent  _Z5sum_iiiiiii          # @_Z5sum_iiiiiii
  _Z5sum_iiiiiii:
  ...
    .cpload $t9 // assign $gp = $t9 by loader when loader load re-entry function 
                // (shared library) of _Z5sum_iiiiiii
    .set  nomacro
  # BB#0:
  ...
    .ent  main                    # @main
  ...
    .cprestore  24  // save $gp to 24($sp)
  ...

Running ``llc -static`` will emit the ``jsub`` instruction instead of 
``jalr``, as shown below:

.. code-block:: console

  118-165-76-131:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -relocation-model=static -filetype=
  asm ch9_1.bc -o ch9_1.cpu0.s
  118-165-76-131:input Jonathan$ cat ch9_1.cpu0.s
  ...
    jsub  _Z5sum_iiiiiii
  ...

Run ch9_1.bc with ``llc -filetype=obj``, and you will find the Cx of 
``jsub Cx`` is 0, since Cx is calculated by the linker, as shown below.  
Mips has the same 0 in its ``jal`` instruction.

.. code-block:: console

  // jsub _Z5sum_iiiiiii translate into 2B 00 00 00
  00F0: 2B 00 00 00 01 2D 00 34 00 ED 00 3C 09 DD 00 40 


The following code will emit ``ld $gp, ($gp save slot on stack)`` after ``jalr``
by creating the file ``Cpu0EmitGPRestore.cpp``, which runs as a function pass.

.. rubric:: lbdex/chapters/Chapter9_3/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH9_3
    :end-before: #endif
  
.. rubric:: lbdex/chapters/Chapter9_3/Cpu0TargetMachine.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: //@Cpu0PassConfig {
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH9_3 //1
    :end-before: #endif //#if CH >= CH9_3 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH9_3 //2
    :end-before: #endif //#if CH >= CH9_3 //2

  
.. rubric:: lbdex/chapters/Chapter9_3/Cpu0.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0.h
    :start-after: #if CH >= CH9_3
    :end-before: #endif //#if CH >= CH9_3
  
.. rubric:: lbdex/chapters/Chapter9_3/Cpu0EmitGPRestore.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0EmitGPRestore.cpp


Variable number of arguments
****************************

Until now, we supported a fixed number of arguments in formal function
definitions (Incoming Arguments). This subsection adds support for a variable
number of arguments, as the C language allows this feature.

Run ``Chapter9_3/`` with ``ch9_3_vararg.cpp`` and use the clang option
``clang -target mips-unknown-linux-gnu`` to get the following result:

.. code-block:: console

  118-165-76-131:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_3_vararg.cpp -emit-llvm -o ch9_3_vararg.bc
  118-165-76-131:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=false 
  -relocation-model=pic -filetype=asm ch9_3_vararg.bc -o ch9_3_vararg.cpu0.s
  118-165-76-131:input Jonathan$ cat ch9_3_vararg.cpu0.s
    .section .mdebug.abi32
    .previous
    .file "ch9_3_vararg.bc"
    .text
    .globl  _Z5sum_iiz
    .align  2
    .type _Z5sum_iiz,@function
    .ent  _Z5sum_iiz              # @_Z5sum_iiz
  _Z5sum_iiz:
    .frame  $fp,24,$lr
    .mask   0x00001000,-4
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -24
    st  $fp, 20($sp)            # 4-byte Folded Spill
    move    $fp, $sp
    ld  $2, 24($fp)     // amount
    st  $2, 16($fp)     // amount
    addiu $2, $zero, 0
    st  $2, 12($fp)     // i
    st  $2, 8($fp)     // val
    st  $2, 4($fp)      // sum
    addiu $3, $fp, 28
    st  $3, 0($fp)      // arg_ptr = 2nd argument = &arg[1], 
                // since &arg[0] = 24($sp)
    st  $2, 12($fp)
  $BB0_1:                                 # =>This Inner Loop Header: Depth=1
    ld  $2, 16($fp)
    ld  $3, 12($fp)
    cmp $sw, $3, $2        // compare(i, amount)
    jge $BB0_4
    nop
    jmp $BB0_2
    nop
  $BB0_2:                                 #   in Loop: Header=BB0_1 Depth=1 
                // i < amount
    ld  $2, 0($fp)
    addiu $3, $2, 4   // arg_ptr  + 4
    st  $3, 0($fp)  
    ld  $2, 0($2)     // *arg_ptr
    st  $2, 8($fp)
    ld  $3, 4($fp)      // sum
    add $2, $3, $2      // sum += *arg_ptr
    st  $2, 4($fp)
  # BB#3:                                 #   in Loop: Header=BB0_1 Depth=1
                // i >= amount
    ld  $2, 12($fp)
    addiu $2, $2, 1   // i++
    st  $2, 12($fp)
    jmp $BB0_1
    nop
  $BB0_4:
    ld  $2, 4($fp)
    move    $sp, $fp
    ld  $fp, 20($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 24
    ret $lr
    .set  macro
    .set  reorder
    .end  _Z5sum_iiz
  $tmp1:
    .size _Z5sum_iiz, ($tmp1)-_Z5sum_iiz
  
    .globl  _Z11test_varargv
    .align  2
    .type _Z11test_varargv,@function
    .ent  _Z11test_varargv                    # @_Z11test_varargv
  _Z11test_varargv:
    .frame  $sp,88,$lr
    .mask   0x00004000,-4
    .set  noreorder
    .cpload $t9
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -48
    st  $lr, 44($sp)            # 4-byte Folded Spill
    st  $fp, 40($sp)            # 4-byte Folded Spill
    move    $fp, $sp
    .cprestore  32
    addiu $2, $zero, 5
    st  $2, 24($sp)
    addiu $2, $zero, 4
    st  $2, 20($sp)
    addiu $2, $zero, 3
    st  $2, 16($sp)
    addiu $2, $zero, 2
    st  $2, 12($sp)
    addiu $2, $zero, 1
    st  $2, 8($sp)
    addiu $2, $zero, 0
    st  $2, 4($sp)
    addiu $2, $zero, 6
    st  $2, 0($sp)
    ld  $t9, %call16(_Z5sum_iiz)($gp)
    jalr  $t9
    nop
    ld  $gp, 28($fp)
    st  $2, 36($fp)
    move    $sp, $fp
    ld  $fp, 40($sp)            # 4-byte Folded Reload
    ld  $lr, 44($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 48
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z11test_varargv
  $tmp1:
    .size _Z11test_varargv, ($tmp1)-_Z11test_varargv


The analysis of output ``ch9_3_vararg.cpu0.s`` is shown in the comments above.

As described in the code in ``# BB#0``, we get the first argument ``amount`` from
``ld $2, 24($fp)``, since the stack size of the callee function ``_Z5sum_iiz()`` is
24. Then we set the argument pointer, ``arg_ptr``, to ``0($fp)``, which is
``&arg[1]``.

Next, we check ``i < amount`` in block ``$BB0_1``. If ``i < amount``, we enter
``$BB0_2``. In ``$BB0_2``, the code performs ``sum += *arg_ptr`` and
``arg_ptr += 4``. In ``# BB#3``, the code increments ``i`` with ``i += 1``.

To support variable numbers of arguments, the following code needs to be added in
``Chapter9_3/``.

The file ``ch9_3_template.cpp`` contains a C++ template example. It can also be
translated into Cpu0 backend code.

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@class Cpu0TargetLowering
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@CH3_4 1 {
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_3 //1
    :end-before: #endif
	
.. code-block:: c++

        ...
  .   };

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_3 //2
    :end-before: #endif
	
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH9_3 //3
    :end-before: #endif
	
.. code-block:: c++

      ...
  . };

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@vararg 1 {
    :end-before: #endif
	
.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #if CH >= CH8_1 //6
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //4
    :end-before: #endif //#if CH >= CH9_3 //4
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #endif //#if CH >= CH12_1 //7
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //5
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@LowerFormalArguments {
    :end-before: #if CH >= CH3_4
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //7
    :end-before: #endif
	
.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@#if CH >= CH9_2 //6 {
    :end-before: //@analyzeCallOperands body {
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //8
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@3 {
    :end-before: //@3 }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //9
    :end-before: #endif
	
.. code-block:: c++

      ...
    }
    ...
  }
  
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //10
    :end-before: #endif

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //11
    :end-before: #endif // #if CH >= CH9_3

.. rubric:: lbdex/input/ch9_3_template.cpp
.. literalinclude:: ../lbdex/input/ch9_3_template.cpp
    :start-after: /// start

MIPS QEMU reference [#mipsqemu]_ can be downloaded and run with GCC to verify the
result using the ``printf()`` function at this point.

We will verify the correctness of the code in the chapter "Verify backend on
Verilog simulator" through the Cpu0 Verilog-language machine.


Dynamic stack allocation support
********************************

Even though the C language rarely uses dynamic stack allocation, some other
languages rely on it frequently. The following C example demonstrates its use.

``Chapter9_3`` supports dynamic stack allocation with the following code added.

.. rubric:: lbdex/chapters/Chapter9_2/Cpu0FrameLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0FrameLowering.cpp
    :start-after: #if CH >= CH9_2
    :end-before: #endif // #if CH >= CH9_2

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0SEFrameLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: //@emitPrologue {
    :end-before: #if CH >= CH3_5 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #if CH >= CH9_3 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #if CH >= CH9_3 //2
    :end-before: //@ENABLE_GPRESTORE {
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #endif // #if CH >= CH3_5 //1
    :end-before: //}
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: //@emitEpilogue {
    :end-before: #if CH >= CH3_5 //2
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #if CH >= CH9_3 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #endif // #if CH >= CH3_5 //2
    :end-before: //}
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #if CH >= CH9_3 //5
    :end-before: //@callsEhReturn

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //2
    :end-before: //@vararg 1 {
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //3
    :end-before: #endif

.. code-block:: c++

  }

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0RegisterInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: //@getReservedRegs {
    :end-before: //@getReservedRegs body {
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: #if CH >= CH9_3 //2
    :end-before: #endif

.. code-block:: c++

  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: //@eliminateFrameIndex {
    :end-before: #if CH >= CH3_5
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: #ifdef ENABLE_GPRESTORE //2
    :end-before: else
	
.. code-block:: c++

  }

Run ``Chapter9_3`` with ``ch9_3_alloc.cpp`` to get the following correct result.

.. code-block:: console

  118-165-72-242:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_3_alloc.cpp -emit-llvm -o ch9_3_alloc.bc
  118-165-72-242:input Jonathan$ llvm-dis ch9_3_alloc.bc -o ch9_3_alloc.ll
  118-165-72-242:input Jonathan$ cat ch9_3_alloc.ll
  ; ModuleID = 'ch9_3_alloc.bc'
  target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-
  f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:
  32:64-S128"
  target triple = "x86_64-apple-macosx10.8.0"
  
  define i32 @_Z5sum_iiiiiii(i32 %x1, i32 %x2, i32 %x3, i32 %x4, i32 %x5, i32 %x6)
   nounwind uwtable ssp {
    ...
    %9 = alloca i8, i32 %8	// int* b = (int*)__builtin_alloca(sizeof(int) * 1 * x1);
    %10 = bitcast i8* %9 to i32*
    store i32* %10, i32** %b, align 4
    ...
  }
  ...

  118-165-72-242:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=false 
  -relocation-model=pic -filetype=asm ch9_3_alloc.bc -o ch9_3_alloc.cpu0.s
  118-165-72-242:input Jonathan$ cat ch9_3_alloc.cpu0.s 
  ...
      .globl  _Z10weight_sumiiiiii
    .align  2
    .type _Z10weight_sumiiiiii,@function
    .ent  _Z10weight_sumiiiiii    # @_Z10weight_sumiiiiii
  _Z10weight_sumiiiiii:
    .frame  $fp,48,$lr
    .mask   0x00005000,-4
    .set  noreorder
    .cpload $t9
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -48
    st  $lr, 44($sp)            # 4-byte Folded Spill
    st  $fp, 40($sp)            # 4-byte Folded Spill
    move   $fp, $sp
    .cprestore  24
    ld  $2, 68($fp)
    ld  $3, 64($fp)
    ld  $t9, 60($fp)
    ld  $7, 56($fp)
    st  $4, 36($fp)
    st  $5, 32($fp)
    st  $7, 28($fp)
    st  $t9, 24($fp)
    st  $3, 20($fp)
    st  $2, 16($fp)
    shl $2, $2, 2    // $2 = sizeof(int) * 1 * x2;
    addiu $2, $2, 7
    addiu $3, $zero, -8
    and $2, $2, $3
    addiu $sp, $sp, 0
    subu  $2, $sp, $2
    addu  $sp, $zero, $2  // set sp to the bottom of alloca area
    addiu $sp, $sp, 0
    st  $2, 12($fp)
    st  $2, 8($fp)
    ld  $2, 12($fp)
    ld  $3, 28($fp)
    st  $3, 0($2)    // *b = x3
    ld  $5, 32($fp)
    ld  $2, 36($fp)
    ld  $3, 20($fp)
    ld  $4, 28($fp)
    ld  $t9, 24($fp)
    ld  $7, 16($fp)
    addiu $sp, $sp, -24
    st  $7, 20($sp)
    st  $t9, 12($sp)
    st  $4, 8($sp)
    shl $3, $3, 1
    st  $3, 16($sp)
    addiu $3, $zero, 3
    mul $4, $2, $3
    ld  $t9, %call16(_Z3sumiiiiii)($gp)
    jalr  $t9
    nop
    ld  $gp, 24($fp)
    addiu $sp, $sp, 24
    st  $2, 4($fp)
    ld  $3, 8($fp)
    ld  $3, 0($3)
    addu  $2, $2, $3
    move   $sp, $fp
    ld  $fp, 40($sp)            # 4-byte Folded Reload
    ld  $lr, 44($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 48
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z10weight_sumiiiiii
  $func_end1:
    .size _Z10weight_sumiiiiii, ($func_end1)-_Z10weight_sumiiiiii
  ...

As you can see, dynamic stack allocation requires frame pointer register ``fp``
support. As shown in the assembly above, the ``sp`` is adjusted to ``sp - 48`` 
when entering the function by the instruction ``addiu $sp, $sp, -48``. 

Next, ``fp`` is set to ``sp``, which is positioned just above the area allocated
by ``alloca()``, as illustrated in :numref:`funccall-f4`, when the instruction 
``move $fp, $sp`` is encountered. 

After that, ``sp`` is moved to the space just below the ``alloca()`` allocation. 
Note that the space pointed to by ``b``, 
``*b = (int*)__builtin_alloca(sizeof(int) * 2 * x6)``, is allocated at run time,
because the size depends on the ``x1`` variable and cannot be determined at link
time.

:numref:`funccall-f5` illustrates how the stack pointer is restored to the
caller’s stack bottom. As described above, ``fp`` is set to the address just
above the ``alloca()`` area. 

The first step restores ``sp`` from ``fp`` using the instruction ``move $sp, $fp``.
Next, ``sp`` is adjusted back to the caller’s stack bottom using 
``addiu $sp, $sp, 40``.

.. _funccall-f4:
.. figure:: ../Fig/funccall/4.png
    :height: 279 px
    :width: 535 px
    :scale: 50 %
    :align: center

    Frame pointer changes when enter function

.. _funccall-f5:
.. figure:: ../Fig/funccall/5.png
    :height: 264 px
    :width: 528 px
    :scale: 50 %
    :align: center

    Stack pointer changes when exit function
    
.. _funccall-f6:
.. figure:: ../Fig/funccall/6.png
    :height: 394 px
    :width: 539 px
    :scale: 50 %
    :align: center

    fp and sp access areas

Using ``fp`` to keep the old stack pointer value is not the only solution. 
In fact, we can store the size of the ``alloca()`` spaces at a specific memory 
address and restore ``sp`` to its previous value by adding back the size of the 
``alloca()`` area.

Most ABIs, such as MIPS and ARM, access the area above ``alloca()`` using ``fp``
and the area below ``alloca()`` using ``sp``, as depicted in :numref:`funccall-f6`.

The reason for this design is performance in accessing local variables. Since 
RISC CPUs commonly use immediate offsets for load and store instructions, using 
both ``fp`` and ``sp`` to access the two separate areas of local variables 
provides better performance compared to using only ``sp``.

.. code-block:: console

  	ld	$2, 64($fp)
  	st	$3, 4($sp)
  	
Cpu0 uses ``fp`` and ``sp`` to access the areas above and below ``alloca()``, 
respectively. As shown in ``ch9_3_alloc.cpu0.s``, it accesses local variables 
(above the ``alloca()`` area) using ``fp`` offset, and accesses outgoing 
arguments (below the ``alloca()`` area) using ``sp`` offset.

Additionally, the instruction ``move $sp, $fp`` is an alias for the actual 
machine instruction ``addu $fp, $sp, $zero``. The machine code emitted is the 
latter, while the former is used for easier readability by users.

This alias is defined by the code added in Chapter3_2 and Chapter3_5, as shown 
below:

.. rubric:: lbdex/chapters/Chapter3_2/InstPrinter/Cpu0InstPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/InstPrinter/Cpu0InstPrinter.cpp
    :start-after: //@1 {
    :end-before: //@1 }

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH3_5 1
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH3_5 13
    :end-before: //#endif

Finally, the ``MFI->hasVarSizedObjects()`` defined in ``hasReservedCallFrame()`` 
of ``Cpu0SEFrameLowering.cpp`` is set to true when the IR contains 
``%9 = alloca i8, i32 %8``, which corresponds to 
``(int*)__builtin_alloca(sizeof(int) * 1 * x1);`` in C code.

This triggers generation of the assembly instruction ``addiu $sp, $sp, -24`` 
for ``ch9_3_alloc.cpp`` by invoking ``adjustStackPtr()`` inside 
``eliminateCallFramePseudoInstr()`` of ``Cpu0FrameLowering.cpp``.

The file ``ch9_3_longlongshift.cpp`` demonstrates support for the type 
**long long shift operations**, which can be tested now as shown below.

.. rubric:: lbdex/input/ch9_3_longlongshift.cpp
.. literalinclude:: ../lbdex/input/ch9_3_longlongshift.cpp
    :start-after: /// start

.. code-block:: console

  114-37-150-209:input Jonathan$ clang -O0 -target mips-unknown-linux-gnu 
  -c ch9_3_longlongshift.cpp -emit-llvm -o ch9_3_longlongshift.bc
  
  114-37-150-209:input Jonathan$ ~/llvm/test/build/bin/
  llvm-dis ch9_3_longlongshift.bc -o -
  ...
  ; Function Attrs: nounwind
  define i64 @_Z19test_longlong_shiftv() #0 {
    %a = alloca i64, align 8
    %b = alloca i64, align 8
    %c = alloca i64, align 8
    %d = alloca i64, align 8
    store i64 4, i64* %a, align 8
    store i64 18, i64* %b, align 8
    %1 = load i64* %b, align 8
    %2 = load i64* %a, align 8
    %3 = ashr i64 %1, %2
    store i64 %3, i64* %c, align 8
    %4 = load i64* %b, align 8
    %5 = load i64* %a, align 8
    %6 = shl i64 %4, %5
    store i64 %6, i64* %d, align 8
    %7 = load i64* %c, align 8
    %8 = load i64* %d, align 8
    %9 = add nsw i64 %7, %8
    ret i64 %9
  }
  ...
  114-37-150-209:input Jonathan$ ~/llvm/test/build/bin/llc 
  -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm 
  ch9_3_longlongshift.bc -o -
    .text
    .section .mdebug.abi32
    .previous
    .file "ch9_3_longlongshift.bc"
    .globl  _Z20test_longlong_shift1v
    .align  2
    .type _Z20test_longlong_shift1v,@function
    .ent  _Z20test_longlong_shift1v # @_Z20test_longlong_shift1v
  _Z20test_longlong_shift1v:
    .frame  $fp,56,$lr
    .mask   0x00005000,-4
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -56
    st  $lr, 52($sp)            # 4-byte Folded Spill
    st  $fp, 48($sp)            # 4-byte Folded Spill
    move   $fp, $sp
    addiu $2, $zero, 4
    st  $2, 44($fp)
    addiu $4, $zero, 0
    st  $4, 40($fp)
    addiu $5, $zero, 18
    st  $5, 36($fp)
    st  $4, 32($fp)
    ld  $2, 44($fp)
    st  $2, 8($sp)
    jsub  __lshrdi3
    nop
    st  $3, 28($fp)
    st  $2, 24($fp)
    ld  $2, 44($fp)
    st  $2, 8($sp)
    ld  $4, 32($fp)
    ld  $5, 36($fp)
    jsub  __ashldi3
    nop
    st  $3, 20($fp)
    st  $2, 16($fp)
    ld  $4, 28($fp)
    addu  $4, $4, $3
    cmp $sw, $4, $3
    andi  $3, $sw, 1
    addu  $2, $3, $2
    ld  $3, 24($fp)
    addu  $2, $3, $2
    addu  $3, $zero, $4
    move   $sp, $fp
    ld  $fp, 48($sp)            # 4-byte Folded Reload
    ld  $lr, 52($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 56
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z20test_longlong_shift1v
  $tmp0:
    .size _Z20test_longlong_shift1v, ($tmp0)-_Z20test_longlong_shift1v
  
    .globl  _Z20test_longlong_shift2v
    .align  2
    .type _Z20test_longlong_shift2v,@function
    .ent  _Z20test_longlong_shift2v # @_Z20test_longlong_shift2v
  _Z20test_longlong_shift2v:
    .frame  $fp,48,$lr
    .mask   0x00005000,-4
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -48
    st  $lr, 44($sp)            # 4-byte Folded Spill
    st  $fp, 40($sp)            # 4-byte Folded Spill
    move   $fp, $sp
    addiu $2, $zero, 48
    st  $2, 36($fp)
    addiu $2, $zero, 0
    st  $2, 32($fp)
    addiu $5, $zero, 10
    st  $5, 28($fp)
    lui $2, 22
    ori $4, $2, 26214
    st  $4, 24($fp)
    ld  $2, 36($fp)
    st  $2, 8($sp)
    jsub  __lshrdi3
    nop
    st  $3, 20($fp)
    st  $2, 16($fp)
    move   $sp, $fp
    ld  $fp, 40($sp)            # 4-byte Folded Reload
    ld  $lr, 44($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 48
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z20test_longlong_shift2v
  $tmp1:
    .size _Z20test_longlong_shift2v, ($tmp1)-_Z20test_longlong_shift2v


Variable sized array support
****************************

LLVM supports variable sized arrays (VLA) as introduced in C99 [#stacksave]_ 
[#wiki-vla]_.
The following code is added to support this feature. These intrinsics are set to 
expand, meaning LLVM replaces them with other DAG nodes during code generation.

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #if CH >= CH8_1 //6
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@llvm.stacksave
    :end-before: #endif

.. code-block:: c++

      ...
    }
    ...
  }

.. rubric:: lbdex/input/ch9_3_stacksave.cpp
.. literalinclude:: ../lbdex/input/ch9_3_stacksave.cpp
    :start-after: /// start

.. code-block:: console
  
  JonathantekiiMac:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_3_stacksave.cpp -emit-llvm -o ch9_3_stacksave.bc
  JonathantekiiMac:input Jonathan$ llvm-dis ch9_3_stacksave.bc -o -
  
  define i32 @_Z21test_stacksaverestorej(i32 zeroext %x) #0 {
    %1 = alloca i32, align 4
    %2 = alloca i8*
    %3 = alloca i32
    store i32 %x, i32* %1, align 4
    %4 = load i32, i32* %1, align 4
    %5 = call i8* @llvm.stacksave()
    store i8* %5, i8** %2
    %6 = alloca i8, i32 %4, align 1
    %7 = load i32, i32* %1, align 4
    %8 = getelementptr inbounds i8, i8* %6, i32 %7
    store i8 5, i8* %8, align 1
    %9 = load i32, i32* %1, align 4
    %10 = getelementptr inbounds i8, i8* %6, i32 %9
    %11 = load i8, i8* %10, align 1
    %12 = sext i8 %11 to i32
    store i32 1, i32* %3
    %13 = load i8*, i8** %2
    call void @llvm.stackrestore(i8* %13)
    ret i32 %12
  }

  JonathantekiiMac:input Jonathan$ ~/llvm/test/build/bin/llc 
  -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm 
  ch9_3_stacksave.bc -o -
  ...

Function related Intrinsics support
***********************************

I believe these LLVM intrinsic IRs are used for implementing exception handling
[#excepthandle]_ [#returnaddr]_. With these IRs, a programmer can record the
frame address and return address, which can be used in C++ programs to
implement exception handlers, as shown in the example below. 

To support these LLVM intrinsic IRs, the following code is added to the Cpu0
backend.

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //0.5
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //0.7
    :end-before: #endif
    
.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #if CH >= CH8_1 //6
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //4.5
    :end-before: #endif

.. code-block:: c++

      ...
    }
    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //5.5
    :end-before: #endif
    

frameaddress and returnaddress intrinsics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following input to get the corresponding result.

.. rubric:: lbdex/input/ch9_3_frame_return_addr.cpp
.. literalinclude:: ../lbdex/input/ch9_3_frame_return_addr.cpp
    :start-after: /// start

.. code-block:: console
  
  JonathantekiiMac:input Jonathan$ ~/llvm/test/build/bin/
  llvm-dis ch9_3_frame_return_addr.bc -o -
  ...
  ; Function Attrs: nounwind
  define i32 @_Z20display_frameaddressv() #0 {
    %1 = call i8* @llvm.frameaddress(i32 0)
    %2 = ptrtoint i8* %1 to i32
    ret i32 %2
  }
  
  ; Function Attrs: nounwind readnone
  declare i8* @llvm.frameaddress(i32) #1
  
  define i32 @_Z22display_returnaddressv() #2 {
    %a = alloca i32, align 4
    %1 = call i8* @llvm.returnaddress(i32 0)
    %2 = ptrtoint i8* %1 to i32
    store i32 %2, i32* %a, align 4
    %3 = call i32 @_Z2fnv()
    %4 = load i32, i32* %a, align 4
    ret i32 %4
  }
  
  JonathantekiiMac:input Jonathan$ ~/llvm/test/build/bin/llc 
  -march=cpu0 -relocation-model=static -filetype=asm ch9_3_frame_return_addr.bc 
  -o -
  	.text
  	.section .mdebug.abiO32
  	.previous
  	.file "ch9_3_frame_return_addr.bc"
  	.globl	_Z20display_frameaddressv
  	.align	2
  	.type _Z20display_frameaddressv,@function
  	.ent	_Z20display_frameaddressv # @_Z20display_frameaddressv
  _Z20display_frameaddressv:
  	.frame	$fp,8,$lr
  	.mask		0x00001000,-4
  	.set	noreorder
  	.set	nomacro
  # BB#0:
  	addiu $sp, $sp, -8
  	st	$fp, 4($sp)							# 4-byte Folded Spill
  	move	 $fp, $sp
  	addu	$2, $zero, $fp
  	move	 $sp, $fp
  	ld	$fp, 4($sp)							# 4-byte Folded Reload
  	addiu $sp, $sp, 8
  	ret $lr
  	nop
  	.set	macro
  	.set	reorder
  	.end	_Z20display_frameaddressv
  $func_end0:
  	.size _Z20display_frameaddressv, ($func_end0)-_Z20display_frameaddressv
  
  	.globl	_Z22display_returnaddress1v
  	.align	2
  	.type _Z22display_returnaddress1v,@function
  	.ent	_Z22display_returnaddress1v # @_Z22display_returnaddress1v
  _Z22display_returnaddress1v:
  	.cfi_startproc
  	.frame	$fp,24,$lr
  	.mask		0x00005000,-4
  	.set	noreorder
  	.set	nomacro
  # BB#0:
  	addiu $sp, $sp, -24
  $tmp0:
  	.cfi_def_cfa_offset 24
  	st	$lr, 20($sp)						# 4-byte Folded Spill
  	st	$fp, 16($sp)						# 4-byte Folded Spill
  $tmp1:
  	.cfi_offset 14, -4
  $tmp2:
  	.cfi_offset 12, -8
  	move	 $fp, $sp
  $tmp3:
  	.cfi_def_cfa_register 12
  	st	$lr, 12($fp)
  	jsub	_Z2fnv
  	nop
  	ld	$2, 12($fp)
  	move	 $sp, $fp
  	ld	$fp, 16($sp)						# 4-byte Folded Reload
  	ld	$lr, 20($sp)						# 4-byte Folded Reload
  	addiu $sp, $sp, 24
  	ret $lr
  	nop
  	.set	macro
  	.set	reorder
  	.end	_Z22display_returnaddress1v
  $func_end1:
  	.size _Z22display_returnaddress1v, ($func_end1)-_Z22display_returnaddress1v
  	.cfi_endproc


The asm ``ld     $2, 12($fp)`` in function ``_Z22display_returnaddress1v``
reloads ``$lr`` to ``$2`` after ``jsub _Z3fnv``. The reason that Cpu0 doesn't
produce ``addiu $2, $zero, $lr`` is that, if a buggy program in ``_Z3fnv``
modifies the ``$lr`` value without following the ABI, then it will load an
incorrect ``$lr`` into ``$2``.

The following code kills the ``$lr`` register and makes the reference to ``$lr``
by loading from a stack slot rather than using the register directly.

.. rubric:: lbdex/chapters/Chapter9_1/Cpu0SEFrameLowering.cpp
.. code-block:: c++
  
  bool Cpu0SEFrameLowering::
  spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            const std::vector<CalleeSavedInfo> &CSI,
                            const TargetRegisterInfo *TRI) const { 
    ...
    for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
      // Add the callee-saved register as live-in. Do not add if the register is
      // LR and return address is taken, because it has already been added in
      // method Cpu0TargetLowering::LowerRETURNADDR.
      // It's killed at the spill, unless the register is LR and return address
      // is taken.
      unsigned Reg = CSI[i].getReg();
      bool IsRAAndRetAddrIsTaken = (Reg == Cpu0::LR)
          && MF->getFrameInfo()->isReturnAddressTaken();
      if (!IsRAAndRetAddrIsTaken)
        EntryBlock->addLiveIn(Reg);
  
      // Insert the spill to the stack frame.
      bool IsKill = !IsRAAndRetAddrIsTaken;
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.storeRegToStackSlot(*EntryBlock, MI, Reg, IsKill,
                              CSI[i].getFrameIdx(), RC, TRI);
    }
    ...
  }


eh.return intrinsic
^^^^^^^^^^^^^^^^^^^^

Considering the following code,

.. rubric:: unwind example
.. code-block:: c++

  int func() {
    if (...) {
      throw std::bad_alloc();
    }
  }
  
  int A() {
    try {
      func();
    }
    catch(...) {
      ...
    }
  }
  
  int B() {
    try {
      func();
      A();
    }
    catch(...) {
      ...
    }
  }

When B() -> calls func() -> exception occurs, the frame is unwound to B and 
handled by B's exception handler. When B() -> calls A() -> calls func() -> 
exception occurs, the frame is unwound to A and handled by A's exception 
handler.

``__builtin_eh_return(offset, handler)`` adjusts the stack by the given offset 
and then jumps to the handler. ``__builtin_eh_return`` is used in the GCC 
unwinder (libgcc), but not in the LLVM unwinder (libunwind) [#ehreturn]_.

Besides ``lowerRETURNADDR()`` in ``Cpu0ISelLowering``, the following code is 
only for ``eh.return`` support. It can run with the input 
``ch9_3_detect_exception.cpp`` as shown below.

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0SEFrameLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: //@emitPrologue {
    :end-before: #if CH >= CH3_5 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #if CH >= CH9_3 //1.5
    :end-before: #endif
    
.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: //@emitEpilogue {
    :end-before: #if CH >= CH3_5 //2
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #if CH >= CH9_3 //4
    :end-before: #endif
    
.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: //@determineCalleeSaves {
    :end-before: //@determineCalleeSaves-body
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: //@callsEhReturn
    :end-before: #endif
    
.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH9_3 //3
    :end-before: //#endif

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0SEInstrInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.h
    :start-after: #if CH >= CH9_3
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0SEInstrInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: //@expandPostRAPseudo
    :end-before: //@expandPostRAPseudo-body
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: #if CH >= CH9_3 //1
    :end-before: #endif
    
.. code-block:: c++

    ...
  }
  
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: #if CH >= CH9_3 //2
    :end-before: #endif

.. rubric:: lbdex/input/ch9_3_detect_exception.cpp
.. literalinclude:: ../lbdex/input/ch9_3_detect_exception.cpp
    :start-after: /// start

.. code-block:: console
  
  114-37-150-48:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_3_detect_exception.cpp -emit-llvm -o ch9_3_detect_exception.bc
  114-37-150-48:input Jonathan$ ~/llvm/test/build/bin/llvm-dis 
  ch9_3_detect_exception.bc -o -
  ; ModuleID = 'ch9_3_detect_exception.bc'
  target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
  target triple = "mips-unknown-linux-gnu"
  
  @exceptionOccur = global i8 0, align 1
  @returnAddr = global i8* null, align 4
  
  ; Function Attrs: nounwind
  define void @_Z17exception_handlerv() #0 {
    %frameaddr = alloca i32, align 4
    store i8 1, i8* @exceptionOccur, align 1
    %1 = call i8* @llvm.frameaddress(i32 0)
    %2 = ptrtoint i8* %1 to i32
    store i32 %2, i32* %frameaddr, align 4
    %3 = load i8*, i8** @returnAddr, align 4
    call void @llvm.eh.return.i32(i32 0, i8* %3)
    unreachable
                                                    ; No predecessors!
    ret void
  }
  
  ; Function Attrs: nounwind readnone
  declare i8* @llvm.frameaddress(i32) #1
  
  ; Function Attrs: nounwind
  declare void @llvm.eh.return.i32(i32, i8*) #2
  
  define weak i32 @_Z21test_detect_exceptionb(i1 zeroext %exception) #3 {
    %1 = alloca i8, align 1
    %handler = alloca i8*, align 4
    %2 = zext i1 %exception to i8
    store i8 %2, i8* %1, align 1
    store i8 0, i8* @exceptionOccur, align 1
    store i8* bitcast (void ()* @_Z17exception_handlerv to i8*), i8** %handler, align 4
    %3 = load i8, i8* %1, align 1
    %4 = trunc i8 %3 to i1
    br i1 %4, label %5, label %8
  
  ; <label>:5                                       ; preds = %0
    %6 = call i8* @llvm.returnaddress(i32 0)
    store i8* %6, i8** @returnAddr, align 4
    %7 = load i8*, i8** %handler, align 4
    call void @llvm.eh.return.i32(i32 0, i8* %7)
    unreachable
  
  ; <label>:8                                       ; preds = %0
    ret i32 0
  }
  
  ; Function Attrs: nounwind readnone
  declare i8* @llvm.returnaddress(i32) #1
  
  attributes #0 = { nounwind ... }
  attributes #1 = { nounwind readnone }
  attributes #2 = { nounwind }
  attributes #3 = { "less-precise-fpmad"="false" ... }
  ...
  
  114-37-150-48:input Jonathan$ ~/llvm/test/build/bin/llc 
  -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm 
  ch9_3_detect_exception.bc -o -
    .text
    .section .mdebug.abiO32
    .previous
    .file "ch9_3_detect_exception.bc"
    .globl  _Z17exception_handlerv
    .align  2
    .type _Z17exception_handlerv,@function
    .ent  _Z17exception_handlerv  # @_Z17exception_handlerv
  _Z17exception_handlerv:
    .frame  $fp,16,$lr
    .mask   0x00001000,-4
    .set  noreorder
    .cpload $t9
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -16
    st  $fp, 12($sp)            # 4-byte Folded Spill
    st  $4, 4($fp)
    st  $5, 0($fp)
    move   $fp, $sp
    lui $2, %got_hi(exceptionOccur)
    addu  $2, $2, $gp
    ld  $2, %got_lo(exceptionOccur)($2)
    addiu $3, $zero, 1
    sb  $3, 0($2)
    st  $fp, 8($fp)
    lui $2, %got_hi(returnAddr)
    addu  $2, $2, $gp
    ld  $2, %got_lo(returnAddr)($2)
    ld  $2, 0($2)
    addiu $3, $zero, 0
    move   $sp, $fp
    ld  $4, 4($fp)
    ld  $5, 0($fp)
    ld  $fp, 12($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 16
    move   $t9, $2
    move   $lr, $2
    addu  $sp, $sp, $3
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z17exception_handlerv
  $func_end0:
    .size _Z17exception_handlerv, ($func_end0)-_Z17exception_handlerv
  
    .weak _Z21test_detect_exceptionb
    .align  2
    .type _Z21test_detect_exceptionb,@function
    .ent  _Z21test_detect_exceptionb # @_Z21test_detect_exceptionb
  _Z21test_detect_exceptionb:
    .cfi_startproc
    .frame  $fp,24,$lr
    .mask   0x00001000,-4
    .set  noreorder
    .cpload $t9
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -24
  $tmp0:
    .cfi_def_cfa_offset 24
    st  $fp, 20($sp)            # 4-byte Folded Spill
  $tmp1:
    .cfi_offset 12, -4
    st  $4, 8($fp)
    st  $5, 4($fp)
  $tmp2:
    .cfi_offset 4, -16
  $tmp3:
    .cfi_offset 5, -20
    move   $fp, $sp
  $tmp4:
    .cfi_def_cfa_register 12
    sb  $4, 16($fp)
    lui $2, %got_hi(exceptionOccur)
    addu  $2, $2, $gp
    ld  $2, %got_lo(exceptionOccur)($2)
    addiu $3, $zero, 0
    sb  $3, 0($2)
    lui $2, %got_hi(_Z17exception_handlerv)
    addu  $2, $2, $gp
    ld  $2, %got_lo(_Z17exception_handlerv)($2)
    st  $2, 12($fp)
    lbu $2, 16($fp)
    andi  $2, $2, 1
    beq $2, $zero, .LBB1_2
    nop
    jmp .LBB1_1
    nop
  .LBB1_2:
    addiu $2, $zero, 0
    move   $sp, $fp
    ld  $4, 8($fp)
    ld  $5, 4($fp)
    ld  $fp, 20($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 24
    ret $lr
    nop
  .LBB1_1:
    lui $2, %got_hi(returnAddr)
    addu  $2, $2, $gp
    ld  $2, %got_lo(returnAddr)($2)
    st  $lr, 0($2)
    ld  $2, 12($fp)
    addiu $3, $zero, 0
    move   $sp, $fp
    ld  $4, 8($fp)
    ld  $5, 4($fp)
    ld  $fp, 20($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 24
    move   $t9, $2
    move   $lr, $2
    addu  $sp, $sp, $3
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z21test_detect_exceptionb
  $func_end1:
    .size _Z21test_detect_exceptionb, ($func_end1)-_Z21test_detect_exceptionb
    .cfi_endproc
  
    .type exceptionOccur,@object  # @exceptionOccur
    .bss
    .globl  exceptionOccur
  exceptionOccur:
    .byte 0                       # 0x0
    .size exceptionOccur, 1
  
    .type returnAddr,@object      # @returnAddr
    .globl  returnAddr
    .align  2
  returnAddr:
    .4byte  0
    .size returnAddr, 4
    ...


If you disable ``__attribute__ ((weak))`` in the C file, then the IR will have 
``nounwind`` in attributes #3. The side effect in the ASM output is that no 
``.cfi_offset`` is issued, as seen in the function ``exception_handler()``.

This example code of exception handler implementation can get the frame, 
return address, and call the exception handler by calling ``__builtin_xxx`` 
in Clang using the C language, without introducing any assembly instruction.
This example can be verified in the chapter "Cpu0 ELF linker" of the other book 
"LLVM Tool Chain for Cpu0" [#cpu0lld]_.

By examining the global variable ``exceptionOccur``, which is true or false, 
the program will set the control flow to ``exception_handler()`` or skip it 
accordingly.

eh.dwarf intrinsic
^^^^^^^^^^^^^^^^^^

Besides ``lowerADD()`` in ``Cpu0ISelLowering``, the following code is only 
for supporting ``eh.dwarf``. It can be run with the input ``eh-dwarf-cfa.ll`` 
as shown below.

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0SEFrameLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEFrameLowering.cpp
    :start-after: #if CH >= CH9_3 //2
    :end-before: //@ Insert instruction "move $fp, $sp" at this location.
    
.. code-block:: c++

    ...
  }

.. rubric:: lbdex/input/eh-dwarf-cfa.ll
.. literalinclude:: ../lbdex/input/eh-dwarf-cfa.ll


bswap intrinsic
^^^^^^^^^^^^^^^

Cpu0 supports the LLVM intrinsic ``bswap`` [#bswapintrnsic]_.

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH9_3 //2.5
    :end-before: #endif

.. code-block:: c++

    ...
  }

.. rubric:: lbdex/input/ch9_3_bswap.cpp
.. literalinclude:: ../lbdex/input/ch9_3_bswap.cpp
    :start-after: /// start


.. code-block:: console
  
  114-37-150-48:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch9_3_bswap.cpp -emit-llvm -o ch9_3_bswap.bc
  114-37-150-48:input Jonathan$ ~/llvm/test/build/bin/llvm-dis 
  ch9_3_bswap.bc -o -
  ...
  define i32 @_Z12test_bswap16v() #0 {
    %a = alloca i32, align 4
    %result = alloca i32, align 4
    store volatile i32 4660, i32* %a, align 4
    %1 = load volatile i32, i32* %a, align 4
    %2 = trunc i32 %1 to i16
    %3 = call i16 @llvm.bswap.i16(i16 %2)
    %4 = zext i16 %3 to i32
    %5 = xor i32 %4, 13330
    store i32 %5, i32* %result, align 4
    %6 = load i32, i32* %result, align 4
    ret i32 %6
  }
  ...


Add specific backend intrinsic function
***************************************

LLVM intrinsic functions are designed to extend LLVM IRs for hardware
acceleration in compiler design [#extendintrnsic]_.
Many CPUs implement their own intrinsic functions for hardware-specific
instructions that improve performance.

Some GPUs use the LLVM infrastructure as their OpenGL/OpenCL backend compiler
and rely on many LLVM-extended intrinsic functions.

To demonstrate how to use backend proprietary intrinsic functions to support
specific instructions for performance improvement in domain-specific languages,
Cpu0 adds an intrinsic function ``@llvm.cpu0.gcd`` for its
greatest common divisor (GCD) instruction.

This instruction demonstrates how to implement a custom intrinsic in LLVM;
however, it is not implemented in the Verilog Cpu0 hardware.

The code is as follows,

.. rubric:: lbdex/llvm/modify/llvm/include/llvm/IR/Intrinsics.td
.. code-block:: c++

    ...
    include "llvm/IR/IntrinsicsCpu0.td"
    ...
  
.. rubric:: lbdex/llvm/modify/llvm/include/llvm/IR/IntrinsicsCpu0.td
.. literalinclude:: ../lbdex/llvm/modify/llvm/include/llvm/IR/IntrinsicsCpu0.td

.. rubric:: lbdex/chapters/Chapter9_3/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH9_3 //1
    :end-before: //#endif //#if CH >= CH9_3 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH9_3 //5
    :end-before: //#endif //#if CH >= CH9_3 //5

When running ``llc`` with ``cpu0_gcd.ll``, it generates the ``gcd`` machine
instruction. Meanwhile, running ``cpu0_gcd_soft.ll`` results in a call to the
``cpu0_gcd_soft`` function.

In other words, ``@llvm.cpu0.gcd`` is an intrinsic function mapped to the ``gcd``
machine instruction, while ``@cpu0_gcd_soft`` is a regular function implemented
in software.

For undefined intrinsic functions in Cpu0, such as ``fmul float %0, %1``, LLVM
will compile them into function calls like ``jsub fmul`` for Cpu0
[#lbd-fmul]_.

The file ``test_memcpy.ll`` is an example of an ``IntrWriteMem`` instruction,
which prevents the operation from being optimized out.


Summary
-------

Now, the Cpu0 backend can handle both integer function calls and control
statements, similar to the example code in the LLVM frontend tutorial.

It can also translate some of the C++ object-oriented programming language into
Cpu0 instructions without much additional backend effort, because the frontend
handles most of the complexity for meeting C++ requirement.

LLVM is a well-structured system that follows compiler theory closely. Any
backend of LLVM benefits from this structure.

The best part of the three-tier compiler architecture is that backends will
automatically support more languages as the frontend expands its language
support, as long as no new IRs are introduced.


.. [#computer_arch_interface] Computer Organization and Design: The Hardware/Software Interface 1st edition (The Morgan Kaufmann Series in Computer Architecture and Design)

.. [#mipsasm] http://math-atlas.sourceforge.net/devel/assembly/007-2418-003.pdf

.. [#abi] http://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf

.. [#secglobal] http://jonathan2251.github.io/lbd/globalvar.html#global-variable

.. [#wikitailcall] http://en.wikipedia.org/wiki/Tail_call

.. [#tailcallopt] http://llvm.org/docs/CodeGenerator.html#tail-call-optimization

.. [#callconv] http://llvm.org/docs/LangRef.html#calling-conventions

.. [#mipsqemu] http://developer.mips.com/clang-llvm/

.. [#stacksave] http://www.llvm.org/docs/LangRef.html#llvm-stacksave-intrinsic

.. [#wiki-vla] https://en.wikipedia.org/wiki/Variable-length_array

.. [#excepthandle] http://llvm.org/docs/ExceptionHandling.html#overview

.. [#returnaddr] http://llvm.org/docs/LangRef.html#llvm-returnaddress-intrinsic

.. [#ehreturn] https://llvm.org/docs/ExceptionHandling.html#exception-handling-support-on-the-target

.. [#cpu0lld] http://jonathan2251.github.io/lbt/lld.html

.. [#bswapintrnsic] http://llvm.org/docs/LangRef.html#llvm-bswap-intrinsics

.. [#extendintrnsic] https://llvm.org/docs/ExtendingLLVM.html

.. [#lbd-fmul] file:///Users/cschen/git/lbd/build/html/othertype.html#float-and-double
