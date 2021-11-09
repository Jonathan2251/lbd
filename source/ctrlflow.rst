.. _sec-controlflow:

Control flow statements
=======================

.. contents::
   :local:
   :depth: 4

This chapter illustrates the corresponding IR for control flow statements, such as 
**“if else”**, **“while”** and **“for”** loop statements in C, and how to 
translate these control flow statements of llvm IR into Cpu0 instructions in 
section I. In `section Remove useless JMP`_, 
an optimization pass of control flow for backend is introduced. 
It's a simple tutorial program to let readers know how to add a 
backend optimization pass and program it. 
`section Conditional instruction`_, includes the Conditional 
Instructions Handling since clang will generate specific IRs, select and 
select_cc, to support the backend optimiation in control flow statement.

Pipeline architecture
---------------------

.. _ctrl_pipeline:
.. figure:: ../Fig/ctrlflow/5_stage_pipeline.png
  :width: 1932 px
  :height: 792 px
  :scale: 50 %
  :align: center

  5 stages of pipeline

- IF: Instruction fetch cycle; ID: Instruction decode/register fetch cycle; 
  EX: Execution/effective address cycle; MEM: Memory access; WB: Write-back cycle.


.. _ctrlflow-cache_banks:
.. figure:: ../Fig/ctrlflow/cache_banks.png
  :width: 1540 px
  :height: 506 px
  :scale: 50 %
  :align: center

  Interleaved cache banks

.. _ctrl-super_pipeline:
.. figure:: ../Fig/ctrlflow/super_pipeline.png
  :width: 1894 px
  :height: 984 px
  :scale: 50 %
  :align: center

  Super pipeline


With cache banks as :numref:`ctrlflow-cache_banks`, :numref:`ctrl-super_pipeline` 
and superscalar (multi-issues) pipeline archtecture are emerged.


Control flow statement
-----------------------

Run ch8_1_1.cpp with clang will get result as follows,

.. rubric:: lbdex/input/ch8_1_1.cpp
.. literalinclude:: ../lbdex/input/ch8_1_1.cpp
    :start-after: /// start

.. code-block:: llvm

    ...
    %0 = load i32* %a, align 4
    %cmp = icmp eq i32 %0, 0
    br i1 %cmp, label %if.then, label %if.end

  ; <label>:3:                                      ; preds = %0
    %1 = load i32* %a, align 4
    %inc = add i32 %1, 1
    store i32 %inc, i32* %a, align 4
    br label %if.end
    ...


The **“icmp ne”** stands for integer compare NotEqual, **“slt”** stands for Set 
Less Than, **“sle”** stands for Set Less or Equal. 
Run version Chapter8_1/ with ``llc  -view-isel-dags`` or ``-debug`` option, you 
can see the **if** statement is translated into 
(br (brcond (%1, setcc(%2, Constant<c>, setne)), BasicBlock_02), BasicBlock_01).
Ignore %1, then we will get the form (br (brcond (setcc(%2, Constant<c>, setne)), 
BasicBlock_02), BasicBlock_01). 
For explanation, listing the IR DAG as follows,

.. code-block:: console
  
  Optimized legalized selection DAG: BB#0 '_Z11test_ifctrlv:entry'
    SelectionDAG has 12 nodes:
      ...
      t5: i32,ch = load<Volatile LD4[%a]> t4, FrameIndex:i32<0>, undef:i32
          t16: i32 = setcc t5, Constant:i32<0>, setne:ch
        t11: ch = brcond t5:1, t16, BasicBlock:ch<if.end 0x10305a338>
      t13: ch = br t11, BasicBlock:ch<if.then 0x10305a288>
    
We want to translate them into Cpu0 instruction DAGs as follows,

.. code-block:: console

    addiu %3, ZERO, Constant<c>
    cmp %2, %3
    jne BasicBlock_02
    jmp BasicBlock_01

For the last IR br, we translate unconditional branch (br BasicBlock_01) into 
jmp BasicBlock_01 by the following pattern definition,

.. rubric:: lbdex/chapters/Chapter8_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH8_1 3
    :end-before: //#endif

.. code-block:: console

    ...
    def JMP     : UncondBranch<0x26, "jmp">;

The pattern [(br bb:$imm24)] in class UncondBranch is translated into jmp 
machine instruction.
The translation for the pair Cpu0 instructions, **cmp** and **jne**, is not 
happened before this chapter. 
To solve this chained IR to machine instructions translation, we define the 
following pattern,

.. rubric:: lbdex/chapters/Chapter8_1/Cpu0InstrInfo.td
.. code:: c++

  // brcond patterns
  multiclass BrcondPatsCmp<RegisterClass RC, Instruction JEQOp, Instruction JNEOp, 
    Instruction JLTOp, Instruction JGTOp, Instruction JLEOp, Instruction JGEOp, 
    Instruction CMPOp> {
  ...
  def : Pat<(brcond (i32 (setne RC:$lhs, RC:$rhs)), bb:$dst),
            (JNEOp (CMPOp RC:$lhs, RC:$rhs), bb:$dst)>;
  ...
  def : Pat<(brcond RC:$cond, bb:$dst),
            (JNEOp (CMPOp RC:$cond, ZEROReg), bb:$dst)>;
  ...
  }

Since the aboved BrcondPats pattern uses RC (Register Class) as operand, the 
following ADDiu pattern defined in Chapter2 will generate instruction 
**addiu** before the instruction **cmp** for the first IR, 
**setcc(%2, Constant<c>, setne)**, as above.

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 17
    :end-before: //#endif

The definition of BrcondPats supports setne, seteq, setlt, ..., register operand
compare and setult, setugt, ..., for unsigned int type. In addition to seteq 
and setne, we define setueq and setune by refering Mips code, even though 
we don't find how to generate setune IR from C language. 
We have tried to define unsigned int type, but clang still generates setne 
instead of setune. 
The order of Pattern Search is from the order of their appearing in context. 
The last pattern (brcond RC:$cond, bb:$dst) means branch to $dst 
if $cond != 0. So we set the corresponding translation to 
(JNEOp (CMPOp RC:$cond, ZEROReg), bb:$dst).

The CMP instruction will set the result to register SW, and then JNE check the 
condition based on SW status as :numref:`ctrlflow-f1`. 
Since SW belongs to a different register class, it will be
correct even an instruction is inserted between CMP and JNE as follows,

.. _ctrlflow-f1:
.. figure:: ../Fig/ctrlflow/1.png
  :width: 446 px
  :height: 465 px
  :scale: 50 %
  :align: center

  JNE (CMP $r2, $r3),

.. code-block:: console

    cmp %2, %3
    addiu $r1, $r2, 3   // $r1 register never be allocated to $SW because in 
                        //  class ArithLogicI, GPROut is the output register 
                        //  class and the GPROut is defined without $SW in 
                        //  Cpu0RegisterInforGPROutForOther.td
    jne BasicBlock_02


The reserved registers setting by the following 
function code we defined before,

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0RegisterInfo.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0RegisterInfo.cpp
    :start-after: //@getReservedRegs {
    :end-before: //@eliminateFrameIndex {

Although the following definition in Cpu0RegisterInfo.td has no real effect in 
Reserved Registers, it's better to comment the Reserved Registers in it for 
readability. Setting SW in both register classes CPURegs and SR to allow access
SW by RISC instructions like ``andi``, and allow programmers use 
traditional assembly instruction ``cmp``. 
The copyPhysReg() is called when both DestReg and SrcReg are belonging to 
different Register Classes. 

.. rubric:: lbdex/chapters/Chapter2/Cpu0RegisterInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.td
    :start-after: //@Register Classes
    :end-before: //#if CH >= CH4_1 2
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.td
    :start-after: //@Status Registers class
    :end-before: //@Co-processor 0 Registers class
  

.. rubric:: lbdex/chapters/Chapter2/Cpu0RegisterInfoGPROutForOther.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfoGPROutForOther.td


Chapter8_1/ include support for control flow statement. 
Run with it as well as the following ``llc`` option, you will get the obj file. 
Dump it's content by gobjdump or hexdump after as follows,

.. code-block:: console

    118-165-79-206:input Jonathan$ /Users/Jonathan/llvm/test/
    build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic 
    -filetype=asm ch8_1_1.bc -o -
    ...
    ld  $4, 36($fp)
    cmp $sw, $4, $3
    jne $BB0_2
    jmp $BB0_1
  $BB0_1:                                 # %if.then
    ld  $4, 36($fp)
    addiu $4, $4, 1
    st  $4, 36($fp)
  $BB0_2:                                 # %if.end
    ld  $4, 32($fp)
    ...

.. code-block:: console
    
    118-165-79-206:input Jonathan$ /Users/Jonathan/llvm/test/
    build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic 
    -filetype=obj ch8_1_1.bc -o ch8_1_1.cpu0.o

    118-165-79-206:input Jonathan$ hexdump ch8_1_1.cpu0.o 
        // jmp offset is 0x10=16 bytes which is correct
    0000080 ...................................... 10 43 00 00
    0000090 31 00 00 10 36 00 00 00 ..........................

The immediate value of jne (op 0x31) is 16; The offset between jne and $BB0_2 
is 20 (5 words = 5*4 bytes). Suppose the jne address is X, then the label 
$BB0_2 is X+20. 
Cpu0's instruction set is designed as a RISC CPU with 5 stages of pipeline just 
like 5 stages of Mips. 
Cpu0 do branch instruction execution at decode stage which like mips too. 
After the jne instruction fetched, the PC (Program Counter) is X+4 since cpu0 
update PC at fetch stage. 
The $BB0_2 address is equal to PC+16 for the jne branch instruction execute at 
decode stage. 
List and explain this again as follows,

.. code-block:: console

                // Fetch instruction stage for jne instruction. The fetch stage 
                // can be divided into 2 cycles. First cycle fetch the 
                // instruction. Second cycle adjust PC = PC+4. 
    jne $BB0_2  // Do jne compare in decode stage. PC = X+4 at this stage. 
                // When jne immediate value is 16, PC = PC+16. It will fetch 
                //  X+20 which equal to label $BB0_2 instruction, ld $4, 32($sp). 
    nop
  $BB0_1:                                 # %if.then
    ld  $4, 36($fp)
    addiu $4, $4, 1
    st  $4, 36($fp)
  $BB0_2:                                 # %if.end
    ld  $4, 32($fp)

If Cpu0 do **"jne"** in execution stage, then we should set PC=PC+12, 
offset of ($BB0_2, jne $BB02) – 8, in this example.

In reality, the conditional branch is important in performance of CPU design. 
According bench mark information, every 7 instructions will meet 1 branch 
instruction in average. 
The cpu032I spends 2 instructions in conditional branch, (jne(cmp...)), while 
cpu032II use one instruction (bne) as follws,

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic 
  -filetype=asm ch8_1_1.bc -o -
    ...
  	cmp	$sw, $4, $3
  	jne	$sw, $BB0_2
  	jmp	$BB0_1
  $BB0_1:
  
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic 
  -filetype=asm ch8_1_1.bc -o -
    ...
  	bne	$4, $zero, $BB0_2
  	jmp	$BB0_1
  $BB0_1:


Beside brcond explained in this section, above code also include DAG opcode 
**br_jt** and label **JumpTable** which occurs during DAG translation for
some kind of program.

The ch8_1_ctrl.cpp include **“nest if”** **“for loop”**, **“while loop”**, 
**“continue”**, **“break”** and **“goto”**.
The ch8_1_br_jt.cpp is for **br_jt** and **JumpTable** test.
The ch8_1_blockaddr.cpp is for **blockaddress** and **indirectbr** test.
You can run with them if you like to test more.

List the control flow statements of C, IR, DAG and Cpu0 instructions as the 
following table.

.. table:: Control flow statements of C, IR, DAG and Cpu0 instructions

  ==================  ============================================================
  C                   if, else, for, while, goto, switch, break
  IR                  (icmp + (eq, ne, sgt, sge, slt, sle)0 + br
  DAG                 (seteq, setne, setgt, setge, setlt, setle) + brcond, 
  -                   (setueq, setune, setugt, setuge, setult, setule) + brcond
  cpu032I             CMP + (JEQ, JNE, JGT, JGE, JLT, JLE)
  cpu032II            (SLT, SLTu, SLTi, SLTiu) + (BEG, BNE)
  ==================  ============================================================



Long branch support
---------------------

As last section, cpu032II uses beq and bne to improve performance but the jump
offset reduces from 24 bits to 16 bits. If program exists more than 16 bits, 
cpu032II will fail to generate code. Mips backend has solution and Cpu0 hire 
the solution from it.

To support long branch the following code added in Chapter8_1.

.. rubric:: lbdex/chapters/Chapter8_2/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH8_2 //3
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0.h
    :start-after: #if CH >= CH8_2 //3
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/MCTargetDesc/Cpu0MCCodeEmitter.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: //@getJumpTargetOpValue {
    :end-before: #if CH >= CH8_1 //3
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: #elif CH >= CH8_2 //1
    :end-before: #else

.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0AsmPrinter.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.h
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0AsmPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: //@EmitInstruction {
    :end-before: //@EmitInstruction body {
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH8_2 //1
    :end-before: #else
	
.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH8_2 //2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0InstrInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.h
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH8_2 1
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@ long branch support //1
    :end-before: //#endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0LongBranch.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0LongBranch.cpp

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0MCInstLower.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.h
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0MCInstLower.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif
	
.. code-block:: c++

  void Cpu0MCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {

.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH8_2 //2
    :end-before: #endif
	
.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0SEInstrInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.h
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0SEInstrInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0TargetMachine.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH8_2 //2
    :end-before: //@8_2 1{
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: //@8_2 2}
    :end-before: #endif


The code of Chapter8_2 will compile the following example as follows,

.. rubric:: lbdex/input/ch8_2_longbranch.cpp
.. literalinclude:: ../lbdex/input/ch8_2_longbranch.cpp
    :start-after: /// start

.. code-block:: console

  118-165-78-10:input Jonathan$ ~/llvm/test/build/bin/llc 
  -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm 
  -force-cpu0-long-branch ch8_2_longbranch.bc -o -
    ...
	  .text
	  .section .mdebug.abiO32
	  .previous
	  .file	"ch8_2_longbranch.bc"
	  .globl	_Z15test_longbranchv
	  .align	2
	  .type	_Z15test_longbranchv,@function
	  .ent	_Z15test_longbranchv    # @_Z15test_longbranchv
  _Z15test_longbranchv:
	  .frame	$fp,16,$lr
	  .mask 	0x00001000,-4
	  .set	noreorder
	  .set	nomacro
  # BB#0:
	  addiu	$sp, $sp, -16
	  st	$fp, 12($sp)            # 4-byte Folded Spill
	  move	 $fp, $sp
	  addiu	$2, $zero, 1
	  st	$2, 8($fp)
	  addiu	$3, $zero, 2
	  st	$3, 4($fp)
	  addiu	$3, $zero, 0
	  st	$3, 0($fp)
	  ld	$3, 8($fp)
	  ld	$4, 4($fp)
	  slt	$3, $3, $4
	  bne	$3, $zero, .LBB0_3
	  nop
  # BB#1:
	  addiu	$sp, $sp, -8
	  st	$lr, 0($sp)
	  lui	$1, %hi(.LBB0_4-.LBB0_2)
	  addiu	$1, $1, %lo(.LBB0_4-.LBB0_2)
	  bal	.LBB0_2
  .LBB0_2:
	  addu	$1, $lr, $1
	  ld	$lr, 0($sp)
	  addiu	$sp, $sp, 8
	  jr	$1
	  nop
  .LBB0_3:
	  st	$2, 0($fp)
  .LBB0_4:
	  ld	$2, 0($fp)
	  move	 $sp, $fp
	  ld	$fp, 12($sp)            # 4-byte Folded Reload
	  addiu	$sp, $sp, 16
	  ret	$lr
	  nop
	  .set	macro
	  .set	reorder
	  .end	_Z15test_longbranchv
  $func_end0:
	  .size	_Z15test_longbranchv, ($func_end0)-_Z15test_longbranchv


Cpu0 backend Optimization: Remove useless JMP
---------------------------------------------

LLVM uses functional pass both in code generation and optimization. 
Following the 3 tiers of compiler architecture, LLVM do most optimization in 
middle tier of LLVM IR, SSA form. 
Beyond middle tier optimization, there are opportunities in 
optimization which depend on backend features. 
The "fill delay slot" in Mips is an example of backend optimization used in 
pipeline RISC machine.
You can port it from Mips if your backend is a pipeline RISC with 
delay slot. 
In this section, we apply the "delete useless jmp" in Cpu0 
backend optimization. 
This algorithm is simple and effective to be a perfect tutorial in optimization. 
Through this example, you can understand how to add an optimization pass and 
coding your complicated optimization algorithm on your backend in real project.

Chapter8_2/ supports "delete useless jmp" optimization algorithm which add 
codes as follows,

.. rubric:: lbdex/chapters/Chapter8_2/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH8_2 //2
    :end-before: #endif
  
.. rubric:: lbdex/chapters/Chapter8_2/Cpu0.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0.h
    :start-after: #if CH >= CH8_2 //2
    :end-before: #endif
  
.. rubric:: lbdex/chapters/Chapter8_2/Cpu0TargetMachine.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH8_2 //2
    :end-before: //@8_2 1{
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: //@8_2 1{
    :end-before: //@8_2 1}

.. code-block:: c++

  }

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0DelUselessJMP.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0DelUselessJMP.cpp


As above code, except Cpu0DelUselessJMP.cpp, other files are changed for 
registering class DelJmp as a functional pass. 
As the comment of above code, MBB is the current 
block and MBBN is the next block. For each last instruction of every MBB, we 
check whether or not it is the JMP instruction and its Operand is the next 
basic block. 
By getMBB() in MachineOperand, you can get the MBB address. 
For the member functions of MachineOperand, please check 
include/llvm/CodeGen/MachineOperand.h
Now, let's run Chapter8_2/ with ch8_2_deluselessjmp.cpp for explanation.

.. rubric:: lbdex/input/ch8_2_deluselessjmp.cpp
.. literalinclude:: ../lbdex/input/ch8_2_deluselessjmp.cpp
    :start-after: /// start

.. code-block:: console

  118-165-78-10:input Jonathan$ clang -target mips-unknown-linux-gnu 
  -c ch8_2_deluselessjmp.cpp -emit-llvm -o ch8_2_deluselessjmp.bc
  118-165-78-10:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=static -filetype=asm -stats 
  ch8_2_deluselessjmp.bc -o -
    ...
	  cmp	$sw, $4, $3
	  jne	$sw, $BB0_2
	  nop
  # BB#1:
    ...
	  cmp	$sw, $3, $2
	  jlt	$sw, $BB0_8
	  nop
  # BB#7:
  ...
  ===-------------------------------------------------------------------------===
                            ... Statistics Collected ...
  ===-------------------------------------------------------------------------===
   ...
   2 del-jmp        - Number of useless jmp deleted
   ...

The terminal displays "Number of useless jmp deleted" by ``llc -stats`` option 
because we set the "STATISTIC(NumDelJmp, "Number of useless jmp deleted")" in 
code. It deletes 2 jmp instructions from block "# BB#0" and "$BB0_6".
You can check it by ``llc -enable-cpu0-del-useless-jmp=false`` option to see 
the difference to non-optimization version.
If you run with ch8_1_1.cpp, you will find 10 jmp instructions are deleted from 
120 lines of assembly code, which meaning 8\% improvement in speed and code size 
[#cache-speed]_.


Fill Branch Delay Slot
-----------------------
 
Cpu0 instruction set is designed to be a classical RISC pipeline machine.
Classical RISC machine has many perfect features [#Quantitative]_ [#wiki-pipeline]_.
I change Cpu0 backend to a 5 stages of classical RISC pipeline machine with 
one delay slot like some of Mips model (The original Cpu0 from its author, is a
3 stages of RISC machine).
With this change, the backend needs
filling the NOP instruction in the branch delay slot.
In order to make this tutorial simple for learning, Cpu0 backend code not
fill the branch delay slot with any useful instruction for optimization.
Readers can reference the MipsDelaySlotFiller.cpp to know how to insert useful
instructions in backend optimization.
Following code added in Chapter8_2 for NOP fill in Branch Delay Slot.


.. rubric:: lbdex/chapters/Chapter8_2/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif
  
.. rubric:: lbdex/chapters/Chapter8_2/Cpu0.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0.h
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0TargetMachine.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH8_2 //2
    :end-before: //@8_2 1{
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: //@8_2 1}
    :end-before: //@8_2 2}

.. code-block:: c++

  }

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0DelaySlotFiller.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0DelaySlotFiller.cpp


To make the basic block label remains same, statement MIBundleBuilder() needs 
to be inserted after the statement BuildMI(..., NOP) of Cpu0DelaySlotFiller.cpp.
MIBundleBuilder() make both the branch instruction and NOP bundled into one
instruction (first part is branch instruction and second part is NOP).

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0AsmPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: //@EmitInstruction {
    :end-before: //@EmitInstruction body {
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: //@print out instruction:
    :end-before: #if CH >= CH9_1
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH8_2 //1
    :end-before: #else
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #endif //#if CH >= CH8_2 //1
    :end-before: //@EmitInstruction }

In order to print the NOP, the Cpu0AsmPrinter.cpp of Chapter3_2 prints all 
bundle instructions in loop.
Without the loop, only the first part of the bundle instruction 
(branch instruction only) is printed. 
In llvm 3.1 the basice block label remains same even if you didn't do the bundle
after it.
But for some reasons, it changed in llvm at some later version and you need doing 
"bundle" in order to keep block label unchanged at later llvm phase.


Conditional instruction
------------------------

.. rubric:: lbdex/input/ch8_2_select.cpp
.. literalinclude:: ../lbdex/input/ch8_2_select.cpp
    :start-after: /// start

Run Chapter8_1 with ch8_2_select.cpp will get the following result.

.. code-block:: console

  114-37-150-209:input Jonathan$ clang -O1 -target mips-unknown-linux-gnu 
  -c ch8_2_select.cpp -emit-llvm -o ch8_2_select.bc
  114-37-150-209:input Jonathan$ ~/llvm/test/build/bin/
  llvm-dis ch8_2_select.bc -o -
  ...
  ; Function Attrs: nounwind uwtable
  define i32 @_Z11test_movx_1v() #0 {
    %a = alloca i32, align 4
    %c = alloca i32, align 4
    store volatile i32 1, i32* %a, align 4
    store i32 0, i32* %c, align 4
    %1 = load volatile i32* %a, align 4
    %2 = icmp ne i32 %1, 0
    %3 = xor i1 %2, true
    %4 = select i1 %3, i32 1, i32 3
    store i32 %4, i32* %c, align 4
    %5 = load i32* %c, align 4
    ret i32 %5
  }

  ; Function Attrs: nounwind uwtable
  define i32 @_Z11test_movx_2v() #0 {
    %a = alloca i32, align 4
    %c = alloca i32, align 4
    store volatile i32 1, i32* %a, align 4
    store i32 0, i32* %c, align 4
    %1 = load volatile i32* %a, align 4
    %2 = icmp ne i32 %1, 0
    %3 = select i1 %2, i32 1, i32 3
    store i32 %3, i32* %c, align 4
    %4 = load i32* %c, align 4
    ret i32 %4
  }
  ...

  114-37-150-209:input Jonathan$ ~/llvm/test/build/bin/llc 
  -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm 
  ch8_2_select.bc -o -
  ...
  LLVM ERROR: Cannot select: 0x39f47c0: i32 = select_cc ...


As above llvm IR, ch8_2_select.bc, clang generates **select** IR for 
small basic control block (if statement only include one assign statement). 
This **select** IR is the result of optimization for CPUs with conditional 
instructions support. 
And from above error message, obviously IR **select** is changed to 
**select_cc** during DAG optimization stages.


Chapter8_2 supports **select** with the following code added and changed.

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH8_2 2
    :end-before: //#endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0CondMov.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0CondMov.td


.. rubric:: lbdex/chapters/Chapter8_2/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH8_2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter8_2/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH8_2 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH8_2 //2
    :end-before: #endif

.. code-block:: c++

  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #if CH >= CH8_1 //6
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH8_2 //3
    :end-before: #endif

.. code-block:: c++

    }
    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH8_2 //4
    :end-before: #endif


Set ISD::SELECT_CC to "Expand" will stop llvm optimization from merging "setcc" 
and "select" into one IR "select_cc" [#wb]_. 
Next the LowerOperation() return Op code directly for ISD::SELECT. 
Finally the pattern defined in Cpu0CondMov.td will 
translate the **select** IR into conditional instruction, **movz** or **movn**. 
Let's run Chapter8_2 with ch8_2_select.cpp to get the following result. 
Again, the cpu032II uses **slt** instead of **cmp** has a little improved in 
instructions number.

.. code-block:: console

  114-37-150-209:input Jonathan$ ~/llvm/test/build/bin/llc 
  -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm ch8_2_select.bc -o -
  ...
	.type	_Z11test_movx_1v,@function
	...
	addiu	$2, $zero, 3
	movz	$2, $3, $4
	...
	.type	_Z11test_movx_2v,@function
	...
	addiu	$2, $zero, 3
	movn	$2, $3, $4
	...


The clang uses **select** IR in small basic block 
to reduce the branch cost in pipeline machine since the branch will make the 
pipeline "stall". 
But it needs the conditional instruction support [#Quantitative]_. 
If your backend has no conditional instruction and needs clang compiler with 
optimization option **O1** above level, you can change clang to force it 
generating traditional branch basic block instead of generating IR **select**.
RISC CPU came from the advantage of pipeline and add more and more instruction 
when time passed. 
Compare Mips and ARM, the Mips has only **movz** and **movn** two 
instructions while ARM has many. We create Cpu0 instructions as a simple 
instructions RISC pipeline machine for compiler toolchain tutorial. 
However the **cmp** instruction is hired because many programmer is used to 
it in past and now (ARM use it). 
This instruction matches the thinking in assembly programming, 
but the **slt** instruction is more efficient in RISC pipleline.
If you designed a backend aimed for C/C++ highlevel language, you may consider 
**slt** instead of **cmp** since assembly code are rare used in programming and 
beside, the assembly programmer can accept **slt** not difficultly since usually 
they are professional.

File ch8_2_select2.cpp will generate IR **select** if compile with ``clang -O1``.

.. rubric:: lbdex/input/ch8_2_select2.cpp
.. literalinclude:: ../lbdex/input/ch8_2_select2.cpp
    :start-after: /// start

List the conditional statements of C, IR, DAG and Cpu0 instructions as the 
following table.

.. table:: Conditional statements of C, IR, DAG and Cpu0 instructions

  ==================  ============================================================
  C                   if (a < b) c = 1; else c = 3;
  -                   c = a ? 1:3;
  IR                  icmp + (eq, ne, sgt, sge, slt, sle) + br
  DAG                 ((seteq, setne, setgt, setge, setlt, setle) + setcc) + select
  Cpu0                movz, movn
  ==================  ============================================================


File ch8_2_select_global_pic.cpp mentioned in Chapter Global variables can be 
tested now as follows,

.. rubric:: lbdex/input/ch8_2_select_global_pic.cpp
.. literalinclude:: ../lbdex/input/ch8_2_select_global_pic.cpp
    :start-after: /// start

.. code-block:: console
  
  JonathantekiiMac:input Jonathan$ clang -O1 -target mips-unknown-linux-gnu 
  -c ch8_2_select_global_pic.cpp -emit-llvm -o ch8_2_select_global_pic.bc
  JonathantekiiMac:input Jonathan$ ~/llvm/test/build/bin/
  llvm-dis ch8_2_select_global_pic.bc -o -
  ...
  @a1 = global i32 1, align 4
  @b1 = global i32 2, align 4
  @gI1 = global i32 100, align 4
  @gJ1 = global i32 50, align 4
  
  ; Function Attrs: nounwind
  define i32 @_Z18test_select_globalv() #0 {
    %1 = load volatile i32* @a1, align 4, !tbaa !1
    %2 = load volatile i32* @b1, align 4, !tbaa !1
    %3 = icmp slt i32 %1, %2
    %gI1.val = load i32* @gI1, align 4
    %gJ1.val = load i32* @gJ1, align 4
    %.0 = select i1 %3, i32 %gI1.val, i32 %gJ1.val
    ret i32 %.0
  }
  ...
  JonathantekiiMac:input Jonathan$ ~/llvm/test/build/bin/
  llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm ch8_2_select_global_pic.bc -o -
    .section .mdebug.abi32
    .previous
    .file "ch8_2_select_global_pic.bc"
    .text
    .globl  _Z18test_select_globalv
    .align  2
    .type _Z18test_select_globalv,@function
    .ent  _Z18test_select_globalv # @_Z18test_select_globalv
  _Z18test_select_globalv:
    .frame  $sp,0,$lr
    .mask   0x00000000,0
    .set  noreorder
    .cpload $t9
    .set  nomacro
  # BB#0:
    lui $2, %got_hi(a1)
    addu  $2, $2, $gp
    ld  $2, %got_lo(a1)($2)
    ld  $2, 0($2)
    lui $3, %got_hi(b1)
    addu  $3, $3, $gp
    ld  $3, %got_lo(b1)($3)
    ld  $3, 0($3)
    cmp $sw, $2, $3
    andi  $2, $sw, 1
    lui $3, %got_hi(gJ1)
    addu  $3, $3, $gp
    ori $3, $3, %got_lo(gJ1)
    lui $4, %got_hi(gI1)
    addu  $4, $4, $gp
    ori $4, $4, %got_lo(gI1)
    movn  $3, $4, $2
    ld  $2, 0($3)
    ld  $2, 0($2)
    ret $lr
    .set  macro
    .set  reorder
    .end  _Z18test_select_globalv
  $tmp0:
    .size _Z18test_select_globalv, ($tmp0)-_Z18test_select_globalv
  
    .type a1,@object              # @a1
    .data
    .globl  a1
    .align  2
  a1:
    .4byte  1                       # 0x1
    .size a1, 4
  
    .type b1,@object              # @b1
    .globl  b1
    .align  2
  b1:
    .4byte  2                       # 0x2
    .size b1, 4
  
    .type gI1,@object             # @gI1
    .globl  gI1
    .align  2
  gI1:
    .4byte  100                     # 0x64
    .size gI1, 4
  
    .type gJ1,@object             # @gJ1
    .globl  gJ1
    .align  2
  gJ1:
    .4byte  50                      # 0x32
    .size gJ1, 4


Phi node
---------

Since phi node is popular used in SSA form [#ssa-wiki]_, llvm applies 
phi node in IR for optimization work either. 
Phi node exists for "live variable analysis". An example for C is here 
[#phi-ex]_. 
As mentioned in wiki web site of reference above, through finding dominance 
frontiers, compiler knows where to insert phi functions.
The following input let you know the benefits of phi node as follows,

.. rubric:: lbdex/input/ch8_2_phinode.cpp
.. literalinclude:: ../lbdex/input/ch8_2_phinode.cpp
    :start-after: /// start

Compile it with debug build clang for O3 as follows,

.. code-block:: console
  
  114-43-212-251:input Jonathan$ ~/llvm/release/build/bin/clang  
  -O3 -target mips-unknown-linux-gnu -c ch8_2_phinode.cpp -emit-llvm -o 
  ch8_2_phinode.bc
  114-43-212-251:input Jonathan$ ~/llvm/test/build/bin/llvm-dis 
  ch8_2_phinode.bc -o -
  ...
  define i32 @_Z12test_phinodeiii(i32 signext %a, i32 signext %b, i32 signext %c) local_unnamed_addr #0 {
  entry:
    %cmp = icmp eq i32 %a, 0
    br i1 %cmp, label %if.end7, label %if.else
  
  if.else:                                          ; preds = %entry
    %cmp1 = icmp eq i32 %b, 0
    br i1 %cmp1, label %if.else3, label %if.then2
  
  if.then2:                                         ; preds = %if.else
    %dec = add nsw i32 %a, -1
    br label %if.end7
  
  if.else3:                                         ; preds = %if.else
    %cmp4 = icmp eq i32 %c, 0
    %add = add nsw i32 %a, 2
    %add.a = select i1 %cmp4, i32 %add, i32 %a
    br label %if.end7
  
  if.end7:                                          ; preds = %entry, %if.else3, %if.then2
    %a.addr.0 = phi i32 [ %dec, %if.then2 ], [ %add.a, %if.else3 ], [ 1, %entry ]
    %add8 = add nsw i32 %a.addr.0, %b
    ret i32 %add8
  }

Because SSA form, the llvm ir for destination variable `a` in different basic 
block (if then, else) must use different name. But how does the source variable 
`a` in "d = a + b;" be named? The basic block "a = a-1;" and "a = a+2;" have different
names. The basic block "a = a-1;" uses %dec and the basic block "a = a+2;" uses "%add"
as destination variable name in SSA llvm ir.
In order to solve the source variable name from different basic blocks in SSA form,
the phi structure is created as above. The compiler option O0 as the following
doesn't apply phi node. Instead, it uses store to solve the source variable name
from different basic block.
  
.. code-block:: console
  
  114-43-212-251:input Jonathan$ ~/llvm/release/build/bin/clang 
  -O0 -target mips-unknown-linux-gnu -c ch8_2_phinode.cpp -emit-llvm -o 
  ch8_2_phinode.bc
  114-43-212-251:input Jonathan$ ~/llvm/test/build/bin/llvm-dis 
  ch8_2_phinode.bc -o -
  ...
  define i32 @_Z12test_phinodeiii(i32 signext %a, i32 signext %b, i32 signext %c) #0 {
  entry:
    %a.addr = alloca i32, align 4
    %b.addr = alloca i32, align 4
    %c.addr = alloca i32, align 4
    %d = alloca i32, align 4
    store i32 %a, i32* %a.addr, align 4
    store i32 %b, i32* %b.addr, align 4
    store i32 %c, i32* %c.addr, align 4
    store i32 2, i32* %d, align 4
    %0 = load i32, i32* %a.addr, align 4
    %cmp = icmp eq i32 %0, 0
    br i1 %cmp, label %if.then, label %if.else
  
  if.then:                                          ; preds = %entry
    %1 = load i32, i32* %a.addr, align 4
    %inc = add nsw i32 %1, 1
    store i32 %inc, i32* %a.addr, align 4
    br label %if.end7
  
  if.else:                                          ; preds = %entry
    %2 = load i32, i32* %b.addr, align 4
    %cmp1 = icmp ne i32 %2, 0
    br i1 %cmp1, label %if.then2, label %if.else3
  
  if.then2:                                         ; preds = %if.else
    %3 = load i32, i32* %a.addr, align 4
    %dec = add nsw i32 %3, -1
    store i32 %dec, i32* %a.addr, align 4
    br label %if.end6
  
  if.else3:                                         ; preds = %if.else
    %4 = load i32, i32* %c.addr, align 4
    %cmp4 = icmp eq i32 %4, 0
    br i1 %cmp4, label %if.then5, label %if.end
  
  if.then5:                                         ; preds = %if.else3
    %5 = load i32, i32* %a.addr, align 4
    %add = add nsw i32 %5, 2
    store i32 %add, i32* %a.addr, align 4
    br label %if.end
  
  if.end:                                           ; preds = %if.then5, %if.else3
    br label %if.end6
  
  if.end6:                                          ; preds = %if.end, %if.then2
    br label %if.end7
  
  if.end7:                                          ; preds = %if.end6, %if.then
    %6 = load i32, i32* %a.addr, align 4
    %7 = load i32, i32* %b.addr, align 4
    %add8 = add nsw i32 %6, %7
    store i32 %add8, i32* %d, align 4
    %8 = load i32, i32* %d, align 4
    ret i32 %8
  }

  
Compile with ``clang -O3`` generate phi function. The phi function can
assign virtual register value directly from multi basic blocks.
Compile with ``clang -O0`` doesn't generate phi, it assigns virtual register
value by loading stack slot where the stack slot is saved in each of multi 
basic blocks before. 
In this example the pointer of %a.addr point to the stack slot, and 
"store i32 %inc, i32* %a.addr, align 4", "store i32 %dec, i32* %a.addr, align 4", 
"store i32 %add, i32* %a.addr, align 4" in label if.then:, if.then2: and 
if.then5:, respectively. In other words, it needs 3 store instructions. 
It's possible that compiler finds that the a == 0 is always true after 
optimization analysis through phi node. 
If so, the phi node version will bring better
result because ``clang -O0`` version uses load and store with pointer %a.addr 
which may cut the optimization opportunity.
Compiler books discuss the Control Flow Graph (CFG) analysis through dominance 
frontiers calculation for setting phi node. Then compiler apply the global 
optimization on CFG with phi node, and remove phi node by replacing with
"load store" at the end.

If you are interested in more details than the wiki web site, please refer book
here [#phi-book]_ for phi node, or book here [#dominator-dragonbooks]_ for the 
dominator tree analysis if you have this book.


RISC CPU knowledge
-------------------

As mentioned in the previous section, Cpu0 is a RISC 
(Reduced Instruction Set 
Computer) CPU with 5 stages of pipeline. 
RISC CPU is full in the world, even the X86 of CISC (Complex Instruction Set 
Computer) is RISC inside (It translates CISC instruction into 
micro-instructions which do pipeline as RISC). 
Knowledge with RISC concept may make you satisfied in compiler design. 
List these two excellent books we have read for reference. 
Sure, there are many books in Computer Architecture and some of them contain 
real RISC CPU knowledge needed, but these two are excellent and popular.

Computer Organization and Design: The Hardware/Software Interface (The Morgan 
Kaufmann Series in Computer Architecture and Design)

Computer Architecture: A Quantitative Approach (The Morgan Kaufmann Series in 
Computer Architecture and Design) 

The book of “Computer Organization and Design: The Hardware/Software Interface” 
(there are 4 editions at the book is written) is for the introduction, while 
“Computer Architecture: A Quantitative Approach” is more complicate and deep 
in CPU architecture (there are 5 editions at the book is written). 

Above two books use Mips CPU as an example since Mips is more RISC-like than 
other market CPUs. 
ARM serials of CPU dominate the embedded market especially in mobile phone and 
other portable devices. The following book is good which I am reading now.

ARM System Developer's Guide: Designing and Optimizing System Software 
(The Morgan Kaufmann Series in Computer Architecture and Design).


.. _section Remove useless JMP:
  http://jonathan2251.github.io/lbd/ctrlflow.html#cpu0-backend-optimization-remove-useless-jmp


.. _section Conditional instruction:
  http://jonathan2251.github.io/lbd/ctrlflow.html#conditional-instruction


.. [#cache-speed] On a platform with cache and DRAM, the cache miss costs 
       serveral tens time of instruction cycle. 
       Usually, the compiler engineers who work in the vendor of platform 
       solution are spending much effort of trying to reduce the cache miss for 
       speed. Reduce code size will decrease the cache miss frequency too.

.. [#wb] http://llvm.org/docs/WritingAnLLVMBackend.html#expand

.. [#Quantitative] See book Computer Architecture: A Quantitative Approach (The 
       Morgan Kaufmann Series in Computer Architecture and Design) 

.. [#wiki-pipeline] http://en.wikipedia.org/wiki/Classic_RISC_pipeline

.. [#ssa-wiki] https://en.wikipedia.org/wiki/Static_single_assignment_form

.. [#phi-ex] http://stackoverflow.com/questions/11485531/what-exactly-phi-instruction-does-and-how-to-use-it-in-llvm

.. [#phi-book] Section 8.11 of Muchnick, Steven S. (1997). Advanced Compiler Design and Implementation. Morgan Kaufmann. ISBN 1-55860-320-4.

.. [#dominator-dragonbooks] Refer chapter 9 of book Compilers: Principles, 
    Techniques, and Tools (2nd Edition) 
