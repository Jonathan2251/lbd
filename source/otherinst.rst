.. _sec-addingmoresupport:

Arithmetic and logic instructions
=================================

.. contents::
   :local:
   :depth: 4

This chapter adds more Cpu0 arithmetic instructions support first.
The `section Display llvm IR nodes with Graphviz`_ 
will show you the steps of DAG optimization and their corresponding ``llc`` 
display options. 
These DAGs translation in some steps of optimization can be displayed by the 
graphic tool of Graphviz which supply useful information with graphic view. 
Logic instructions support will come after arithmetic section.
In spite of that llvm backend handle the IR only, we get the IR from the 
corresponding C operators with designed C example code. 
Through compiling with C code, readers can know exactly what C statements are
handled by each chapter's appending code.
Instead of focusing on classes relationship in this backend structure of last
chapter, readers should focus on the mapping of C operators and llvm IR and 
how to define the mapping relationship of IR and instructions in td. 
HILO and C0 register class are defined in this chapter. 
Readers will know how to handle other register classes beside general 
purpose register class, and why they are needed, from this chapter.

Arithmetic
-----------

The code added in Chapter4_1/ to support arithmetic instructions as follows,

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0Subtarget.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.cpp
    :start-after: #if CH >= CH4_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.cpp
    :start-after: //@1 {
    :end-before: //@1 }
	
.. code-block:: c++
  
    ...
  
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.cpp
    :start-after: #if CH >= CH4_1 //2
    :end-before: #endif
	
.. code-block:: c++
   
    ...
  }

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_1 1
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_1 2
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_1 3
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_1 4
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_1 5
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_1 6
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_1 7
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_1 8
    :end-before: //#endif

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH4_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH4_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH4_1 //2
    :end-before: #endif

.. code-block:: c++

    ...
  }
  ...

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH4_1 //3
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0RegisterInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.td
    :start-after: //@ All registers definition
    :end-before: //@ General Purpose Registers
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.td
    :start-after: //#if CH >= CH4_1 1
    :end-before: //#endif
	
.. code-block:: c++

  }
  ...

.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.td
    :start-after: //#if CH >= CH4_1 2
    :end-before: //#endif

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0Schedule.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0Schedule.td
    :start-after: //#if CH >= CH4_1 1
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0Schedule.td
    :start-after: //@ http://llvm.org/docs/doxygen/html/structllvm_1_1InstrStage.html
    :end-before: //@2
.. literalinclude:: ../lbdex/Cpu0/Cpu0Schedule.td
    :start-after: //#if CH >= CH4_1 2
    :end-before: //#endif

.. code-block:: c++

  ]>;

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0SEISelDAGToDAG.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelDAGToDAG.h
    :start-after: #if CH >= CH4_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0SEISelDAGToDAG.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelDAGToDAG.cpp
    :start-after: #if CH >= CH4_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelDAGToDAG.cpp
    :start-after: //@selectNode
    :end-before: #if CH >= CH7_1 //2
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelDAGToDAG.cpp
    :start-after: #if CH >= CH4_1 //2
    :end-before: #endif
	
.. code-block:: c++

      }
      ...
    }

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0SEInstrInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.h
    :start-after: #if CH >= CH4_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0SEInstrInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: #if CH >= CH4_1
    :end-before: #endif


**+, -, \*, <<,** and **>>**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ADDu, ADD, SUBu, SUB and MUL defined in Chapter4_1/Cpu0InstrInfo.td are for
operators **+, -, \***.
SHL (defined before) and SHLV are for **<<**.
SRA, SRAV, SHR and SHRV are for **>>**.

In RISC CPU like Mips, the multiply/divide function unit and add/sub/logic unit 
are designed from two different hardware circuits, and more, their data path are 
separate. Cpu0 is same, so these two function units can be executed at same 
time (instruction level parallelism). Reference [#instrstage]_ for instruction itineraries.

Chapter4_1/ can handle **+, -, \*, <<,** and **>>** operators in C 
language. 
The corresponding llvm IR instructions are **add, sub, mul, shl, ashr**. 
The **'ashr'** instruction (arithmetic shift right) returns the first operand 
shifted to the right a specified number of bits with sign extension. 
In brief, we call **ashr** is “shift with sign extension fill”.

.. note:: **ashr**

  Example:
    <result> = ashr i32 4, 1   ; yields {i32}:result = 2
    
    <result> = ashr i8 -2, 1   ; yields {i8}:result = -1
    
    <result> = ashr i32 1, 32  ; undefined

The semantic of C operator **>>** for negative operand is dependent on 
implementation. 
Most compilers translate it into “shift with sign extension fill”, and 
Mips **sra** is this instruction. 
Following is the Micosoft web site's explanation,

.. note:: **>>**, Microsoft Specific

  The result of a right shift of a signed negative quantity is implementation 
  dependent. 
  Although Microsoft C++ propagates the most-significant bit to fill vacated 
  bit positions, there is no guarantee that other implementations will do 
  likewise.

In addition to **ashr**, the other instruction “shift with zero filled” 
**lshr** in llvm (Mips implement lshr with instruction **srl**) has the 
following meaning. 

.. note:: **lshr**

  Example:
  <result> = lshr i8 -2, 1   ; yields {i8}:result = 0x7FFFFFFF 
  
In llvm, IR node **sra** is defined for ashr IR instruction, and node **srl** is 
defined for lshr instruction (We don't know why it doesn't use ashr and lshr as 
the IR node name directly). Summary as the Table: C operator >> implementation.


.. table:: C operator >> implementation

  ======================================= ======================  =====================================
  Description                             Shift with zero filled  Shift with signed extension filled
  ======================================= ======================  =====================================
  symbol in .bc                           lshr                    ashr
  symbol in IR node                       srl                     sra
  Mips instruction                        srl                     sra
  Cpu0 instruction                        shr                     sra
  signed example before x >> 1            0xfffffffe i.e. -2      0xfffffffe i.e. -2
  signed example after x >> 1             0x7fffffff i.e 2G-1     0xffffffff i.e. -1
  unsigned example before x >> 1          0xfffffffe i.e. 4G-2    0xfffffffe i.e. 4G-2
  unsigned example after x >> 1           0x7fffffff i.e 2G-1     0xffffffff i.e. 4G-1
  ======================================= ======================  =====================================
  
**lshr:** Logical SHift Right

**ashr:** Arithmetic SHift right

**srl:**  Shift Right Logically

**sra:**  Shift Right Arithmetically

**shr:**  SHift Right


If we consider the x >> 1 definition is x = x/2 for compiler implementation.
Then as you can see from Table: C operator >> implementation, **lshr** will fail 
on some signed value (such as -2). In the same way, **ashr** will fail on some 
unsigned value (such as 4G-2). So, in order to satisfy this definition in 
both signed and unsigned integers of x, we need these two instructions, 
**lshr** and **ashr**.

.. table:: C operator << implementation

  ======================================= ======================
  Description                             Shift with zero filled
  ======================================= ======================
  symbol in .bc                           shl
  symbol in IR node                       shl
  Mips instruction                        sll
  Cpu0 instruction                        shl
  signed example before x << 1            0x40000000 i.e. 1G
  signed example after x << 1             0x80000000 i.e -2G
  unsigned example before x << 1          0x40000000 i.e. 1G
  unsigned example after x << 1           0x80000000 i.e 2G
  ======================================= ======================

Again, consider the x << 1 definition is x = x*2. 
From Table: C operator << implementation, we see **lshr** satisfy "unsigned 
x=1G" but fails on signed x=1G. 
It's fine since 2G is out of 32 bits signed integer range (-2G ~ 2G-1). 
For the overflow case, no way to keep the correct result in register. So, any 
value in register is OK. You can check that **lshr** satisfy x = x*2, for all 
x << 1 and the x result is not out of range, no matter operand x is signed 
or unsigned integer.

Micorsoft implementation references here [#msdn]_.

The ‘ashr‘ Instruction" reference here [#ashr]_, ‘lshr‘ reference here [#lshr]_.

The srav, shlv and shrv are for two virtual input registers instructions while 
the sra, ... are for 1 virtual input registers and 1 constant input operands.

Now, let's build Chapter4_1/ and run with input file ch4_math.ll as follows,

.. rubric:: lbdex/input/ch4_math.ll
.. literalinclude:: ../lbdex/input/ch4_math.ll
  
.. code-block:: bash

  118-165-78-12:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_math.ll -o -
    ...
	  ld	$2, 0($sp)
	  ld	$3, 4($sp)
	  subu	$4, $3, $2
	  addu	$5, $3, $2
	  addu	$4, $5, $4
	  mul	$5, $3, $2
	  addu	$4, $4, $5
	  shl	$5, $3, 2
	  addu	$4, $4, $5
	  sra	$5, $3, 2
	  addu	$4, $4, $5
	  addiu	$5, $zero, 128
	  shrv	$5, $5, $2
	  addiu	$t9, $zero, 1
	  shlv	$t9, $t9, $2
	  srav	$2, $3, $2
	  shr	$3, $3, 30
	  addu	$3, $4, $3
	  addu	$3, $3, $t9
	  addu	$3, $3, $5
	  addu	$2, $3, $2
	  addiu	$sp, $sp, 8
	  ret	$lr


Example input ch4_1_math.cpp as the following is the C file which include **+, -, 
\*, <<,** and **>>** operators. 
It will generate corresponding llvm IR instructions, 
**add, sub, mul, shl, ashr** by clang as Chapter 3 indicated.

.. rubric:: lbdex/input/ch4_1_math.cpp
.. literalinclude:: ../lbdex/input/ch4_1_math.cpp
    :start-after: /// start

    
Cpu0 instructions add and sub will trigger overflow exception while addu and subu
truncate overflow value directly. Compile ch4_1_addsuboverflow.cpp with 
``llc -cpu0-enable-overflow=true`` will generate add and sub instructions as 
follows,

.. rubric:: lbdex/input/ch4_1_addsuboverflow.cpp
.. literalinclude:: ../lbdex/input/ch4_1_addsuboverflow.cpp
    :start-after: /// start

.. code-block:: bash

  118-165-78-12:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch4_1_addsuboverflow.cpp -emit-llvm -o ch4_1_addsuboverflow.bc
  118-165-78-12:input Jonathan$ llvm-dis ch4_1_addsuboverflow.bc -o -
  ...
  ; Function Attrs: nounwind
  define i32 @_Z13test_overflowv() #0 {
    ...
    %3 = add nsw i32 %1, %2
    ...
    %6 = sub nsw i32 %4, %5
    ...
  }

  118-165-78-12:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm 
  -cpu0-enable-overflow=true ch4_1_addsuboverflow.bc -o -
	...
	add	$3, $4, $3
	...
	sub	$3, $4, $3
	...

In modern CPU, programmers are used to using truncate overflow instructions for
C operators + and -. 
Anyway, through option -cpu0-enable-overflow=true, programmer get the
chance to compile program with overflow exception program. Usually, this option
used in debug purpose. Compile with this option can help to identify the bug and
fix it early.


Display llvm IR nodes with Graphviz
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous section, display the DAG translation process in text on terminal 
by option ``llc -debug``. 
The ``llc`` also supports the graphic displaying. 
The `section Install other tools on iMac`_ include the download and installation
of tool Graphivz. 
The ``llc`` graphic displaying with tool Graphviz is introduced in this section. 
The graphic displaying is more readable by eyes than displaying text in terminal. 
It's not a must-have, but helps a lot especially when you are tired in tracking 
the DAG translation process. 
List the ``llc`` graphic support options from the sub-section "SelectionDAG 
Instruction Selection Process" of web "The LLVM Target-Independent Code Generator" 
[#instructionsel]_ as follows,

.. note:: The ``llc`` Graphviz DAG display options

  -view-dag-combine1-dags displays the DAG after being built, before the 
  first optimization pass. 
  
  -view-legalize-dags displays the DAG before Legalization. 
  
  -view-dag-combine2-dags displays the DAG before the second optimization 
  pass. 
  
  -view-isel-dags displays the DAG before the Select phase. 
  
  -view-sched-dags displays the DAG before Scheduling. 
  
By tracking ``llc -debug``, you can see the steps of DAG translation as follows,

.. code-block:: bash

  Initial selection DAG
  Optimized lowered selection DAG
  Type-legalized selection DAG
  Optimized type-legalized selection DAG
  Legalized selection DAG
  Optimized legalized selection DAG
  Instruction selection
  Selected selection DAG
  Scheduling
  ...


Let's run ``llc`` with option -view-dag-combine1-dags, and open the output 
result with Graphviz as follows,

.. code-block:: bash

  118-165-12-177:input Jonathan$ /Users/Jonathan/llvm/test/
  cmake_debug_build/Debug/bin/llc -view-dag-combine1-dags -march=cpu0 
  -relocation-model=pic -filetype=asm ch4_1_mult.bc -o ch4_1_mult.cpu0.s
  Writing '/tmp/llvm_84ibpm/dag.main.dot'...  done. 
  118-165-12-177:input Jonathan$ Graphviz /tmp/llvm_84ibpm/dag.main.dot 

It will show the /tmp/llvm_84ibpm/dag.main.dot as :num:`Figure #otherinst-f1`.

.. _otherinst-f1:
.. figure:: ../Fig/otherinst/1.png
  :height: 851 px
  :width: 687 px
  :scale: 100 %
  :align: center

  llc option -view-dag-combine1-dags graphic view
  
:num:`Figure #otherinst-f1` is the stage of "Initial selection DAG". 
List the other view options and their corresponding stages of DAG translation as 
follows,

.. note:: ``llc`` Graphviz options and the corresponding stages of DAG translation

  -view-dag-combine1-dags: Initial selection DAG
  
  -view-legalize-dags: Optimized type-legalized selection DAG
  
  -view-dag-combine2-dags: Legalized selection DAG
  
  -view-isel-dags: Optimized legalized selection DAG
  
  -view-sched-dags: Selected selection DAG

The -view-isel-dags is important and often used by an llvm backend writer 
because it is the DAGs before instruction selection. 
In order to writing the pattern match instruction in target description file 
.td, backend programmer needs knowing what the DAG nodes are for a given C 
operator.


Operator % and /
~~~~~~~~~~~~~~~~~~

The DAG of %
+++++++++++++++

Example input code ch4_1_mult.cpp which contains the C operator **“%”** and it's 
corresponding llvm IR, as follows,

.. rubric:: lbdex/input/ch4_1_mult.cpp
.. literalinclude:: ../lbdex/input/ch4_1_mult.cpp
    :start-after: /// start

.. code-block:: bash

  ...
  define i32 @_Z8test_multv() #0 {
    %b = alloca i32, align 4
    store i32 11, i32* %b, align 4
    %1 = load i32* %b, align 4
    %2 = add nsw i32 %1, 1
    %3 = srem i32 %2, 12
    store i32 %3, i32* %b, align 4
    %4 = load i32* %b, align 4
    ret i32 %4
  }

LLVM **srem** is the IR of corresponding **“%”**, reference here [#srem]_. 
Copy the reference as follows,

.. note:: **'srem'** Instruction 

  Syntax:
  **<result> = srem <ty> <op1>, <op2>   ; yields {ty}:result**
    
  Overview:
  The **'srem'** instruction returns the remainder from the signed division of its 
  two operands. This instruction can also take vector versions of the values in 
  which case the elements must be integers.
  
  Arguments:
  The two arguments to the **'srem'** instruction must be integer or vector of 
  integer values. Both arguments must have identical types.
  
  Semantics:
  This instruction returns the remainder of a division (where the result is 
  either zero or has the same sign as the dividend, op1), not the modulo operator 
  (where the result is either zero or has the same sign as the divisor, op2) of 
  a value. For more information about the difference, see The Math Forum. For a 
  table of how this is implemented in various languages, please see Wikipedia: 
  modulo operation.
  
  Note that signed integer remainder and unsigned integer remainder are distinct 
  operations; for unsigned integer remainder, use **'urem'**.
  
  Taking the remainder of a division by zero leads to undefined behavior. 
  Overflow also leads to undefined behavior; this is a rare case, but can occur, 
  for example, by taking the remainder of a 32-bit division of -2147483648 by -1. 
  (The remainder doesn't actually overflow, but this rule lets srem be 
  implemented using instructions that return both the result of the division and 
  the remainder.)
  
  Example:
  <result> = **srem i32 4, %var**      ; yields {i32}:result = 4 % %var


Run Chapter3_5/ with input file ch4_1_mult.bc via option ``llc –view-isel-dags``, 
will get the following error message and the llvm DAGs of 
:num:`Figure #otherinst-f2` below.

.. code-block:: bash

  118-165-79-37:input Jonathan$ /Users/Jonathan/llvm/test/
  cmake_debug_build/Debug/bin/llc -march=cpu0 -view-isel-dags -relocation-model=
  pic -filetype=asm ch4_1_mult.bc -o -
  ...
  LLVM ERROR: Cannot select: 0x7fa73a02ea10: i32 = mulhs 0x7fa73a02c610, 
  0x7fa73a02e910 [ID=12]
  0x7fa73a02c610: i32 = Constant<12> [ORD=5] [ID=7]
  0x7fa73a02e910: i32 = Constant<715827883> [ID=9]


.. _otherinst-f2:
.. figure:: ../Fig/otherinst/2.png
  :height: 629 px
  :width: 580 px
  :scale: 100 %
  :align: center

  ch4_1_mult.bc DAG

LLVM replaces srem divide operation with multiply operation in DAG optimization 
because DIV operation costs more in time than MUL. 
Example code **“int b = 11; b=(b+1)%12;”** is translated into DAGs as
:num:`Figure #otherinst-f2`. 
The DAGs of generated result is verified and explained by calculating the value 
in each node. 
The 0xC*0x2AAAAAAB=0x2,00000004, (mulhs 0xC, 0x2AAAAAAAB) meaning get the Signed 
mul high word (32bits). 
Multiply with 2 operands of 1 word size probably generate the 2 word size of 
result (0x2, 0xAAAAAAAB). 
The result of high word, in this case is 0x2. 
The final result (sub 12, 12) is 0 which match the statement (11+1)%12.

 
Arm solution
+++++++++++++

To run with ARM solution, change Cpu0InstrInfo.td and Cpu0ISelDAGToDAG.cpp from 
Chapter4_1/ as follows,

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0InstrInfo.td
.. code-block:: c++

  /// Multiply and Divide Instructions.
  def SMMUL   : ArithLogicR<0x41, "smmul", mulhs, IIImul, CPURegs, 1>;
  def UMMUL   : ArithLogicR<0x42, "ummul", mulhu, IIImul, CPURegs, 1>;
  //def MULT    : Mult32<0x41, "mult", IIImul>;
  //def MULTu   : Mult32<0x42, "multu", IIImul>;

.. rubric:: lbdex/chapters/Chapter4_1/Cpu0ISelDAGToDAG.cpp
.. code-block:: c++

  #if 0
  /// Select multiply instructions.
  std::pair<SDNode*, SDNode*>
  Cpu0DAGToDAGISel::SelectMULT(SDNode *N, unsigned Opc, SDLoc DL, EVT Ty,
                               bool HasLo, bool HasHi) {
    SDNode *Lo = 0, *Hi = 0;
    SDNode *Mul = CurDAG->getMachineNode(Opc, DL, MVT::Glue, N->getOperand(0),
                                         N->getOperand(1));
    SDValue InFlag = SDValue(Mul, 0);

    if (HasLo) {
      Lo = CurDAG->getMachineNode(Cpu0::MFLO, DL,
                                  Ty, MVT::Glue, InFlag);
      InFlag = SDValue(Lo, 1);
    }
    if (HasHi)
      Hi = CurDAG->getMachineNode(Cpu0::MFHI, DL,
                                  Ty, InFlag);

    return std::make_pair(Lo, Hi);
  }
  #endif

  /// Select instructions not customized! Used for
  /// expanded, promoted and normal instructions
  SDNode* Cpu0DAGToDAGISel::Select(SDNode *Node) {
  ...
    switch(Opcode) {
    default: break;
  #if 0
    case ISD::MULHS:
    case ISD::MULHU: {
      MultOpc = (Opcode == ISD::MULHU ? Cpu0::MULTu : Cpu0::MULT);
      return SelectMULT(Node, MultOpc, DL, NodeTy, false, true).second;
    }
  #endif
   ...
  }


Let's run above changes with ch4_1_mult.cpp as well as ``llc -view-sched-dags`` option 
to get :num:`Figure #otherinst-f3`. 
Instruction SMMUL will get the high word of multiply result.

.. _otherinst-f3:
.. figure:: ../Fig/otherinst/3.png
  :height: 743 px
  :width: 684 px
  :scale: 100 %
  :align: center

  DAG for ch4_1_mult.bc with ARM style SMMUL

The following is the result of run above changes with ch4_1_mult.bc.

.. code-block:: bash

  118-165-66-82:input Jonathan$ /Users/Jonathan/llvm/test/cmake_
  debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm 
  ch4_1_mult.bc -o -
    ...
  # BB#0:                                 # %entry
    addiu $sp, $sp, -8
  $tmp1:
    .cfi_def_cfa_offset 8
    addiu $2, $zero, 0
    st  $2, 4($fp)
    addiu $2, $zero, 11
    st  $2, 0($fp)
    lui $2, 10922
    ori $3, $2, 43691
    addiu $2, $zero, 12
    smmul $3, $2, $3
    shr $4, $3, 31
    sra $3, $3, 1
    addu  $3, $3, $4
    mul $3, $3, $2
    subu  $2, $2, $3
    st  $2, 0($fp)
    addiu $sp, $sp, 8
    ret $lr


The other instruction UMMUL and llvm IR mulhu are unsigned int type for 
operator %. 
You can check it by unmark the **“unsigned int b = 11;”** in ch4_1_mult.cpp.

Using SMMUL instruction to get the high word of multiplication result is adopted 
in ARM. 


Mips solution
++++++++++++++

Mips uses MULT instruction and save the high & low part to registers HI and LO,
respectively. 
After that, uses mfhi/mflo to move register HI/LO to your general purpose 
registers. 
ARM SMMUL is fast if you only need the HI part of result (it ignores the LO part 
of operation). ARM also provides SMULL (signed multiply long) to get the whole 
64 bits result.
If you need the LO part of result, you can use Cpu0 MUL instruction to get the 
LO part of result only. 
Chapter4_1/ is implemented with Mips MULT style. 
We choose it as the implementation of this book for adding instructions as less 
as possible. This approach make Cpu0 better both as a tutorial architecture 
for school teaching purpose material, and an engineer learning 
materials in compiler design.
The MULT, MULTu, MFHI, MFLO, MTHI, MTLO added in Chapter4_1/Cpu0InstrInfo.td; 
HI, LO registers in Chapter4_1/Cpu0RegisterInfo.td and Chapter4_1/MCTargetDesc/
Cpu0BaseInfo.h; IIHiLo, IIImul in Chapter4_1/Cpu0Schedule.td; SelectMULT() in
Chapter4_1/Cpu0ISelDAGToDAG.cpp are for Mips style implementation.

The related DAG nodes, mulhs and mulhu, both are used in Chapter4_1/, which
come from TargetSelectionDAG.td as follows,
  
.. rubric:: include/llvm/Target/TargetSelectionDAG.td
.. code-block:: c++

  def mulhs    : SDNode<"ISD::MULHS"     , SDTIntBinOp, [SDNPCommutative]>;
  def mulhu    : SDNode<"ISD::MULHU"     , SDTIntBinOp, [SDNPCommutative]>;

  
Except the custom type, llvm IR operations of type expand and promote will call 
Cpu0DAGToDAGISel::Select() during instruction selection of DAG translation. 
The SelectMULT() which called by Select() return the HI part of 
multiplication result to HI register for IR operations of mulhs or mulhu. 
After that, MFHI instruction moves the HI register to Cpu0 field "a" register, 
$ra. 
MFHI instruction is FL format and only use Cpu0 field "a" register, we set 
the $rb and imm16 to 0. 
:num:`Figure #otherinst-f4` and ch4_1_mult.cpu0.s are the results of compile 
ch4_1_mult.bc.

.. _otherinst-f4:
.. figure:: ../Fig/otherinst/4.png
  :height: 837 px
  :width: 554 px
  :scale: 90 %
  :align: center

  DAG for ch4_1_mult.bc with Mips style MULT

.. code-block:: bash

  118-165-66-82:input Jonathan$ cat ch4_1_mult.cpu0.s 
    ...
  # BB#0:
    addiu $sp, $sp, -8
    addiu $2, $zero, 11
    st  $2, 4($sp)
    lui $2, 10922
    ori $3, $2, 43691
    addiu $2, $zero, 12
    mult  $2, $3
    mfhi  $3
    shr $4, $3, 31
    sra $3, $3, 1
    addu  $3, $3, $4
    mul $3, $3, $2
    subu  $2, $2, $3
    st  $2, 4($sp)
    addiu $sp, $sp, 8
    ret $lr
    

Full support \%, and /
++++++++++++++++++++++

The sensitive readers may find llvm using **“multiplication”** instead 
of **“div”** to get the **“\%”** result just because our example uses 
constant as divider, **“(b+1)\%12”** in our example. 
If programmer uses variable as the divider like **“(b+1)\%a”**, then: what will 
happen next? 
The answer is our code will has error in handling this. 

Cpu0 just like Mips uses LO and HI registers to hold the **"quotient"** and 
**"remainder"**. And 
uses instructions **“mflo”** and **“mfhi”** to get the result from LO or HI 
registers furthermore. 
With this solution, the **“c = a / b”** can be finished by **“div a, b”** and 
**“mflo c”**; the **“c = a \% b”** can be finished by **“div a, b”** and 
**“mfhi c”**.
 
To supports operators **“\%”** and **“/”**, the following code added in 
Chapter4_1.

1. SDIV, UDIV and it's reference class, nodes in Cpu0InstrInfo.td.

2. The copyPhysReg() declared and defined in Cpu0InstrInfo.h and 
   Cpu0InstrInfo.cpp.

3. The setOperationAction(ISD::SDIV, MVT::i32, Expand), ..., 
   setTargetDAGCombine(ISD::SDIVREM) in constructore of Cpu0ISelLowering.cpp;  
   PerformDivRemCombine() and PerformDAGCombine() in Cpu0ISelLowering.cpp.


The IR instruction **sdiv** stands for signed div while **udiv** stands for 
unsigned div.

.. rubric:: lbdex/input/ch4_1_mult2.cpp
.. literalinclude:: ../lbdex/input/ch4_1_mult2.cpp
    :start-after: /// start

If we run with ch4_1_mult2.cpp, the **“div”** cannot be gotten for operator 
**“%”**. 
It still uses **"multiplication"** instead of **"div"** in ch4_1_mult2.cpp because 
llvm do **“Constant Propagation Optimization”** in this. 
The ch4_1_mod.cpp can get the **“div”** for **“%”** result since it makes 
llvm **“Constant Propagation Optimization”** useless in it. 
  
.. rubric:: lbdex/input/ch4_1_mod.cpp
.. literalinclude:: ../lbdex/input/ch4_1_mod.cpp
    :start-after: /// start

.. code-block:: bash

  118-165-77-79:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch4_1_mod.cpp -emit-llvm -o ch4_1_mod.bc
  118-165-77-79:input Jonathan$ /Users/Jonathan/llvm/test/cmake_
  debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm 
  ch4_1_mod.bc -o -
  ...
  div $zero, $3, $2
  mflo  $2
  ...

To explains how to work with **“div”**, let's run ch4_1_mod.cpp with debug option
as follows,

.. code-block:: bash

  118-165-83-58:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch4_1_mod.cpp -I/Applications/Xcode.app/Contents/Developer/Platforms/
  MacOSX.platform/Developer/SDKs/MacOSX10.8.sdk/usr/include/ -emit-llvm -o 
  ch4_1_mod.bc
  118-165-83-58:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm -debug 
  ch4_1_mod.bc -o -
  ...
  === _Z8test_modi
  Initial selection DAG: BB#0 '_Z8test_mod2i:'
  SelectionDAG has 21 nodes:
    ...
      0x2447448: <multiple use>
          0x24470d0: <multiple use>
          0x24471f8: i32 = Constant<1>

        0x2447320: i32 = add 0x24470d0, 0x24471f8 [ORD=7]

        0x2447448: <multiple use>
      0x2447570: i32 = srem 0x2447320, 0x2447448 [ORD=9]

      0x24468b8: <multiple use>
      0x2446b08: <multiple use>
    0x2448fc0: ch = store 0x2447448:1, 0x2447570, 0x24468b8, ...

    0x2449210: i32 = Register %V0

      0x2448fc0: <multiple use>
      0x2449210: <multiple use>
        0x2448fc0: <multiple use>
        0x24468b8: <multiple use>
        0x2446b08: <multiple use>
      0x24490e8: i32,ch = load 0x2448fc0, 0x24468b8, 0x2446b08<LD4[%b]> [ORD=11]

    0x2449338: ch,glue = CopyToReg 0x2448fc0, 0x2449210, 0x24490e8 [ORD=12]

      0x2449338: <multiple use>
      0x2449210: <multiple use>
      0x2449338: <multiple use>
    0x2449460: ch = Cpu0ISD::Ret 0x2449338, 0x2449210, 0x2449338:1 [ORD=12]

  Replacing.1 0x24490e8: i32,ch = load 0x2448fc0, 0x24468b8, ...

  With: 0x2447570: i32 = srem 0x2447320, 0x2447448 [ORD=9]
   and 1 other values
  ...

  Optimized lowered selection DAG: BB#0 '_Z8test_mod2i:'
  ...
    0x2447570: i32 = srem 0x2447320, 0x2447448 [ORD=9]
  ...
  
  Type-legalized selection DAG: BB#0 '_Z8test_mod2i:'
  SelectionDAG has 16 nodes:
    ...
    0x7fed6882d610: i32,ch = load 0x7fed6882d210, 0x7fed6882cd10, 
    0x7fed6882cb10<LD4[%1]> [ORD=5] [ID=-3]
  
      0x7fed6882d810: i32 = Constant<12> [ID=-3]
  
      0x7fed6882d610: <multiple use>
    0x7fed6882d710: i32 = srem 0x7fed6882d810, 0x7fed6882d610 [ORD=6] [ID=-3]
    ...
    
  Legalized selection DAG: BB#0 '_Z8test_mod2i:'
    ...
      ... i32 = srem 0x2447320, 0x2447448 [ORD=9] [ID=-3]
    ...
   ... replacing: ...: i32 = srem 0x2447320, 0x2447448 [ORD=9] [ID=13]
       with:      ...: i32,i32 = sdivrem 0x2447320, 0x2447448 [ORD=9]
  
  Optimized legalized selection DAG: BB#0 '_Z8test_mod2i:'
  SelectionDAG has 18 nodes:
    ...
      0x2449588: i32 = Register %HI

          0x24470d0: <multiple use>
          0x24471f8: i32 = Constant<1> [ID=6]

        0x2447320: i32 = add 0x24470d0, 0x24471f8 [ORD=7] [ID=12]

        0x2447448: <multiple use>
      0x24490e8: glue = Cpu0ISD::DivRem 0x2447320, 0x2447448 [ORD=9]

    0x24496b0: i32,ch,glue = CopyFromReg 0x240d480, 0x2449588, 0x24490e8 [ORD=9]

      0x2449338: <multiple use>
      0x2449210: <multiple use>
      0x2449338: <multiple use>
    0x2449460: ch = Cpu0ISD::Ret 0x2449338, 0x2449210, ...
    ...
  
  ===== Instruction selection begins: BB#0 ''
  ...
  Selecting: 0x24490e8: glue = Cpu0ISD::DivRem 0x2447320, 0x2447448 [ORD=9] [ID=14]

  ISEL: Starting pattern match on root node: 0x24490e8: glue = Cpu0ISD::DivRem 
  0x2447320, 0x2447448 [ORD=9] [ID=14]

    Initial Opcode index to 4044
    Morphed node: 0x24490e8: i32,glue = SDIV 0x2447320, 0x2447448 [ORD=9]

  ISEL: Match complete!
  => 0x24490e8: i32,glue = SDIV 0x2447320, 0x2447448 [ORD=9]
  ...


Summary above DAGs translation messages into 4 steps:

1. Reduce DAG nodes in stage "Optimized lowered selection DAG" (Replacing ... 
   displayed before "Optimized lowered selection DAG:"). 
   Since SSA form has some redundant nodes for store and load, they can be 
   removed.

2. Change DAG srem to sdivrem in stage "Legalized selection DAG".

3. Change DAG sdivrem to Cpu0ISD::DivRem and in stage "Optimized legalized 
   selection DAG".

4. Add DAG "i32 = Register %HI" and "CopyFromReg ..." in stage "Optimized 
   legalized selection DAG".

Summary as Table: Stages for C operator % and Table: Functions handle the DAG 
translation and pattern match for C operator %.

.. table:: Stages for C operator %

  ==================================  ==============================================
  Stage                               IR/DAG/instruction
  ==================================  ==============================================
  .bc                                 srem        
  Legalized selection DAG             sdivrem       
  Optimized legalized selection DAG   Cpu0ISD::DivRem, CopyFromReg xx, Hi, Cpu0ISD::DivRem
  pattern match                       div, mfhi
  ==================================  ==============================================


.. table:: Functions handle the DAG translation and pattern match for C operator %

  ====================================  ============================
  Translation                           Do by
  ====================================  ============================
  srem => sdivrem                       setOperationAction(ISD::SREM, MVT::i32, Expand);
  sdivrem => Cpu0ISD::DivRem            setTargetDAGCombine(ISD::SDIVREM);
  sdivrem => CopyFromReg xx, Hi, xx     PerformDivRemCombine();
  Cpu0ISD::DivRem => div                SDIV (Cpu0InstrInfo.td)
  CopyFromReg xx, Hi, xx => mfhi        MFLO (Cpu0InstrInfo.td)
  ====================================  ============================


Step 2 as above, is triggered by code 
"setOperationAction(ISD::SREM, MVT::i32, Expand);" in Cpu0ISelLowering.cpp. 
About **Expand** please ref. [#expand]_ and [#legalizetypes]_. Step 3 is 
triggered by code "setTargetDAGCombine(ISD::SDIVREM);" in Cpu0ISelLowering.cpp.
Step 4 is did by PerformDivRemCombine() which called by performDAGCombine().
Since the **%** corresponding **srem** makes the "N->hasAnyUseOfValue(1)" to 
true in PerformDivRemCombine(), it creates DAG of "CopyFromReg". 
When using **"/"** in C, it will make "N->hasAnyUseOfValue(0)" to ture.
For sdivrem, **sdiv** makes "N->hasAnyUseOfValue(0)" true while **srem** makes 
"N->hasAnyUseOfValue(1)" ture.

Above steps will change the DAGs when ``llc`` is running. After that, the pattern 
match defined in Chapter4_1/Cpu0InstrInfo.td will translate **Cpu0ISD::DivRem** 
into **div**; and **"CopyFromReg xxDAG, Register %H, Cpu0ISD::DivRem"** 
to **mfhi**.

The ch4_1_div.cpp is for **/** div operator test.


Rotate instructions
~~~~~~~~~~~~~~~~~~~~

Chapter4_1 include the rotate operations translation. The instructions "rol", 
"ror", "rolv" and "rorv" defined in Cpu0InstrInfo.td handle the translation.
Compile ch4_1_rotate.cpp will get Cpu0 "rol" instruction.

.. rubric:: lbdex/input/ch4_1_rotate.cpp
.. literalinclude:: ../lbdex/input/ch4_1_rotate.cpp
    :start-after: /// start

.. code-block:: bash
  
  114-43-200-122:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch4_1_rotate.cpp -emit-llvm -o ch4_1_rotate.bc
  114-43-200-122:input Jonathan$ llvm-dis ch4_1_rotate.bc -o -
  
  define i32 @_Z16test_rotate_leftv() #0 {
    %a = alloca i32, align 4
    %result = alloca i32, align 4
    store i32 8, i32* %a, align 4
    %1 = load i32* %a, align 4
    %2 = shl i32 %1, 30
    %3 = load i32* %a, align 4
    %4 = ashr i32 %3, 2
    %5 = or i32 %2, %4
    store i32 %5, i32* %result, align 4
    %6 = load i32* %result, align 4
    ret i32 %6
  }
  
  114-43-200-122:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/Debug/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch4_1_rotate.bc -o -
    ...
    rol $2, $2, 30
    ...

Instructions "rolv" and "rorv" cannot be tested at this moment, they need 
logic "or" implementation which supported at next section. 
Like the previous subsection mentioned at this 
chapter, some IRs in function @_Z16test_rotate_leftv() will be combined into 
one one IR **rotl** during DAGs translation.


Logic
-------

Chapter4_2 supports logic operators **&, |, ^, !, ==, !=, <, <=, > and >=**.
They are trivial and easy. Listing the added code with comments and table for 
these operators IR, DAG and instructions as below. Please check them with the
run result of bc and asm instructions for ch4_2_logic.cpp as below.

.. rubric:: lbdex/chapters/Chapter4_2/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_2 1
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_2 2
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_2 3
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_2 4
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_2 5
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_2 6
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_2 7
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_2 8
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH4_2 9
    :end-before: //#endif

.. rubric:: lbdex/chapters/Chapter4_2/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH4_2 //1.2
    :end-before: #endif

.. code-block:: c++

    ...
  }

.. rubric:: lbdex/input/ch4_2_logic.cpp
.. literalinclude:: ../lbdex/input/ch4_2_logic.cpp
    :start-after: /// start

.. code-block:: bash

  114-43-204-152:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch4_2_logic.cpp -emit-llvm -o ch4_2_logic.bc
  114-43-204-152:input Jonathan$ llvm-dis ch4_2_logic.bc -o -
  ...
  ; Function Attrs: nounwind uwtable
  define i32 @_Z16test_andorxornotv() #0 {
  entry:
    ...
    %and = and i32 %0, %1
    ...
    %or = or i32 %2, %3
    ...
    %xor = xor i32 %4, %5
    ...
    %tobool = icmp ne i32 %6, 0
    %lnot = xor i1 %tobool, true
    %conv = zext i1 %lnot to i32
    ...
  }

  ; Function Attrs: nounwind uwtable
  define i32 @_Z10test_setxxv() #0 {
  entry:
    ...
    %cmp = icmp eq i32 %0, %1
    %conv = zext i1 %cmp to i32
    store i32 %conv, i32* %c, align 4
    ...
    %cmp1 = icmp ne i32 %2, %3
    %conv2 = zext i1 %cmp1 to i32
    store i32 %conv2, i32* %d, align 4
    ...
    %cmp3 = icmp slt i32 %4, %5
    %conv4 = zext i1 %cmp3 to i32
    store i32 %conv4, i32* %e, align 4
    ...
    %cmp5 = icmp sle i32 %6, %7
    %conv6 = zext i1 %cmp5 to i32
    store i32 %conv6, i32* %f, align 4
    ...
    %cmp7 = icmp sgt i32 %8, %9
    %conv8 = zext i1 %cmp7 to i32
    store i32 %conv8, i32* %g, align 4
    ...
    %cmp9 = icmp sge i32 %10, %11
    %conv10 = zext i1 %cmp9 to i32
    store i32 %conv10, i32* %h, align 4
    ...
  }
  
  114-43-204-152:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -filetype=asm 
  ch4_2_logic.bc -o -

    .globl  _Z16test_andorxornotv
    ...
    and $3, $4, $3
    ...
    or  $3, $4, $3
    ...
    xor $3, $4, $3
    ...
    cmp $sw, $3, $2
    andi  $2, $sw, 2
    shr $2, $2, 1
    ...

    .globl  _Z10test_setxxv
    ...
    cmp $sw, $3, $2
    andi  $2, $sw, 2
    shr $2, $2, 1
    ...
    cmp $sw, $3, $2
    andi  $2, $sw, 2
    shr $2, $2, 1
    xori  $2, $2, 1
    ...
    cmp $sw, $3, $2
    andi  $2, $sw, 1
    ...
    cmp $sw, $3, $2
    andi  $2, $sw, 1
    xori  $2, $2, 1
    ...
    cmp $sw, $3, $2
    andi  $2, $sw, 1
    ...
    cmp $sw, $3, $2
    andi  $2, $sw, 1
    xori  $2, $2, 1
    ...

  114-43-204-152:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic -filetype=asm 
  ch4_2_logic.bc -o -
    ...
	sltiu	$2, $2, 1
	andi	$2, $2, 1
	...

.. table:: Logic operators for cpu032I

  ==========  =================================  ====================================  =======================
  C           .bc                                Optimized legalized selection DAG     cpu032I
  ==========  =================================  ====================================  =======================
  &, &&       and                                and                                   and
  \|, \|\|    or                                 or                                    or
  ^           xor                                xor                                   xor
  !           - %tobool = icmp ne i32 %6, 0      - %lnot = (setcc %tobool, 0, seteq)   - xor $3, $4, $3
              - %lnot = xor i1 %tobool, true     - %conv = (and %lnot, 1)
              - %conv = zext i1 %lnot to i32     - 
  ==          - %cmp = icmp eq i32 %0, %1        - %cmp = (setcc %0, %1, seteq)        - cmp $sw, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - andi  $2, $sw, 2
                                                                                       - shr $2, $2, 1
                                                                                       - andi $2, $2, 1
  !=          - %cmp = icmp ne i32 %0, %1        - %cmp = (setcc %0, %1, setne)        - cmp $sw, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - andi  $2, $sw, 2
                                                                                       - shr $2, $2, 1
                                                                                       - andi $2, $2, 1
  <           - %cmp = icmp lt i32 %0, %1        - (setcc %0, %1, setlt)               - cmp $sw, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - andi  $2, $sw, 2
                                                                                       - andi $2, $2, 1
                                                                                       - andi $2, $2, 1
  <=          - %cmp = icmp le i32 %0, %1        - (setcc %0, %1, setle)               - cmp $sw, $2, $3
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - andi  $2, $sw, 1
                                                                                       - xori  $2, $2, 1
                                                                                       - andi $2, $2, 1
  >           - %cmp = icmp gt i32 %0, %1        - (setcc %0, %1, setgt)               - cmp $sw, $2, $3
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - andi  $2, $sw, 2
                                                                                       - andi $2, $2, 1
  >=          - %cmp = icmp le i32 %0, %1        - (setcc %0, %1, setle)               - cmp $sw, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - andi  $2, $sw, 1
                                                                                       - xori  $2, $2, 1
                                                                                       - andi $2, $2, 1
  ==========  =================================  ====================================  =======================

.. table:: Logic operators for cpu032II

  ==========  =================================  ====================================  =======================
  C           .bc                                Optimized legalized selection DAG     cpu032II
  ==========  =================================  ====================================  =======================
  &, &&       and                                and                                   and
  \|, \|\|    or                                 or                                    or
  ^           xor                                xor                                   xor
  !           - %tobool = icmp ne i32 %6, 0      - %lnot = (setcc %tobool, 0, seteq)   - xor $3, $4, $3
              - %lnot = xor i1 %tobool, true     - %conv = (and %lnot, 1)
              - %conv = zext i1 %lnot to i32     - 
  ==          - %cmp = icmp eq i32 %0, %1        - %cmp = (setcc %0, %1, seteq)        - xor $2, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - sltiu  $2, $2, 1
                                                                                       - andi $2, $2, 1
  !=          - %cmp = icmp ne i32 %0, %1        - %cmp = (setcc %0, %1, setne)        - xor $2, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - sltu  $2, $zero, 2
                                                                                       - shr $2, $2, 1
                                                                                       - andi $2, $2, 1
  <           - %cmp = icmp lt i32 %0, %1        - (setcc %0, %1, setlt)               - slt $2, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - andi  $2, $2, 1
  <=          - %cmp = icmp le i32 %0, %1        - (setcc %0, %1, setle)               - slt $2, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - xori  $2, $2, 1
                                                                                       - andi $2, $2, 1
  >           - %cmp = icmp gt i32 %0, %1        - (setcc %0, %1, setgt)               - slt $2, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - andi  $2, $2, 1
  >=          - %cmp = icmp le i32 %0, %1        - (setcc %0, %1, setle)               - slt $2, $3, $2
              - %conv = zext i1 %cmp to i32      - and %cmp, 1                         - xori  $2, $2, 1
                                                                                       - andi $2, $2, 1
  ==========  =================================  ====================================  =======================

In relation operators ==, !=, ..., %0 = $3 = 5, %1 = $2 = 3 for ch4_2_logic.cpp.

The "Optimized legalized selection DAG" is the last DAG stage just before the 
"instruction selection" as the previous section mentioned in this chapter. 
You can see the whole DAG stages by ``llc -debug`` option.

From above result, slt spend less instructions than cmp for relation 
operators translation. Beyond that, slt uses general purpose register while 
cmp uses $sw dedicated register.

.. rubric:: lbdex/input/ch4_2_slt_explain.cpp
.. literalinclude:: ../lbdex/input/ch4_2_slt_explain.cpp
    :start-after: /// start

.. code-block:: bash

  118-165-78-10:input Jonathan$ clang -target mips-unknown-linux-gnu -O2 
  -c ch4_2_slt_explain.cpp -emit-llvm -o ch4_2_slt_explain.bc
  118-165-78-10:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -mcpu=cpu032I -relocation-model=static -filetype=asm 
  ch4_2_slt_explain.bc -o -
    ...
    ld  $3, 20($sp)
    cmp $sw, $3, $2
    andi  $2, $sw, 1
    andi  $2, $2, 1
    st  $2, 12($sp)
    addiu $2, $zero, 2
    ld  $3, 16($sp)
    cmp $sw, $3, $2
    andi  $2, $sw, 1
    andi  $2, $2, 1
    ...
  118-165-78-10:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -mcpu=cpu032II -relocation-model=static -filetype=asm 
  ch4_2_slt_explain.bc -o -
    ...
    ld  $2, 20($sp)
    slti  $2, $2, 1
    andi  $2, $2, 1
    st  $2, 12($sp)
    ld  $2, 16($sp)
    slti  $2, $2, 2
    andi  $2, $2, 1
    st  $2, 8($sp)
    ...

Run these two `llc -mcpu` option for Chapter4_2 with ch4_2_slt_explain.cpp to get the 
above result. Regardless of the move between \$sw and general purpose register 
in `llc -mcpu=cpu032I`, the two cmp instructions in it will has hazard in 
instruction reorder since both of them use \$sw register. The  
`llc -mcpu=cpu032II` has not this problem because it uses slti [#Quantitative]_. 
The slti version can reorder as follows,

.. code-block:: bash

    ...
    ld  $2, 16($sp)
    slti  $2, $2, 2
    andi  $2, $2, 1
    st  $2, 8($sp)
    ld  $2, 20($sp)
    slti  $2, $2, 1
    andi  $2, $2, 1
    st  $2, 12($sp)
    ...

Chapter4_2 include instructions cmp and slt. Though cpu032II include both of 
these two instructions, the slt takes the priority since 
"let Predicates = [HasSlt]" appeared before "let Predicates = [HasCmp]" in 
Cpu0InstrInfo.td.


Summary
--------

List C operators, IR of .bc, Optimized legalized selection DAG and Cpu0 
instructions implemented in this chapter in Table: Chapter 4 mathmetic 
operators. There are over 20 operators totally in mathmetic and logic support in
this chapter and spend 4xx lines of source code. 

.. table:: Chapter 4 mathmetic operators

  ================  =================================  ====================================  ==========
  C                 .bc                                Optimized legalized selection DAG     Cpu0
  ================  =================================  ====================================  ==========
  \+                add                                add                                   addu
  \-                sub                                sub                                   subu
  \*                mul                                mul                                   mul
  /                 sdiv                               Cpu0ISD::DivRem                       div
  -                 udiv                               Cpu0ISD::DivRemU                      divu
  <<                shl                                shl                                   shl
  >>                - ashr                             - sra                                 - sra
                    - lshr                             - srl                                 - shr
  !                 - %tobool = icmp ne i32 %0, 0      - %lnot = (setcc %tobool, 0, seteq)   - %1 = (xor %tobool, 0)
                    - %lnot = xor i1 %tobool, true     - %conv = (and %lnot, 1)              - %true = (addiu $r0, 1)
                                                                                             - %lnot = (xor %1, %true)
  -                 - %conv = zext i1 %lnot to i32     - %conv = (and %lnot, 1)              - %conv = (and %lnot, 1)
  %                 - srem                             - Cpu0ISD::DivRem                     - div
                    - sremu                            - Cpu0ISD::DivRemU                    - divu
  (x<<n)|(x>>32-n)  shl + lshr                         rotl, rotr                            rol, rolv, ror, rorv 
  ================  =================================  ====================================  ==========



.. _section Display llvm IR nodes with Graphviz:
  http://jonathan2251.github.io/lbd/otherinst.html#display-llvm-ir-nodes-
  with-graphviz

.. _section Install other tools on iMac:
  http://jonathan2251.github.io/lbd/install.html#install-other-tools-on-imac


.. [#instrstage] http://llvm.org/docs/doxygen/html/structllvm_1_1InstrStage.html

.. [#msdn] http://msdn.microsoft.com/en-us/library/336xbhcz%28v=vs.80%29.aspx

.. [#ashr] http://llvm.org/docs/LangRef.html#ashr-instruction

.. [#lshr] http://llvm.org/docs/LangRef.html#lshr-instruction

.. [#instructionsel] http://llvm.org/docs/CodeGenerator.html#selectiondag-instruction-selection-process

.. [#srem] http://llvm.org/docs/LangRef.html#srem-instruction

.. [#expand] http://llvm.org/docs/WritingAnLLVMBackend.html#expand

.. [#legalizetypes] http://llvm.org/docs/CodeGenerator.html#selectiondag-legalizetypes-phase

.. [#Quantitative] See book Computer Architecture: A Quantitative Approach (The Morgan 
       Kaufmann Series in Computer Architecture and Design) 


