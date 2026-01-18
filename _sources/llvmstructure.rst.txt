.. _sec-llvmstructure:

Cpu0 Architecture and LLVM Structure
=====================================

.. contents::
   :local:
   :depth: 4

Before you begin this tutorial, you should know that you can always try to 
develop your own backend by porting code from existing backends. 
The majority of the code you will want to investigate can be found in the 
/lib/Target directory of your root LLVM installation. 
As most major RISC instruction sets have some similarities, this may be the 
avenue you might try if you are an experienced programmer and knowledgable of 
compiler backends.

On the other hand, there is a steep learning curve and you may easily get stuck 
debugging your new backend. You can easily spend a lot of time tracing which 
methods are callbacks of some function, or which are calling some overridden 
method deep in the LLVM codebase - and with a codebase as large as LLVM, all of 
this can easily become difficult to keep track of. 
This tutorial will help you work through this process while learning the 
fundamentals of LLVM backend design. 
It will show you what is necessary to get your first backend functional and 
complete, and it should help you understand how to debug your backend when it 
produces incorrect machine code using output provided by the compiler.

This chapter details the Cpu0 instruction set and the structure of LLVM. 
The LLVM structure information is adapted from Chris Lattner's LLVM chapter of 
the Architecture of Open Source Applications book [#aosa-book]_. You can read 
the original article from the AOSA website if you prefer. 

At the end of this Chapter, you will begin to create a new LLVM backend by 
writing register and instruction definitions in the Target Description files 
which will be used in next chapter.

Finally, there are compiler knowledge like DAG (Directed-Acyclic-Graph) and 
instruction selection needed in llvm backend design, and they are explained 
here. 


Cpu0 Processor Architecture Details
-----------------------------------

This section is based on materials available here [#cpu0-chinese]_ (Chinese)
and here [#cpu0-english]_ (English). However, I changed some ISA from original
Cpu0 for designing a simple integer operational CPU and llvm backend. This is
my intention for writing this book that I want to know what a simple and robotic
CPU ISA and llvm backend can be.

Brief introduction
******************

Cpu0 is a 32-bit architecture. It has 16 general purpose registers (R0, ..., 
R15), co-processor registers (like Mips), and other special registers. Its 
structure is illustrated in :numref:`llvmstructure-f1` below.

.. _llvmstructure-f1: 
.. figure:: ../Fig/llvmstructure/1.png
  :width: 608 px
  :height: 360 px
  :align: center

  Architectural block diagram of the Cpu0 processor


The registers are used for the following purposes:

.. table:: Cpu0 general purpose registers (GPR)

  ============  ===========
  Register      Description
  ============  ===========
  R0            Constant register, value is 0
  R1-R10        General-purpose registers
  R11           Global Pointer register (GP)
  R12           Frame Pointer register (FP)
  R13           Stack Pointer register (SP)
  R14           Link Register (LR)
  R15           Status Word Register (SW)
  ============  ===========

.. table:: Cpu0 co-processor 0 registers (C0R)

  ============  ===========
  Register      Description
  ============  ===========
  0             Program Counter (PC)
  1             Error Program Counter (EPC)
  ============  ===========

.. table:: Cpu0 other registers

  ============  ===========
  Register      Description
  ============  ===========
  IR            Instruction register
  MAR           Memory Address Register (MAR)
  MDR           Memory Data Register (MDR)
  HI            High part of MULT result
  LO            Low part of MULT result
  ============  ===========

The Cpu0 Instruction Set
************************

The Cpu0 instruction set is categorized into three types:  

- **L-type instructions**: Primarily used for memory operations.  
- **A-type instructions**: Designed for arithmetic operations.  
- **J-type instructions**: Typically used for altering control flow (e.g., jumps).  

:numref:`llvmstructure-f2` illustrates the bitfield breakdown for each  
instruction type.

.. _llvmstructure-f2: 
.. figure:: ../Fig/llvmstructure/2.png
  :scale: 50%
  :align: center

  Cpu0's three instruction formats

.. raw:: latex

   \clearpage
      
.. table:: C, llvm-ir [#langref]_ and Cpu0

  ====================  =================================  ====================================  =======  ================
  C                     llvm-ir                            Cpu0                                  I or II  Comment
  ====================  =================================  ====================================  =======  ================
  =                     load/store                         ld/lb/lbu/lh/lhu                      I
  &, &&                 and                                and                                   I
  \|, \|\|              or                                 or                                    I
  ^                     xor                                xor/nor                               I        ! can be got from two ir
  !                     - %tobool = icmp ne i32 %6, 0      - cmp                                 
                        - %lnot = xor i1 %tobool, true     - xor                                 
  ==, !=, <, <=, >, >=  icmp/fcmp <cond> cond:eq/ne,...    cmp/ucmp ... + floating-lib           I
   "                      "                                slt/sltu/slti/sltiu                   II       slti/sltiu: ex. a == 3 reduce instructions
  if (a <= b)           icmp/fcmp <cond> +                 cmp/uccmp + jeq/jne/jlt/jgt/jle/jge   I        Conditional branch
                        br i1 <cond>, ...
  if (bool)             br i1 <cond>, ...                  jeq/jne                               I
    "                     "                                beq/bne                               II
  goto                  br <dest>                          jmp                                   I        Uncondictional branch
  call sub-function     call                               jsub                                  I        Provide 24-bit address range of calling sub-function (the address from caller to callee is within 24-bit)
    "                     "                                jalr                                  I        Add for 32-bit address range of calling sub-function
  return                ret                                ret                                   I
  +, -, \*              add/fadd, sub/fsub, mul/fmul       add/addu/addiu, sub/subu, mul         I
  /, %                  udiv/sdiv/fdiv, urem/srem/frem     div, mfhi/mflo/mthi/mtlo              I
  <<, >>                shl, lshr/ashr                     shl/rol/rolv, srl/sra/ror/rorv        II
  float <-> int         fptoui, sitofp, ...                                                               Cpu0 uses SW for floating value, and these two IR are for HW floating instruction 
  __builtin_clz/clo     llvm.clz/llvm_clo                  floating-lib + clz, clo               I        For SW floating-lib, uses __builtin_clz / __builtin_clo in clang and clang generates llvm.clz/llvm.clo intrinsic function
  __builtin_eh_xxx      llvm.eh.xxx                        st/ld                                 I        pass information to exception handler through $4, $5
  ====================  =================================  ====================================  =======  ================

.. table:: C++, llvm-ir [#langref]_ and Cpu0

  ====================  =================================  ====================================  =======  ================
  C++                   llvm-ir                            Cpu0                                  I or II  Comment
  ====================  =================================  ====================================  =======  ================
  try {  }              invoke void @_Z15throw_exception   jsub  _Z15throw_exception             I
  catch { }             landingpad...catch                 st and ld                             I        st/ld $4 & $5 to/from stack, $4:exception address, $5: exception typeid
  ====================  =================================  ====================================  =======  ================

.. note:: **Selection of LLVM-IR and the ISA for a RISC CPU**

  - LLVM-IR and the ISA of a RISC CPU emerged after the C language.  
    As shown in the table above, they can be selected based on C language  
    constructs.  

  - Not listed in the table, LLVM-IR includes terminator instructions such as  
    `switch`, `invoke`, and others, as well as atomic operations and a variety  
    of LLVM intrinsics. These intrinsics provide better performance for backend  
    implementations, such as `llvm.vector.reduce.*`.  

  - For vector processing on CPUs/GPUs, vector-type math LLVM-IR or  
    LLVM intrinsics can be used for implementation.  

.. note:: **Selection of the ISA for Cpu0**

  - The original author of Cpu0 designed its ISA as a teaching material,  
    without focusing on performance.  

  - My goal is to refine the ISA selection and design, considering both its  
    role as an LLVM tutorial and its basic performance as an ISA. I am not  
    interested in a poorly designed ISA.  

    - As shown in the table above, `"if (a <= b)"` can be rewritten as  
      `"t = (a <= b)"` followed by `"if (t)"`.  
      Thus, I designed **ISA II of Cpu0** to use `"slt + beq"` instead of  
      `"cmp + jeq"`, reducing six conditional jump instructions  
      (`jeq/jne/jlt/jgt/jle/jge`) to just two (`beq/bne`).  
      This balances complexity and performance in the Cpu0 ISA.  

    - For the same reason, I adopted **slt** from **MIPS** instead of **cmp**  
      from **ARM**. This allows the destination register to be any general-  
      purpose register (GPR), avoiding bottlenecks caused by a shared  
      "status register."  

    - Floating-point operations can be implemented in software, so Cpu0  
      only supports integer instructions. I added **clz** (count leading zeros)  
      and **clo** (count leading ones) to Cpu0 since floating-point libraries,  
      such as `compiler-rt/builtin`, rely on these built-in functions.  
      Floating-point normalization can leverage **clz** and **clo** for  
      performance improvements. Although Cpu0 could use multiple instructions  
      to implement `llvm.clz` and `llvm.clo`, having dedicated **clz/clo**  
      instructions allows execution in a single instruction.  

    - I extended **ISA II of Cpu0** for better performance, following the  
      principles of MIPS.

The following table provides details on the cpu032I instruction set:

- First column F\.: meaning Format.

.. list-table:: cpu032I Instruction Set
  :widths: 1 4 3 11 7 10
  :header-rows: 1

  * - F\.
    - Mnemonic
    - Opcode
    - Meaning
    - Syntax
    - Operation
  * - L
    - NOP
    - 00
    - No Operation
    - 
    - 
  * - L
    - LD
    - 01
    - Load word
    - LD Ra, [Rb+Cx]
    - Ra <= [Rb+Cx]
  * - L
    - ST
    - 02
    - Store word
    - ST Ra, [Rb+Cx]
    - [Rb+Cx] <= Ra
  * - L
    - LB
    - 03
    - Load byte
    - LB Ra, [Rb+Cx]
    - Ra <= (byte)[Rb+Cx] [#lb-note]_
  * - L
    - LBu
    - 04
    - Load byte unsigned
    - LBu Ra, [Rb+Cx]
    - Ra <= (byte)[Rb+Cx] [#lb-note]_
  * - L
    - SB
    - 05
    - Store byte
    - SB Ra, [Rb+Cx]
    - [Rb+Cx] <= (byte)Ra
  * - L
    - LH
    - 06
    - Load half word
    - LH Ra, [Rb+Cx]
    - Ra <= (2bytes)[Rb+Cx] [#lb-note]_
  * - L
    - LHu
    - 07
    - Load half word unsigned
    - LHu Ra, [Rb+Cx]
    - Ra <= (2bytes)[Rb+Cx] [#lb-note]_
  * - L
    - SH
    - 08
    - Store half word
    - SH Ra, [Rb+Cx]
    - [Rb+Cx] <= Ra
  * - L
    - ADDiu
    - 09
    - Add immediate
    - ADDiu Ra, Rb, Cx
    - Ra <= (Rb + Cx)
  * - L
    - ANDi
    - 0C
    - AND imm
    - ANDi Ra, Rb, Cx
    - Ra <= (Rb & Cx)
  * - L
    - ORi
    - 0D
    - OR
    - ORi Ra, Rb, Cx
    - Ra <= (Rb | Cx)
  * - L
    - XORi
    - 0E
    - XOR
    - XORi Ra, Rb, Cx
    - Ra <= (Rb \^ Cx)
  * - L
    - LUi
    - 0F
    - Load upper
    - LUi Ra, Cx
    - Ra <= (Cx << 16)
  * - A
    - ADDu
    - 11
    - Add unsigned
    - ADD Ra, Rb, Rc
    - Ra <= Rb + Rc [#u-note]_
  * - A
    - SUBu
    - 12
    - Sub unsigned
    - SUB Ra, Rb, Rc
    - Ra <= Rb - Rc [#u-note]_
  * - A
    - ADD
    - 13
    - Add
    - ADD Ra, Rb, Rc
    - Ra <= Rb + Rc [#u-note]_
  * - A
    - SUB
    - 14
    - Subtract
    - SUB Ra, Rb, Rc
    - Ra <= Rb - Rc [#u-note]_
  * - A
    - CLZ
    - 15
    - Count Leading Zero
    - CLZ Ra, Rb
    - Ra <= bits of leading zero on Rb
  * - A
    - CLO
    - 16
    - Count Leading One
    - CLO Ra, Rb
    - Ra <= bits of leading one on Rb
  * - A
    - MUL
    - 17
    - Multiply
    - MUL Ra, Rb, Rc
    - Ra <= Rb * Rc
  * - A
    - AND
    - 18
    - Bitwise and
    - AND Ra, Rb, Rc
    - Ra <= Rb & Rc
  * - A
    - OR
    - 19
    - Bitwise or
    - OR Ra, Rb, Rc
    - Ra <= Rb | Rc
  * - A
    - XOR
    - 1A
    - Bitwise exclusive or
    - XOR Ra, Rb, Rc
    - Ra <= Rb ^ Rc
  * - A
    - NOR
    - 1B
    - Bitwise boolean nor
    - NOR Ra, Rb, Rc
    - Ra <= Rb nor Rc
  * - A
    - ROL
    - 1C
    - Rotate left
    - ROL Ra, Rb, Cx
    - Ra <= Rb rol Cx
  * - A
    - ROR
    - 1D
    - Rotate right
    - ROR Ra, Rb, Cx
    - Ra <= Rb ror Cx
  * - A
    - SHL
    - 1E
    - Shift left
    - SHL Ra, Rb, Cx
    - Ra <= Rb << Cx
  * - A
    - SHR
    - 1F
    - Shift right
    - SHR Ra, Rb, Cx
    - Ra <= Rb >> Cx
  * - A
    - SRA
    - 20
    - Shift right
    - SRA Ra, Rb, Cx
    - Ra <= Rb '>> Cx [#sra-note]_
  * - A
    - SRAV
    - 21
    - Shift right
    - SRAV Ra, Rb, Rc
    - Ra <= Rb '>> Rc [#sra-note]_
  * - A
    - SHLV
    - 22
    - Shift left
    - SHLV Ra, Rb, Rc
    - Ra <= Rb << Rc
  * - A
    - SHRV
    - 23
    - Shift right
    - SHRV Ra, Rb, Rc
    - Ra <= Rb >> Rc
  * - A
    - ROL
    - 24
    - Rotate left
    - ROL Ra, Rb, Rc
    - Ra <= Rb rol Rc
  * - A
    - ROR
    - 25
    - Rotate right
    - ROR Ra, Rb, Rc
    - Ra <= Rb ror Rc
  * - A
    - CMP
    - 2A
    - Compare
    - CMP Ra, Rb
    - SW <= (Ra cond Rb) [#cond-note]_
  * - A
    - CMPu
    - 2B
    - Compare
    - CMPu Ra, Rb
    - SW <= (Ra cond Rb) [#cond-note]_
  * - J
    - JEQ
    - 30
    - Jump if equal (==)
    - JEQ Cx
    - if SW(==), PC <= PC + Cx
  * - J
    - JNE
    - 31
    - Jump if not equal (!=)
    - JNE Cx
    - if SW(!=), PC <= PC + Cx
  * - J
    - JLT
    - 32
    - Jump if less than (<)
    - JLT Cx
    - if SW(<), PC <= PC + Cx
  * - J
    - JGT
    - 33
    - Jump if greater than (>)
    - JGT Cx
    - if SW(>), PC <= PC + Cx
  * - J
    - JLE
    - 34
    - Jump if less than or equals (<=)
    - JLE Cx
    - if SW(<=), PC <= PC + Cx
  * - J
    - JGE
    - 35
    - Jump if greater than or equals (>=)
    - JGE Cx
    - if SW(>=), PC <= PC + Cx
  * - J
    - JMP
    - 36
    - Jump (unconditional)
    - JMP Cx
    - PC <= PC + Cx
  * - J
    - JALR
    - 39
    - Indirect jump
    - JALR Rb
    - LR <= PC; PC <= Rb [#call-note]_
  * - J
    - BAL
    - 3A
    - Branch and link
    - BAL Cx
    - LR <= PC; PC <= PC + Cx
  * - J
    - JSUB
    - 3B
    - Jump to subroutine
    - JSUB Cx
    - LR <= PC; PC <= PC + Cx
  * - J
    - JR/RET
    - 3C
    - Return from subroutine
    - JR $1 or RET LR
    - PC <= LR [#jr-note]_
  * - A
    - MULT
    - 41
    - Multiply for 64 bits result
    - MULT Ra, Rb
    - (HI,LO) <= MULT(Ra,Rb)
  * - A
    - MULTU
    - 42
    - MULT for unsigned 64 bits
    - MULTU Ra, Rb
    - (HI,LO) <= MULTU(Ra,Rb)
  * - A
    - DIV
    - 43
    - Divide
    - DIV Ra, Rb
    - HI<=Ra%Rb, LO<=Ra/Rb
  * - A
    - DIVU
    - 44
    - Divide unsigned
    - DIVU Ra, Rb
    - HI<=Ra%Rb, LO<=Ra/Rb
  * - A
    - MFHI
    - 46
    - Move HI to GPR
    - MFHI Ra
    - Ra <= HI
  * - A
    - MFLO
    - 47
    - Move LO to GPR
    - MFLO Ra
    - Ra <= LO
  * - A
    - MTHI
    - 48
    - Move GPR to HI
    - MTHI Ra
    - HI <= Ra
  * - A
    - MTLO
    - 49
    - Move GPR to LO
    - MTLO Ra
    - LO <= Ra
  * - A
    - MFC0
    - 50
    - Move C0R to GPR
    - MFC0 Ra, Rb
    - Ra <= Rb
  * - A
    - MTC0
    - 51
    - Move GPR to C0R
    - MTC0 Ra, Rb
    - Ra <= Rb
  * - A
    - C0MOV
    - 52
    - Move C0R to C0R
    - C0MOV Ra, Rb
    - Ra <= Rb


The following table provides details on the newly added cpu032II instruction set:

.. list-table:: cpu032II Instruction Set
  :widths: 1 4 3 11 7 10
  :header-rows: 1

  * - F\.
    - Mnemonic
    - Opcode
    - Meaning
    - Syntax
    - Operation
  * - L
    - SLTi
    - 26
    - Set less Then
    - SLTi Ra, Rb, Cx
    - Ra <= (Rb < Cx)
  * - L
    - SLTiu
    - 27
    - SLTi unsigned 
    - SLTiu Ra, Rb, Cx
    - Ra <= (Rb < Cx)
  * - A
    - SLT
    - 28
    - Set less Then
    - SLT Ra, Rb, Rc
    - Ra <= (Rb < Rc)
  * - A
    - SLTu
    - 29
    - SLT unsigned
    - SLTu Ra, Rb, Rc
    - Ra <= (Rb < Rc)
  * - L
    - BEQ
    - 37
    - Branch if equal
    - BEQ Ra, Rb, Cx
    - if (Ra==Rb), PC <= PC + Cx
  * - L
    - BNE
    - 38
    - Branch if not equal
    - BNE Ra, Rb, Cx
    - if (Ra!=Rb), PC <= PC + Cx

.. note:: **Cpu0 Unsigned Instructions**  

   Like MIPS, except for `DIVU`, arithmetic unsigned instructions such as  
   `ADDu` and `SUBu` do not trigger overflow exceptions.  
   The `ADDu` and `SUBu` handle both signed and unsigned integers correctly.  

   For example:  

   - `(ADDu 1, -2) = -1`  
   - `(ADDu 0x01, 0xfffffffe) = 0xffffffff (4G - 1)`  

   If you interpret the result as a negative value, it is `-1`.  
   If interpreted as positive, it is `+4G - 1`.  

Why Not Use ADD Instead of SUB?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From introductory computer science textbooks, we know that `SUB` can be  
replaced by `ADD` as follows:  

- `(A - B) = (A + (-B))`  

Since MIPS represents `int` in C using 32 bits, consider the case where  
`B = -2G`:  

- `(A - (-2G)) = (A + 2G)`  

However, the problem is that while `-2G` can be represented in a 32-bit  
machine, `+2G` cannot. This is because the range of 32-bit two's complement  
representation is `(-2G .. 2G-1)`.  

Two's complement representation allows for efficient computation in hardware  
design, making it widely used in real CPU implementations.  
This is why almost every CPU includes a `SUB` instruction rather than relying  
solely on `ADD`.  

The Status Register
*******************

The Cpu0 status word register (`SW`) contains the state of the following flags:  

- **Negative (N)**  
- **Zero (Z)**  
- **Carry (C)**  
- **Overflow (V)**  
- **Debug (D)**  
- **Mode (M)**  
- **Interrupt (I)**  

The bit layout of the `SW` register is shown in :numref:`llvmstructure-f3` below.

.. _llvmstructure-f3: 
.. figure:: ../Fig/llvmstructure/3.png
  :width: 684 px
  :height: 126 px
  :align: center

  Cpu0 status word (SW) register

When a `CMP Ra, Rb` instruction executes, it updates the condition flags in the  
status word (`SW`) register as follows:  

- **If `Ra > Rb`**, then `N = 0`, `Z = 0`  
- **If `Ra < Rb`**, then `N = 1`, `Z = 0`  
- **If `Ra = Rb`**, then `N = 0`, `Z = 1`  

The direction (i.e., taken or not taken) of conditional jump instructions  
(`JGT`, `JLT`, `JGE`, `JLE`, `JEQ`, `JNE`) is determined by the values of the  
`N` and `Z` flags in the `SW` register.

Cpu0's Stages of Instruction Execution
**************************************

The Cpu0 architecture has a five-stage pipeline. The stages are:  
instruction fetch (IF), instruction decode (ID), execute (EX), memory access  
(MEM), and write-back (WB).  

Below is a description of what happens in each stage of the processor:  

1) **Instruction Fetch (IF)**  

- The Cpu0 fetches the instruction pointed to by the Program Counter (PC)  
  into the Instruction Register (IR): `IR = [PC]`.  
- The PC is then updated to point to the next instruction: `PC = PC + 4`.  

2) **Instruction Decode (ID)**  

- The control unit decodes the instruction stored in `IR`, routes necessary  
  data from registers to the ALU, and sets the ALU's operation mode based on  
  the instruction's opcode.  

3) **Execute (EX)**  

- The ALU executes the operation designated by the control unit on the data  
  in registers.  
- Except for load and store instructions, the result is stored in the  
  destination register after execution.  

4) **Memory Access (MEM)**  

- If the instruction is a load, data is read from the data cache into the  
  pipeline register `MEM/WB`.  
- If the instruction is a store, data is written from the register to the  
  data cache.  

5) **Write-Back (WB)**  

- If the instruction is a load, data is moved from the pipeline register  
  `MEM/WB` to the destination register.

Cpu0's Interrupt Vector
***********************

.. table:: Cpu0's Interrupt Vector

  ========  ===========
  Address   type
  ========  ===========
  0x00      Reset
  0x04      Error Handle
  0x08      Interrupt
  ========  ===========


Clang
-----

LLVM is middleware for compilers, with Clang as its frontend.
The Clang project provides a language front-end and tooling infrastructure for 
languages in the C language family (C, C++, Objective C/C++, OpenCL, and CUDA) 
for the LLVM project. 

Context Free Grammar
********************

**Definition:**

- “A context-free grammar defines a language that can be parsed independently
  of surrounding input context; each production rule applies based solely on 
  the current nonterminal, not on neighboring symbols.”

- All context-free grammars (CFGs) can be expressed in Backus-Naur Form (BNF).

.. note::

  Computer languages have been adding complexity features for users' programming:

  ☆ "(**≈ 30 years ago**) Programming languages are **context‑free**." 
  :math:`\Rightarrow` 
  "(**Today**) The syntax is context‑free, but the language is 
  **context‑sensitive**."

⚠️  1. What older textbooks said

Textbooks from the 1980s–1990s typically taught:

- “**Programming languages** are mostly **context‑free**.”
- “**Parsing is done with CFGs**.”
- “Semantic analysis comes later.”

They treated semantic constraints as a separate phase, not as part of the grammar

✅  2. What modern textbooks say

Modern compiler books (e.g., newer editions of Aho/Ullman, Appel, Cooper/Torczon, Muchnick, and engineering‑oriented texts) now teach something closer to:

- The surface syntax is context‑free.
- Real languages require context-sensitive semantic analysis.
- Some languages **require semantic information during parsing** (C++, Rust, 
  Swift).
- Macro systems and type inference break the clean CFG model.

They **no longer pretend that a CFG fully describes a real language**.

So the modern interpretation is:

- “The grammar is context‑free, but the language is not.”

This is a subtle but important shift.


.. list-table:: Context-Free Grammar Shifts in Compiler Theory
   :header-rows: 1
   :widths: 25 35 40

   * - Statement
     - Meaning
     - How Modern Languages Changed

   * - **Old Textbook Statement  
       (≈ 30 years ago)**  
       “**Programming languages** are **context‑free**.”
     - Focused only on the **parser grammar**.  
       Treated semantic rules as a separate phase,  
       not part of the language definition.
     - Languages were simpler (C, Pascal, early C++).  
       Few features required semantic feedback during parsing.  
       Grammar-based teaching matched real compilers more closely.

   * - **Modern Understanding  
       (today)**  
       “The **syntax is context‑free**, but the **language is context‑sensitive**.”
     - Parsing still uses a CFG, but real languages rely on  
       **name resolution**, **type inference**, **generics**,  
       **macro expansion**, and **semantic disambiguation**.
     - Modern languages (C++, Rust, Swift, Kotlin, Go)  
       require semantic information during parsing.  
       Macro systems and type systems break pure CFG boundaries.  
       Compilers use multi‑phase frontends to resolve context.


Why doesn't the Clang compiler use YACC/LEX tools to parse C++?
****************************************************************

Clang does not use YACC/LEX because **C++ is too complex and context-sensitive**
for traditional parser generators. YACC and LEX work with context-free grammars, 
but C++ has many context-sensitive features, especially in templates below:

.. rubric:: Context-sensitive template instantiation
.. literalinclude:: ../References/cpp-template.cpp
   :language: c++

.. code-block:: c++

  References % clang++ -DFUNCTION=1 -DTEMPLATE=0 cpp-template.cpp
  References % ./a.out                                           
  Non-template f(int)
  Non-template f(int)
  References % clang++ -DFUNCTION=0 -DTEMPLATE=1 cpp-template.cpp
  References % ./a.out                                           
  Template f(T)
  Template f(T)
  Template f(T)
  References % clang++ -DFUNCTION=1 -DTEMPLATE=1 cpp-template.cpp
  References % ./a.out                                           
  Non-template f(int)
  Template f(T)
  Template f(T)

In the C++ code above, both f(42) and f('a') can match either the template 
function or the non-template function.

✅  Why This Is Hard for YACC:

YACC operates on context-free grammars, but this example is context-sensitive.
The expression f('a'); selects a template if a template definition exists; 
otherwise, it selects a function if a function definition exists. As a result, 
this behavior cannot be implemented using BNF-based tools like YACC/LEX.

To parse this example, the following are required:

- Template argument deduction: The compiler must infer T from the call.

- Overload resolution: It must choose between the template and non-template 
  versions.

- Implicit conversions: 'a' can be converted to int, which affects overload 
  ranking.

- Explicit template instantiation: f<int>('a') forces the template, but YACC 
  doesn’t track template types.

To model this in YACC:

You’d need to simulate template instantiation and ranking — which is way beyond 
what YACC was designed for.

This kind of logic is not just syntactic — it’s deeply semantic. That’s why 
compilers like Clang use handwritten parsers with tight integration between 
parsing and semantic analysis.

Clang doesn’t use YACC/LEX because:

==================================  ========  ====================
Feature                             YACC/LEX  Hand-written Parser
==================================  ========  ====================
Handles context-sensitive grammar   ❌        ✅
Good error recovery                 ❌        ✅
Integration with semantic analysis  ❌        ✅
Easy to maintain/extend for C++     ❌        ✅
Fine-grained control                ❌        ✅
==================================  ========  ====================

The GNU `g++` compiler abandoned BNF tools starting from version 3.x.  


Compiler-Compiler Tools for Context-Sensitive C++ Parsing
*********************************************************

While traditional tools like YACC/Lex are limited to context-free grammars, 
modern compiler construction requires handling context-sensitive features — 
especially in C++ templates, overload resolution, and semantic analysis. 
Below is a list of tools that attempt to address these challenges.

====================  ========================  ============================  =======================================================================
Tool                  Generates Parser Code?    Context-Sensitive Support?    Notes
====================  ========================  ============================  =======================================================================
ANTLR                 ✅ Yes                    ⚠️ Limited                     Supports semantic predicates; struggles with full C++ complexity
BNFLite               ✅ Yes                    ⚠️ Partial                     Lightweight C++ template library; ideal for DSLs, not full C++
PEGTL                 ✅ Yes                    ⚠️ Limited                     PEG-based parser combinator library in C++; expressive but limited
GLR Parsers (Elsa)    ✅ Yes                    ✅ Yes                        Can handle ambiguity and deferred resolution; used in research
Clang LibTooling      ✅ Yes (via AST)          ✅ Yes                        Offers full C++ parsing + semantic analysis; industrial-grade tooling
====================  ========================  ============================  =======================================================================

Why Most Tools Fall Short:

- C++ templates are **Turing-complete**, making static analysis alone insufficient.

- Overload resolution requires understanding **types, scopes, and conversions**.

- C++ syntax is **deeply ambiguous**, defying context-free parsing strategies.

Recommended Approach:

For building C++ parsers:

- Use **GLR-based tools** like Elsa if ambiguity and template complexity must 
  be handled directly.

- Or leverage **Clang LibTooling** for full semantic integration, AST 
  manipulation, and robust code analysis.

In summary, while modern tools improve on YACC/LEX, **the complexity of C++ still
requires a custom parser that deeply integrates with semantic analysis and type
resolution. Clang’s approach remains the most practical for full C++ support.
Moreover the error messages and recovery are still weaker than Clang**.

While C++ compilers do not benefit from BNF  
generator tools, many other programming and scripting languages, which are  
more context-free, can take advantage of them. 
The following information comes from Wikipedia:  

Java syntax has a context-free grammar that can be parsed by a simple LALR  
parser. Parsing C++ is more complicated [#java-cpp]_.  


LLVM Structure
--------------

This section introduces the compiler's data structures, algorithms, and  
mechanisms used in LLVM.  

SSA Form
********

Static Single Assignment (SSA) form ensures that each variable is assigned  
exactly once. In SSA form, a single instruction has one variable (destination  
virtual registers).
However one virtual register may map to two real registers.
LLVM handles it by packing them into a single value, like a struct or a vector, 
or using multiple instructions as follows:

.. code-block:: console

  %res = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %sum = extractvalue {i32, i1} %res, 0
  %overflow = extractvalue {i32, i1} %res, 1

.. code-block:: console

  %y = call <4 x float> @llvm.ceil.v4f32(<4 x float> %x)

LLVM IR follows SSA form, meaning it has an **unbounded number of virtual  
registers**—each variable is assigned exactly once and is stored in a separate  
virtual register.  

As a result, the optimization steps in the code generation sequence—including  
**Instruction Selection**, **Scheduling and Formation**, and **Register  
Allocation**—retain all optimization opportunities.  

For example, if we used a limited number of virtual registers instead of an  
unlimited set, as shown in the following code:

.. code-block:: console

    %a = add nsw i32 1, i32 0
    store i32 %a, i32* %c, align 4
    %a = add nsw i32 2, i32 0
    store i32 %a, i32* %c, align 4

In the above example, a limited number of virtual registers is used, causing  
virtual register `%a` to be assigned twice.  

As a result, the compiler must generate the following code, since `%a` is  
assigned as an output in two different statements.

.. code-block:: console

  => %a = add i32 1, i32 0
      st %a,  i32* %c, 1
      %a = add i32 2, i32 0
      st %a,  i32* %c, 2

The above code must execute sequentially.  

In contrast, the SSA form shown below can be reordered and executed in parallel  
using the following alternative version [#dragonbooks-10.2.3]_.

.. code-block:: console

    %a = add nsw i32 1, i32 0
    store i32 %a, i32* %c, align 4
    %b = add nsw i32 2, i32 0
    store i32 %b, i32* %d, align 4

  // version 1
  => %a = add i32 1, i32 0
      st %a,  i32* %c, 0
      %b = add i32 2, i32 0
      st %b,  i32* %d, 0

  // version 2
  => %a = add i32 1, i32 0
      %b = add i32 2, i32 0
      st %a,  i32* %c, 0
      st %b,  i32* %d, 0

  // version 3
  => %b = add i32 2, i32 0
      st %b,  i32* %d, 0
      %a = add i32 1, i32 0
      st %a,  i32* %c, 0


DSA Form
*********

.. code-block:: console

    for (int i = 0; i < 1000; i++) {
      b[i] = f(g(a[i]));
    }
    
For the source program above, the following represent its SSA form at both the  
source code level and the LLVM IR level, respectively.  

.. code-block:: c++

    for (int i = 0; i < 1000; i++) {
      t = g(a[i]);
      b[i] = f(t);
    }
    
.. code-block:: llvm

      %pi = alloca i32
      store i32 0, i32* %pi
      %i = load i32, i32* %pi
      %cmp = icmp slt i32 %i, 1000
      br i1 %cmp, label %true, label %end
    true:
      %a_idx = add i32 %i, i32 %a_addr
      %val0 = load i32, i32* %a_idx
      %t = call i64 %g(i32 %val0)
      %val1 = call i64 %f(i32 %t)
      %b_idx = add i32 %i, i32 %b_addr
      store i32 %val1, i32* %b_idx
    end:

    
The following represents the **DSA (Dynamic Single Assignment) form**.  

.. code-block:: c++

    for (int i = 0; i < 1000; i++) {
      t[i] = g(a[i]);
      b[i] = f(t[i]);
    }
    
.. code-block:: llvm

      %pi = alloca i32
      store i32 0, i32* %pi
      %i = load i32, i32* %pi
      %cmp = icmp slt i32 %i, 1000
      br i1 %cmp, label %true, label %end
    true:
      %a_idx = add i32 %i, i32 %a_addr
      %val0 = load i32, i32* %a_idx
      %t_idx = add i32 %i, i32 %t_addr
      %temp = call i64 %g(i32 %val0)
      store i32 %temp, i32* %t_idx
      %val1 = call i64 %f(i32 %temp)
      %b_idx = add i32 %i, i32 %b_addr
      store i32 %val1, i32* %b_idx
    end:
    
In some internet video applications and multi-core (SMP) platforms, splitting  
`g()` and `f()` into two separate loops can improve performance.  

DSA allows this transformation, whereas SSA does not. While extra analysis on  
`%temp` in SSA could reconstruct `%t_idx` and `%t_addr` as shown in the DSA  
form below, compiler transformations typically follow a high-to-low approach.  

Additionally, LLVM IR already loses the `for` loop structure, even though part
of the losted information  
can be reconstructed through further analysis.  

For this reason, in this book—as well as in most compiler-related research—the  
discussion follows a high-to-low transformation premise. Otherwise, it would  
fall into the domain of **reverse engineering** in assemblers or compilers.  

.. code-block:: c++

    for (int i = 0; i < 1000; i++) {
      t[i] = g(a[i]);
    }
    
    for (int i = 0; i < 1000; i++) {
      b[i] = f(t[i]);
    }
    
.. code-block:: llvm

      %pi = alloca i32
      store i32 0, i32* %pi
      %i = load i32, i32* %pi
      %cmp = icmp slt i32 %i, 1000
      br i1 %cmp, label %true, label %end
    true:
      %a_idx = add i32 %i, i32 %a_addr
      %val0 = load i32, i32* %a_idx
      %t_idx = add i32 %i, i32 %t_addr
      %temp = call i32 %g(i32 %val0)
      store i32 %temp, i32* %t_idx
    end:

      %pi = alloca i32
      store i32 0, i32* %pi
      %i = load i32, i32* %pi
      %cmp = icmp slt i32 %i, 1000
      br i1 %cmp, label %true, label %end
    true:
      %t_idx = add i32 %i, i32 %t_addr
      %temp = load i32, i32* %t_idx
      %val1 = call i32 %f(i32 %temp)
      %b_idx = add i32 %i, i32 %b_addr
      store i32 %val1, i32* %b_idx
    end:

Now, data dependencies exist only on `t[i]` between `"t[i] = g(a[i])"` and  
`"b[i] = f(t[i])"` for each `i = (0..999)`.  

As a result, the program can execute in various orders, offering significant  
parallel processing opportunities for **multi-core (SMP) systems** and  
**heterogeneous processors**.  

For example, `g(x)` can be executed on a **GPU**, while `f(x)` runs on a **CPU**.  

Three-Phase Design
******************

This content and the following sub-section are adapted from the AOSA chapter  
on LLVM written by Chris Lattner [#aosa-book]_.  

The most common design for a traditional static compiler (such as most C  
compilers) follows a three-phase structure, consisting of the front end,  
the optimizer, and the back end, as shown in :numref:`llvmstructure-f6`.  

The **front end** parses the source code, checks for errors, and constructs  
a language-specific Abstract Syntax Tree (AST) to represent the input code.  
The AST may then be converted into an intermediate representation for  
optimization, after which the **optimizer** and **back end** process the code.

.. _llvmstructure-f6: 
.. figure:: ../Fig/llvmstructure/6.png
  :width: 470 px
  :height: 63 px
  :scale: 70 %
  :align: center

  Three Major Components of a Three Phase Compiler

The optimizer performs a wide range of transformations to improve code execution  
efficiency, such as eliminating redundant computations. It is generally  
independent of both the source language and the target architecture.  

The back end, also known as the code generator, maps the optimized code onto  
the target instruction set. In addition to producing correct code, it is  
responsible for generating efficient code that leverages the unique features of  
the target architecture. Common components of a compiler back end include  
instruction selection, register allocation, and instruction scheduling.  

This model applies equally well to interpreters and Just-In-Time (JIT)  
compilers. The Java Virtual Machine (JVM) is an example of this model, using  
Java bytecode as the interface between the front end and the optimizer.  

The greatest advantage of this classical design becomes evident when a compiler  
supports multiple source languages or target architectures. If the compiler's  
optimizer uses a common intermediate representation, a front end can be written  
for any language that compiles to this representation, and a back end can be  
developed for any target that compiles from it, as illustrated in  
:numref:`llvmstructure-f7`.  

.. _llvmstructure-f7: 
.. figure:: ../Fig/llvmstructure/7.png
  :align: center
  :width: 837 px
  :height: 299 px
  :scale: 70 %

  Retargetablity

With this design, porting the compiler to support a new source language  
(e.g., Algol or BASIC) requires implementing a new front end, while the  
existing optimizer and back end can be reused. If these components were not  
separated, adding a new source language would require rebuilding the entire  
compiler from scratch. Supporting `N` targets and `M` source languages would  
then necessitate developing `N * M` compilers.  

Another advantage of the three-phase design, which stems from its  
retargetability, is that the compiler can serve a broader range of programmers  
compared to one that supports only a single source language and target. For an  
open-source project, this translates to a larger community of potential  
contributors, leading to more enhancements and improvements.  

This is why open-source compilers that cater to diverse communities, such as  
GCC, often generate better-optimized machine code than narrower compilers like  
FreePASCAL. In contrast, the quality of proprietary compilers depends directly  
on their development budget. For example, the Intel ICC compiler is widely  
recognized for producing high-quality machine code despite serving a smaller  
audience.  

A final major benefit of the three-phase design is that the skills required to  
develop a front end differ from those needed for the optimizer and back end.  
By separating these concerns, "front-end developers" can focus on enhancing  
and maintaining their part of the compiler. While this is a social rather than  
a technical factor, it has a significant impact in practice—especially for  
open-source projects aiming to lower barriers to contribution.  

The most critical aspect of this design is the **LLVM Intermediate  
Representation (IR)**, which serves as the compiler's core code representation.  
LLVM IR is designed to support mid-level analysis and transformations commonly  
found in the optimization phase of a compiler.  

It was created with several specific goals, including support for lightweight  
runtime optimizations, cross-function and interprocedural optimizations, whole-  
program analysis, and aggressive restructuring transformations. However, its  
most defining characteristic is that it is a first-class language with well-  
defined semantics.  

To illustrate this, here is a simple example of an LLVM `.ll` file:

.. code-block:: llvm

  define i32 @add1(i32 %a, i32 %b) {
  entry:
    %tmp1 = add i32 %a, %b
    ret i32 %tmp1
  }
  define i32 @add2(i32 %a, i32 %b) {
  entry:
    %tmp1 = icmp eq i32 %a, 0
    br i1 %tmp1, label %done, label %recurse
  recurse:
    %tmp2 = sub i32 %a, 1
    %tmp3 = add i32 %b, 1
    %tmp4 = call i32 @add2(i32 %tmp2, i32 %tmp3)
    ret i32 %tmp4
  done:
    ret i32 %b
  }

.. code-block:: c++

  // Above LLVM IR corresponds to this C code, which provides two different ways to
  //  add integers:
  unsigned add1(unsigned a, unsigned b) {
    return a+b;
  }
  // Perhaps not the most efficient way to add two numbers.
  unsigned add2(unsigned a, unsigned b) {
    if (a == 0) return b;
    return add2(a-1, b+1);
  }

As shown in this example, LLVM IR is a low-level, RISC-like virtual instruction  
set. Like a real RISC instruction set, it supports linear sequences of simple  
instructions such as `add`, `subtract`, `compare`, and `branch`.  

These instructions follow a three-address format, meaning they take inputs and  
produce a result in a different register. LLVM IR supports labels and generally  
resembles an unusual form of assembly language.  

Unlike most RISC instruction sets, LLVM IR is strongly typed and uses a simple  
type system (e.g., `i32` represents a 32-bit integer, and `i32**` is a pointer  
to a pointer to a 32-bit integer). Additionally, some machine-specific details  
are abstracted away.  

For instance, the calling convention is handled through `call` and `ret`  
instructions with explicit arguments. Another key difference from machine code  
is that LLVM IR does not use a fixed set of named registers. Instead, it  
employs an infinite set of temporaries prefixed with `%`.  

Beyond being a language, LLVM IR exists in three isomorphic forms:

- A **textual format** (as seen above).  
- An **in-memory data structure** used by optimizations.  
- A **compact binary "bitcode" format** stored on disk.  

The LLVM project provides tools to convert between these forms:

- `llvm-as` assembles a textual `.ll` file into a `.bc` file containing  
  bitcode.  
- `llvm-dis` disassembles a `.bc` file back into a `.ll` file.  

The intermediate representation (IR) of a compiler is crucial because it  
creates an ideal environment for optimizations. Unlike the front end and  
back end, the optimizer is not restricted to a specific source language or  
target machine.  

However, it must effectively serve both. It should be easy for the front end  
to generate while remaining expressive enough to enable important  
optimizations for real hardware targets.

LLVM's Target Description Files: .td
************************************

The "mix and match" approach allows target authors to select components that  
best suit their architecture, enabling significant code reuse across different  
targets.  

However, this introduces a challenge: each shared component must be capable of  
handling target-specific properties in a generic way. For instance, a shared  
register allocator must be aware of the register file of each target and the  
constraints that exist between instructions and their register operands.  

LLVM addresses this challenge by requiring each target to provide a target  
description using a declarative domain-specific language, defined in a set of  
`.td` files. These files are processed by the `tblgen` tool to generate the  
necessary target-specific data structures.  

The simplified build process for the x86 target is illustrated in  
:numref:`llvmstructure-f8`.

.. _llvmstructure-f8: 
.. figure:: ../Fig/llvmstructure/8.png
  :align: center
  :width: 850 px
  :height: 428 px
  :scale: 70 %

  Simplified x86 Target Definition

The different subsystems supported by `.td` files enable target authors to  
construct various components of their target architecture.  

For example, the x86 backend defines a register class named `"GR32"`, which  
contains all 32-bit registers. In `.td` files, target-specific definitions  
are conventionally written in all capital letters. The definition is as follows:

.. code-block:: c++

  def GR32 : RegisterClass<[i32], 32,
    [EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP,
     R8D, R9D, R10D, R11D, R14D, R15D, R12D, R13D]> { ... }
     
The language used in `.td` files is the Target (Hardware) Description Language,  
which allows LLVM backend compiler engineers to define the transformation from  
LLVM IR to machine instructions for their CPUs.  

In the frontend, compiler development tools provide a **Parser Generator** for  
building compilers. In the backend, they offer a **Machine Code Generator** to  
facilitate instruction selection and code generation, as shown in  
:numref:`llvmstructure_frontendTblGen` and :numref:`llvmstructure_llvmTblGen`.

.. _llvmstructure_frontendTblGen:
.. graphviz:: ../Fig/llvmstructure/frontendTblGen.gv
  :caption: Frontend TableGen Flow

.. _llvmstructure_llvmTblGen:
.. graphviz:: ../Fig/llvmstructure/llvmTblGen.gv
  :caption: llvm TableGen Flow

LLVM Code Generation Sequence
*****************************

Following diagram is from `tricore_llvm.pdf`.

.. _llvmstructure-f9: 
.. figure:: ../Fig/llvmstructure/9.png
  :width: 1030 px
  :height: 537 px
  :align: center

  `tricore_llvm.pdf`: **Code Generation Sequence**  
  On the path from LLVM code to assembly code, numerous passes are executed,  
  and several data structures are used to represent intermediate results.

LLVM is a **Static Single Assignment (SSA)**-based representation.  
It provides an infinite number of virtual registers that can hold values of  
primitive types, including integral, floating-point, and pointer values.  

In LLVM's SSA representation, each operand is stored in a separate virtual  
register. Comments in LLVM IR are denoted by the `;` symbol.  

The following are examples of LLVM SSA instructions:

.. code-block:: llvm

  store i32 0, i32* %a  ; store i32 type of 0 to virtual register %a, %a is
              ;  pointer type which point to i32 value
  store i32 %b, i32* %c ; store %b contents to %c point to, %b isi32 type virtual
              ;  register, %c is pointer type which point to i32 value.
  %a1 = load i32* %a    ; load the memory value where %a point to and assign the
              ;  memory value to %a1
  %a3 = add i32 %a2, 1  ; add %a2 and 1 and save to %a3

We explain the code generation process below.  
If you are unfamiliar with the concepts, we recommend first reviewing  
Section 4.2 of `tricore_llvm.pdf`.  

You may also refer to *The LLVM Target-Independent Code Generator* [#codegen]_  
and the *LLVM Language Reference Manual* [#langref]_. However, we believe that  
Section 4.2 of `tricore_llvm.pdf` provides sufficient information.  

We suggest consulting the above web documents only if you still have  
difficulties understanding the material, even after reading this section and  
the next two sections on **DAG** and **Instruction Selection**.

1. **Instruction Selection**  

.. code-block:: console  

  // In this stage, the LLVM opcode is transformed into a machine opcode,  
  // but the operand remains an LLVM virtual operand.  
      store i16 0, i16* %a  // Store 0 of i16 type to the location pointed to by %a  
  =>  st i16 0, i32* %a     // Use the Cpu0 backend instruction `st` instead of `store`.  

2. **Scheduling and Formation**  

.. code-block:: console  

  // In this stage, instruction order is optimized for execution cycles  
  // or to reduce register pressure.  
      st i32 %a, i16* %b, i16 5  // Store %a to *(%b + 5)  
      st %b, i32* %c, i16 0  
      %d = ld i32* %c  

  // The instruction order is rearranged. In RISC CPUs like MIPS,  
  // `ld %c` depends on the previous `st %c`, requiring a 1-cycle delay.  
  // This means `ld` cannot immediately follow `st`.  
  =>  st %b, i32* %c, i16 0  
      st i32 %a, i16* %b, i16 5  
      %d = ld i32* %c, i16 0  

  // Without instruction reordering, a `nop` instruction must be inserted,  
  // adding an extra cycle. (In reality, MIPS dynamically schedules  
  // instructions and inserts `nop` between `st` and `ld` if necessary.)  
      st i32 %a, i16* %b, i16 5  
      st %b, i32* %c, i16 0  
      nop  
      %d = ld i32* %c, i16 0  

  // **Minimizing Register Pressure**  
  // Suppose `%c` remains live after the basic block, but `%a` and `%b` do not.  
  // Without reordering, at least 3 registers are required:  
      %a = add i32 1, i32 0  
      %b = add i32 2, i32 0  
      st %a, i32* %c, 1  
      st %b, i32* %c, 2  

  // The reordered version reduces register usage to 2 by allocating `%a`  
  // and `%b` in the same...

  // Register allocation optimization  
  => %a = add i32 1, i32 0  
      st %a, i32* %c, 1  
      %b = add i32 2, i32 0  
      st %b, i32* %c, 2    

3. **SSA-Based Machine Code Optimization**  

   For example, common subexpression elimination, as shown in the next  
   section on **DAG**.  

4. **Register Allocation**  

   Assign physical registers to virtual registers.  

5. **Prologue/Epilogue Code Insertion**  

   Explained in the section **Add Prologue/Epilogue Functions**.  

6. **Late Machine Code Optimizations**  

   Any "last-minute" peephole optimizations of the final machine code  
   are applied in this phase.  
   For example, replacing `x = x * 2` with `x = x << 1` for integer operands.  

7. **Code Emission**  

   The final machine code is emitted.  
   - For **static compilation**, the output is an assembly file.  
   - For **JIT compilation**, machine instruction opcodes are written into memory.  

The LLVM code generation sequence can also be viewed using:  

``llc -debug-pass=Structure``  

as shown below. The first four code generation stages from  
:numref:`llvmstructure-f9` appear in the  
**'DAG->DAG Pattern Instruction Selection'** section of the  
``llc -debug-pass=Structure`` output.  

The order of **Peephole Optimizations** and **Prologue/Epilogue Insertion**  
differs between :numref:`llvmstructure-f9` and  
``llc -debug-pass=Structure`` (marked with `*` in the output).  

There is no need to be concerned about this, as LLVM is continuously evolving,  
and its internal sequence may change over time.

.. code-block:: console

  118-165-79-200:input Jonathan$ llc --help-hidden
  OVERVIEW: llvm system compiler
  
  USAGE: llc [options] <input bitcode>
  
  OPTIONS:
  ...
    -debug-pass                             - Print PassManager debugging information
      =None                                 -   disable debug output
      =Arguments                            -   print pass arguments to pass to 'opt'
      =Structure                            -   print pass structure before run()
      =Executions                           -   print pass name before it is executed
      =Details                              -   print pass details when it is executed
  
  118-165-79-200:input Jonathan$ llc -march=mips -debug-pass=Structure ch3.bc
  ...
  Target Library Information
  Target Transform Info
  Data Layout
  Target Pass Configuration
  No Alias Analysis (always returns 'may' alias)
  Type-Based Alias Analysis
  Basic Alias Analysis (stateless AA impl)
  Create Garbage Collector Module Metadata
  Machine Module Information
  Machine Branch Probability Analysis
    ModulePass Manager
      FunctionPass Manager
        Preliminary module verification
        Dominator Tree Construction
        Module Verifier
        Natural Loop Information
        Loop Pass Manager
          Canonicalize natural loops
        Scalar Evolution Analysis
        Loop Pass Manager
          Canonicalize natural loops
          Induction Variable Users
          Loop Strength Reduction
        Lower Garbage Collection Instructions
        Remove unreachable blocks from the CFG
        Exception handling preparation
        Optimize for code generation
        Insert stack protectors
        Preliminary module verification
        Dominator Tree Construction
        Module Verifier
        Machine Function Analysis
        Natural Loop Information
        Branch Probability Analysis
      * MIPS DAG->DAG Pattern Instruction Selection
        Expand ISel Pseudo-instructions
        Tail Duplication
        Optimize machine instruction PHIs
        MachineDominator Tree Construction
        Slot index numbering
        Merge disjoint stack slots
        Local Stack Slot Allocation
        Remove dead machine instructions
        MachineDominator Tree Construction
        Machine Natural Loop Construction
        Machine Loop Invariant Code Motion
        Machine Common Subexpression Elimination
        Machine code sinking
      * Peephole Optimizations
        Process Implicit Definitions
        Remove unreachable machine basic blocks
        Live Variable Analysis
        Eliminate PHI nodes for register allocation
        Two-Address instruction pass
        Slot index numbering
        Live Interval Analysis
        Debug Variable Analysis
        Simple Register Coalescing
        Live Stack Slot Analysis
        Calculate spill weights
        Virtual Register Map
        Live Register Matrix
        Bundle Machine CFG Edges
        Spill Code Placement Analysis
      * Greedy Register Allocator
        Virtual Register Rewriter
        Stack Slot Coloring
        Machine Loop Invariant Code Motion
      * Prologue/Epilogue Insertion & Frame Finalization
        Control Flow Optimizer
        Tail Duplication
        Machine Copy Propagation Pass
      * Post-RA pseudo instruction expansion pass
        MachineDominator Tree Construction
        Machine Natural Loop Construction
        Post RA top-down list latency scheduler
        Analyze Machine Code For Garbage Collection
        Machine Block Frequency Analysis
        Branch Probability Basic Block Placement
        Mips Delay Slot Filler
        Mips Long Branch
        MachineDominator Tree Construction
        Machine Natural Loop Construction
      * Mips Assembly Printer
        Delete Garbage Collector Information

- Since **Instruction Scheduling** and **Dead Code Elimination** affect  
  **Register Allocation**, LLVM does not revisit earlier passes once a later  
  pass is completed. **Register Allocation** occurs after **Instruction  
  Scheduling**.  

  The passes from **Live Variable Analysis** to **Greedy Register Allocator**  
  handle **Register Allocation**. More details on register allocation passes  
  can be found here: [#cmu-rac]_ [#ra-wiki]_.  


LLVM vs. GCC in Structure
*************************

The official GCC documentation can be found here: [#gnu]_.

.. table:: clang vs gcc-frontend

  ======================  ============================  =============
  frontend                clang                         gcc-frontend [#gcc-frontend]_
  ======================  ============================  =============
  LANGUAGE                C/C++                         C/C++
  parsing                 parsing                       parsing
  AST                     clang-AST                     GENERIC [#generic]_
  optimization & codgen   clang-backend                 gimplifier
  IR                      LLVM IR                       GIMPLE [#gimple]_
  ======================  ============================  =============

.. table:: llvm vs gcc (kernal and target/backend)

  ======================  ============================  =============
  backend                 llvm                          gcc
  ======================  ============================  =============
  IR                      LLVM IR                       GIMPLE
  transfer                optimziation & pass           optimization & plugins
  DAG                     DAG                           RTL [#rtl]_
  codgen                  tblgen for td                 codgen for md [#md]_
  ======================  ============================  =============

Both **LLVM IR** and **GIMPLE** use SSA form.  

LLVM IR was originally designed to be fully reusable across various tools,  
not just within the compiler itself. In contrast, the **GCC community** never  
intended for GIMPLE to be used beyond the compiler.  

Richard Stallman actively resisted efforts to make GCC's IR more reusable to  
prevent third-party commercial tools from leveraging GCC frontends.  
As a result, **GIMPLE (GCC's IR)** was never designed to fully describe a  
compiled program.  

For example, it lacks critical information such as the program's **call graph**,  
**type definitions**, **stack offsets**, and **alias information**  
[#llvm-ir-vs-gimple]_.  

LLVM Blog
*********

A user may rely on a **null pointer** as a guard to ensure code correctness.  
However, **undef** values occur only during compiler optimizations  
[#null_pointer_ex]_.  

If a user fails to explicitly bind a null pointer—either directly or  
indirectly—compilers like **LLVM** and **GCC** may interpret the null pointer  
as **undef**, leading to unexpected optimization behavior  
[#null_pointer]_.  

CFG (Control Flow Graph)
************************

The SSA form can be represented using a **Control Flow Graph (CFG)** and  
optimized by analyzing it.  

Each node in the graph represents a **basic block (BB)**—a straight-line  
sequence of code without any jumps or jump targets. A jump target always  
**starts** a basic block, while a jump **ends** one [#cfg-wiki]_.  

The following is an example of a **CFG**.  
**Jumps and branches always appear in the last statement of basic blocks (BBs)**  
as shown in :numref:`cfg_ex`.

.. rubric:: Fig/llvmstructure/cfg-ex.cpp
.. literalinclude:: ../Fig/llvmstructure/cfg-ex.cpp
   :language: c++

.. rubric:: Fig/llvmstructure/cfg-ex.ll
.. literalinclude:: ../Fig/llvmstructure/cfg-ex.ll
   :language: llvm

.. _cfg_ex:
.. graphviz:: ../Fig/llvmstructure/cfg-ex.dot
  :caption: CFG for cfg-ex.ll

DAG (Directed Acyclic Graph)
****************************

The SSA form within each **Basic Block (BB)** from the **Control Flow Graph  
(CFG)**, as discussed in the previous section, can be represented using a  
**Directed Acyclic Graph (DAG)**.  

Many key **local optimization** techniques begin by transforming a basic block  
into a DAG [#dragonbooks-8.5]_.  

For example, the basic block code and its corresponding DAG are illustrated in  
:numref:`llvmstructure-dag-ex`.  

.. _llvmstructure-dag-ex:  
.. graphviz:: ../Fig/llvmstructure/dag-ex.gv  
   :caption: The left example includes two destination registers, while  
             the right has only one destination.  

DAG and SSA allow instructions to have two destination virtual registers.  

Assume the `ediv` operation performs integer division, storing the **quotient**  
in `a` and the **remainder** in `d`.  

If only one destination register is used, the DAG may be simplified, as shown  
on the right in :numref:`llvmstructure-dag-ex`.  

If `b` is not live at the exit of the block, we can apply **common subexpression  
elimination**, as demonstrated in the table below.  

.. table:: Common Subexpression Elimination Process  
  
  ====================================  ==================================================================
  Replace node b with node d             Replace b\ :sub:`0`\ , c\ :sub:`0`\ , d\ :sub:`0`\  with b, c, d
  ====================================  ==================================================================
  a = b\ :sub:`0`\  + c\ :sub:`0`\       a = b + c
  d = a – d\ :sub:`0`\                   d = a – d
  c = d + c                              c = d + c
  ====================================  ==================================================================

After removing `b` and traversing the DAG from bottom to top  
(using **Depth-First In-Order Search** in a binary tree),  
we obtain the first column of the table above.  

As you can imagine, **common subexpression elimination** can be applied  
both at the **IR** level and in **machine code**.  

A **DAG** resembles a tree where **opcodes** are nodes,  
and **operands** (registers, constants, immediates, or offsets) are leaves.  
It can also be represented as a **prefix-ordered list** in a tree structure.  
For example, `(+ b, c)` and `(+ b, 1)` are IR DAG representations.  

In addition to **DAG optimization**, **kill registers** are discussed  
in Section 8.5.5 of the compiler book [#dragonbooks-8.5]_.  
This optimization method is also applied in LLVM.  

Instruction Selection
*********************

A major function of the backend is to **translate IR code into machine code**  
during **Instruction Selection**, as illustrated in :numref:`llvmstructure-f11`.  

.. _llvmstructure-f11:  
.. figure:: ../Fig/llvmstructure/11.png  
  :width: 495 px  
  :height: 116 px  
  :scale: 70 %  
  :align: center  

  IR and its corresponding machine instruction  

For **machine instruction selection**, the best approach is to represent both  
**IR** and **machine instructions** as a **DAG**.  

To simplify visualization, **register leaves** are omitted in  
:numref:`llvmstructure-f12`.  

The expression `rₖ + rⱼ` represents an **IR DAG** (used as a symbolic notation,  
not in LLVM SSA form). `ADD` is the corresponding machine instruction.

.. _llvmstructure-f12:  
.. figure:: ../Fig/llvmstructure/12.png  
  :width: 986 px  
  :height: 609 px  
  :scale: 70 %  
  :align: center  

  Instruction DAG representation  

The **IR DAG** and **machine instruction DAG** can also be represented as lists.  
For example:  

- **IR DAG lists:** `(+ rᵢ, rⱼ)` and `(- rᵢ, 1)`  
- **Machine instruction DAG lists:** `(ADD rᵢ, rⱼ)` and `(SUBI rᵢ, 1)`  

Now, let's examine the **ADDiu** instruction defined in `Cpu0InstrInfo.td`:  

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrFormats.td  
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrFormats.td  
    :start-after: //@class FL {  
    :end-before: //@class FL }  

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td  
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td  
    :start-after: //#if CH >= CH2 6  
    :end-before: #endif  

.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td  
    :start-after: //#if CH >= CH2 10  
    :end-before: #endif  

.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td  
    :start-after: //#if CH >= CH2 14  
    :end-before: #endif

:numref:`llvmstructure-f13` illustrates how pattern matching works between the  
**IR node** `add` and the **instruction node** `ADDiu`, both defined in  
`Cpu0InstrInfo.td`.  

In this example, the IR node `"add %a, 5"` is translated into `"addiu $r1, 5"`  
after `%a` is allocated to register `$r1` during the **register allocation**  
stage.  

This translation occurs because the IR pattern  
`(set RC:$ra, (OpNode RC:$rb, imm_type:$imm16))` is defined for `ADDiu`,  
where the second operand is a **signed immediate** that matches `%a, 5`.  

In addition to pattern matching, the `.td` file specifies the **assembly  
mnemonic** `"addiu"` and the **opcode** `0x09`.  

Using this information, **LLVM TableGen** automatically generates both assembly  
instructions and binary encodings. The resulting **binary instruction** can be  
included in an **ELF object file**, which will be explained in a later chapter.  

Similarly, **machine instruction DAG nodes** `LD` and `ST` are translated from  
the IR DAG nodes **load** and **store**.  

Note that in :numref:`llvmstructure-f13`, `$rb` represents a **virtual register**  
rather than a physical machine register. The details are further illustrated  
in :numref:`llvmstructure-dag`.  

.. _llvmstructure-f13:  
.. figure:: ../Fig/llvmstructure/13.png  
  :width: 643 px  
  :height: 421 px  
  :scale: 80 %  
  :align: center  

  Pattern matching for `ADDiu` instruction and IR node `add`  

.. _llvmstructure-dag:  
.. graphviz:: ../Fig/llvmstructure/DAG.gv  
   :caption: Detailed pattern matching for `ADDiu` instruction and IR node `add`  

During **DAG instruction selection**, the **leaf node must be a Data Node**.  
`ADDiu` follows the **L-type instruction format**, requiring the last operand  
to fit within a **16-bit signed range**.  

To enforce this constraint, `Cpu0InstrInfo.td` defines a **PatLeaf** type  
`immSExt16`, allowing the LLVM system to recognize the valid operand range.  

If the immediate value exceeds this range,  
`"isInt<16>(N->getSExtValue())"` returns `false`, and the **`ADDiu` pattern  
is not selected** during instruction selection.

Some CPUs and **floating-point units (FPUs)** include a **multiply-and-add**  
floating-point instruction, `fmadd`.  

This instruction can be represented using a **DAG list** as follows:  
`(fadd (fmul ra, rc), rb)`.  

To implement this, we define the **fmadd DAG pattern** in the instruction `.td`  
file as shown below:  

.. code:: text  

  def FMADDS : AForm_1<59, 29,  
            (ops F4RC:$FRT, F4RC:$FRA, F4RC:$FRC, F4RC:$FRB),  
            "fmadds $FRT, $FRA, $FRC, $FRB",  
            [(set F4RC:$FRT, (fadd (fmul F4RC:$FRA, F4RC:$FRC),  
                         F4RC:$FRB))]>;  

Similar to `ADDiu`, the pattern  
`[(set F4RC:$FRT, (fadd (fmul F4RC:$FRA, F4RC:$FRC), F4RC:$FRB))]`  
includes both **fmul** and **fadd** nodes.  

Now, consider the following **basic block notation IR** and **LLVM SSA IR** code:

.. code:: text

  d = a * c
  e = d + b
  ...
  
.. code-block:: llvm
  
  %d = fmul %a, %c
  %e = fadd %d, %b
  ...

The **Instruction Selection Process** will translate these two IR DAG nodes:  

  `(fmul %a, %c)`  
  `(fadd %d, %b)`  

into a **single** machine instruction DAG node:  

  `(**fmadd** %a, %c, %b)`  

instead of translating them into **two separate** machine instruction nodes  
(`**fmul**` and `**fadd**`).  

This optimization occurs **only if** `FMADDS` appears **before** `FMUL` and  
`FADD` in your `.td` file.

.. code-block:: console

  %e = fmadd %a, %c, %b
  ...

As you can see, the **IR notation representation** is easier to read than  
the **LLVM SSA IR** form.  

For this reason, this notation is occasionally used in this book.  

Now, consider the following **basic block code**:

.. code-block:: console

  a = b + c   // in notation IR form
  d = a – d
  %e = fmadd %a, %c, %b // in llvm SSA IR form

We can apply :numref:`llvmstructure-f8` **Instruction Tree Patterns**  
to generate the following **machine code**:

.. code-block:: console

  load  rb, M(sp+8); // assume b allocate in sp+8, sp is stack point register
  load  rc, M(sp+16);
  add ra, rb, rc;
  load  rd, M(sp+24);
  sub rd, ra, rd;
  fmadd re, ra, rc, rb;


Caller and Callee Saved Registers
*********************************

.. rubric:: lbdex/input/ch9_caller_callee_save_registers.cpp  
.. literalinclude:: ../lbdex/input/ch9_caller_callee_save_registers.cpp  
    :start-after: /// start  

Running the **MIPS backend** with the above input will produce the following  
result:

.. code-block:: console

  JonathantekiiMac:input Jonathan$ ~/llvm/debug/build/bin/llc 
  -O0 -march=mips -relocation-model=static -filetype=asm 
  ch9_caller_callee_save_registers.bc -o -
  	.text
  	.abicalls
  	.option	pic0
  	.section	.mdebug.abi32,"",@progbits
  	.nan	legacy
  	.file	"ch9_caller_callee_save_registers.bc"
  	.text
  	.globl	_Z6calleev
  	.align	2
  	.type	_Z6calleev,@function
  	.set	nomicromips
  	.set	nomips16
  	.ent	_Z6calleev
  _Z6callerv:                             # @_Z6callerv
  	.cfi_startproc
  	.frame	$fp,32,$ra
  	.mask 	0xc0000000,-4
  	.fmask	0x00000000,0
  	.set	noreorder
  	.set	nomacro
  	.set	noat
  # BB#0:
  	addiu	$sp, $sp, -32
  $tmp0:
  	.cfi_def_cfa_offset 32
  	sw	$ra, 28($sp)            # 4-byte Folded Spill
  	sw	$fp, 24($sp)            # 4-byte Folded Spill
  $tmp1:
  	.cfi_offset 31, -4
  $tmp2:
  	.cfi_offset 30, -8
  	move	 $fp, $sp
  $tmp3:
  	.cfi_def_cfa_register 30
  	addiu	$1, $zero, 3
  	sw	$1, 20($fp)   # store t1 to 20($fp)
  	move	 $4, $1
  	jal	_Z4add1i
  	nop
  	sw	$2, 16($fp)   # $2 : the return vaule for fuction add1()
  	lw	$1, 20($fp)   # load t1 from 20($fp)
  	subu	$1, $2, $1
  	sw	$1, 16($fp)
  	move	 $2, $1     # move result to return register $2
  	move	 $sp, $fp
  	lw	$fp, 24($sp)            # 4-byte Folded Reload
  	lw	$ra, 28($sp)            # 4-byte Folded Reload
  	addiu	$sp, $sp, 32
  	jr	$ra
  	nop
  	.set	at
  	.set	macro
  	.set	reorder
  	.end	_Z6calleev
  $func_end0:
  	.size	_Z6calleev, ($func_end0)-_Z6calleev
  	.cfi_endproc

Caller and callee saved registers definition as follows,

- If the **caller** wants to use **caller-saved registers** after calling a  
  function, it must save their contents to memory before the function call  
  and restore them afterward.  

- If the **callee** wants to use **callee-saved registers**, it must save  
  their contents to memory before using them and restore them before returning.  

According to the definition above, if a register is **not** callee-saved,  
then it must be **caller-saved**, since the callee does not restore it and  
its value may change after the function call.  

Thus, **MIPS** only defines **callee-saved registers** in `MipsCallingConv.td`,  
which can be found in `CSR_O32_SaveList` of `MipsGenRegisterInfo.inc` for the  
default ABI.  

From the assembly output, MIPS allocates the **t1** variable to register `$1`,  
which does **not** need to be spilled because `$1` is a **caller-saved register**.  

On the other hand, `$ra` is a **callee-saved register**, so it is spilled at  
the beginning of the assembly output, as `jal` uses the `$ra` register.  

For **Cpu0**, the `$lr` register corresponds to MIPS `$ra`.  
Thus, the function `setAliasRegs(MF, SavedRegs, Cpu0::LR)` is called in  
`determineCalleeSaves()` within `Cpu0SEFrameLowering.cpp` when a function  
calls another function.

Live-In and Live-Out Registers  
*******************************

As seen in the previous subsection, `$ra` is a **live-in** register because  
the return address is determined by the caller.  

Similarly, `$2` is a **live-out** register since the function’s return value  
is stored in this register. The caller retrieves the result by reading `$2`  
directly, as noted in the previous example.  

By marking **live-in** and **live-out** registers, the backend provides  
LLVM’s **middle layer** with information to eliminate redundant variable  
access instructions.  

LLVM applies **DAG analysis**, as discussed in the previous subsection,  
to perform this optimization.  

Since **C supports separate compilation**, the **live-in** and **live-out**  
information from the backend offers additional optimization opportunities  
to LLVM.  

LLVM provides the function `addLiveIn()` to mark a **live-in register**,  
but it does **not** offer a corresponding `addLiveOut()` function.  

Instead, the **MIPS backend** marks **live-out** registers by using:  

  `DAG = DAG.getCopyToReg(..., $2, ...)`  

and then returning the modified **DAG**, as all local variables cease to exist  
after the function exits.

Create Cpu0 Backend  
--------------------  

From this point onward, the **Cpu0 backend** will be created **step by step  
from scratch**.  

To help readers understand the **backend structure**, the Cpu0 example code  
can be generated **chapter by chapter** using the command provided here  
[#chapters-ex]_.  

The **Cpu0 example code (`lbdex`)** can be found near the bottom left of  
this website or downloaded from:  

  `http://jonathan2251.github.io/lbd/lbdex.tar.gz`  

Cpu0 Backend Machine ID and Relocation Records  
***********************************************

To create a **new backend**, several files in `<<llvm root dir>>` must be  
modified.  

The required modifications include adding both the **machine ID and name**,  
as well as defining **relocation records**.  

The **ELF Support** chapter provides an introduction to **relocation records**.  

The following files are modified to **add the Cpu0 backend**:

.. rubric:: lbdex/llvm/modify/llvm/config-ix.cmake
.. code:: text
  
  ...
  elseif (LLVM_NATIVE_ARCH MATCHES "cpu0")
    set(LLVM_NATIVE_ARCH Cpu0)
  ...

.. rubric:: lbdex/llvm/modify/llvm/CMakeLists.txt
.. code-block:: cmake
  
  set(LLVM_ALL_TARGETS
    ...
    Cpu0
    ...
    )

.. rubric:: lbdex/llvm/modify/llvm/include/llvm/ADT/Triple.h
.. code-block:: c++
  
  ...
  #undef mips
  #undef cpu0
  ...
  class Triple {
  public:
    enum ArchType {
      ...
      cpu0,       // For Tutorial Backend Cpu0
      cpu0el,
      ...
    };
    ...
  }

.. rubric:: lbdex/llvm/modify/llvm/include/llvm/Object/ELFObjectFile.h
.. code-block:: c++
  
  ...
  template <class ELFT>
  StringRef ELFObjectFile<ELFT>::getFileFormatName() const {
    switch (EF.getHeader()->e_ident[ELF::EI_CLASS]) {
    case ELF::ELFCLASS32:
      switch (EF.getHeader()->e_machine) {
      ...
      case ELF::EM_CPU0:	// llvm-objdump -t -r
        return "ELF32-cpu0";
      ...
    }
    ...
  }
  ...
  template <class ELFT>
  unsigned ELFObjectFile<ELFT>::getArch() const {
    bool IsLittleEndian = ELFT::TargetEndianness == support::little;
    switch (EF.getHeader()->e_machine) {
    ...
    case ELF::EM_CPU0:	// llvm-objdump -t -r
      switch (EF.getHeader()->e_ident[ELF::EI_CLASS]) {
      case ELF::ELFCLASS32:
      return IsLittleEndian ? Triple::cpu0el : Triple::cpu0;
      default:
        report_fatal_error("Invalid ELFCLASS!");
      }
    ...
    }
  }

.. rubric:: lbdex/llvm/modify/llvm/include/llvm/Support/ELF.h
.. code-block:: c++
  
  enum {
    ...
    EM_CPU0          = 999  // Document LLVM Backend Tutorial Cpu0
  };
  ...
  // Cpu0 Specific e_flags
  enum {
    EF_CPU0_NOREORDER = 0x00000001, // Don't reorder instructions
    EF_CPU0_PIC       = 0x00000002, // Position independent code
    EF_CPU0_ARCH_32   = 0x50000000, // CPU032 instruction set per linux not elf.h
    EF_CPU0_ARCH      = 0xf0000000  // Mask for applying EF_CPU0_ARCH_ variant
  };
  
  // ELF Relocation types for Mips
  enum {
  #include "ELFRelocs/Cpu0.def"
  };
  ...

.. rubric:: lbdex/llvm/modify/llvm/lib/MC/MCSubtargetInfo.cpp
.. code-block:: c++
  
  bool Cpu0DisableUnreconginizedMessage = false;
  
  void MCSubtargetInfo::InitMCProcessorInfo(StringRef CPU, StringRef FS) {
    #if 1 // Disable reconginized processor message. For Cpu0
    if (TargetTriple.getArch() == llvm::Triple::cpu0 ||
        TargetTriple.getArch() == llvm::Triple::cpu0el)
      Cpu0DisableUnreconginizedMessage = true;
    #endif
    ...
  }
  ...
  const MCSchedModel &MCSubtargetInfo::getSchedModelForCPU(StringRef CPU) const {
    ...
      #if 1 // Disable reconginized processor message. For Cpu0
      if (TargetTriple.getArch() != llvm::Triple::cpu0 &&
          TargetTriple.getArch() != llvm::Triple::cpu0el)
      #endif
    ...
  }

.. rubric:: lbdex/llvm/modify/llvm/lib/MC/SubtargetFeature.cpp
.. code-block:: c++
  
  extern bool Cpu0DisableUnreconginizedMessage; // For Cpu0
  ...
  FeatureBitset
  SubtargetFeatures::ToggleFeature(FeatureBitset Bits, StringRef Feature,
                                   ArrayRef<SubtargetFeatureKV> FeatureTable) {
    ...
      if (!Cpu0DisableUnreconginizedMessage) // For Cpu0
    ...
  }
  
  FeatureBitset
  SubtargetFeatures::ApplyFeatureFlag(FeatureBitset Bits, StringRef Feature,
                                      ArrayRef<SubtargetFeatureKV> FeatureTable) {
    ...
      if (!Cpu0DisableUnreconginizedMessage) // For Cpu0
    ...
  }
  
  FeatureBitset
  SubtargetFeatures::getFeatureBits(StringRef CPU,
                                    ArrayRef<SubtargetFeatureKV> CPUTable,
                                    ArrayRef<SubtargetFeatureKV> FeatureTable) {
    ...
      if (!Cpu0DisableUnreconginizedMessage) // For Cpu0
    ...
  }


.. rubric:: lib/object/ELF.cpp
.. code-block:: c++

  ...

  StringRef getELFRelocationTypeName(uint32_t Machine, uint32_t Type) {
    switch (Machine) {
    ...
    case ELF::EM_CPU0:
      switch (Type) {
  #include "llvm/Support/ELFRelocs/Cpu0.def"
      default:
        break;
      }
      break;
    ...
    }

.. rubric:: include/llvm/Support/ELFRelocs/Cpu0.def
.. literalinclude:: ../lbdex/llvm/modify/llvm/include/llvm/BinaryFormat/ELFRelocs/Cpu0.def

.. rubric:: lbdex/llvm/modify/llvm/lib/Support/Triple.cpp
.. code-block:: c++
  
  const char *Triple::getArchTypeName(ArchType Kind) {
    switch (Kind) {
    ...
    case cpu0:        return "cpu0";
    case cpu0el:      return "cpu0el";
    ...
    }
  }
  ...
  const char *Triple::getArchTypePrefix(ArchType Kind) {
    switch (Kind) {
    ...
    case cpu0:
    case cpu0el:      return "cpu0";
    ...
  }
  ...
  Triple::ArchType Triple::getArchTypeForLLVMName(StringRef Name) {
    return StringSwitch<Triple::ArchType>(Name)
      ...
      .Case("cpu0", cpu0)
      .Case("cpu0el", cpu0el)
      ...
  }
  ...
  static Triple::ArchType parseArch(StringRef ArchName) {
    return StringSwitch<Triple::ArchType>(ArchName)
      ...
      .Cases("cpu0", "cpu0eb", "cpu0allegrex", Triple::cpu0)
      .Cases("cpu0el", "cpu0allegrexel", Triple::cpu0el)
      ...
  }
  ...
  static Triple::ObjectFormatType getDefaultFormat(const Triple &T) {
    ...
    case Triple::cpu0:
    case Triple::cpu0el:
    ...
  }
  ...
  static unsigned getArchPointerBitWidth(llvm::Triple::ArchType Arch) {
    switch (Arch) {
    ...
    case llvm::Triple::cpu0:
    case llvm::Triple::cpu0el:
    ...
      return 32;
    }
  }
  ...
  Triple Triple::get32BitArchVariant() const {
    Triple T(*this);
    switch (getArch()) {
    ...
    case Triple::cpu0:
    case Triple::cpu0el:
    ...
      // Already 32-bit.
      break;
    }
    return T;
  }


Creating the Initial Cpu0 .td Files  
************************************

As discussed in the previous section, **LLVM** uses **target description files**  
(`.td` files) to define various components of a target's backend.  

For example, these `.td` files may describe:  
- A target's **register set**  
- Its **instruction set**  
- **Instruction scheduling** details  
- **Calling conventions**  

When the backend is compiled, **LLVM's TableGen tool** translates these `.td`  
files into **C++ source code**, which is then written to `.inc` files.  

For more details on how to use **TableGen**, please refer to [#tblgen]_.  

Each backend has its own `.td` files to define target-specific information.  
These files have a **C++-like syntax**.  

For **Cpu0**, the primary target description file is **Cpu0Other.td**,  
as shown below:  

.. rubric:: lbdex/chapters/Chapter2/Cpu0.td  
.. literalinclude:: ../lbdex/chapters/Chapter2/Cpu0.td  

.. rubric:: lbdex/chapters/Chapter2/Cpu0Other.td  
.. literalinclude:: ../lbdex/chapters/Chapter2/Cpu0Other.td  

`Cpu0Other.td` and `Cpu0.td` include several other `.td` files.  

`Cpu0RegisterInfo.td` (shown below) describes the **Cpu0 register set**.  

In this file, each register is assigned a name.  
For example, **`def PC`** defines a register named **PC**.  

In addition to **register definitions**, this file also defines **register classes**.  
A target may have multiple register classes, such as:  
- **CPURegs**  
- **SR**  
- **C0Regs**  
- **GPROut**  

The **GPROut** register class, defined in `Cpu0RegisterInfoGPROutForOther.td`,  
includes all **CPURegs** **except** `SW`, ensuring that `SW` is **not allocated**  
as an output register during the **register allocation stage**.

.. rubric:: lbdex/chapters/Chapter2/Cpu0RegisterInfo.td
.. literalinclude:: ../lbdex/chapters/Chapter2/Cpu0RegisterInfo.td

.. rubric:: lbdex/chapters/Chapter2/Cpu0RegisterInfoGPROutForOther.td
.. literalinclude:: ../lbdex/chapters/Chapter2/Cpu0RegisterInfoGPROutForOther.td


In C++, a **class** typically defines a structure to organize data and functions,  
while **definitions** allocate memory for specific instances of the class.  

For example:

.. code-block:: c++

  class Date {  // declare Date
    int year, month, day;
  }; 
  Date birthday;  // define birthday, an instance of Date

The class **Date** has the members **year**, **month**, and **day**,  
but these do not yet belong to an actual object.  

By defining an instance of **Date** called **birthday**, memory is allocated  
for a specific object, allowing you to set its **year**, **month**, and **day**.  

In `.td` files, a **class** describes the **structure** of how data is laid out,  
while **definitions** act as **specific instances** of the class.  

If you refer back to the `Cpu0RegisterInfo.td` file, you will see a class called  
**Cpu0Reg**, which is derived from the **Register** class provided by **LLVM**.  
`Cpu0Reg` **inherits all fields** from the `Register` class.  

The statement **"let HWEncoding = Enc"** assigns the field `HWEncoding`  
from the parameter `Enc`.  

Since **Cpu0 reserves 4 bits for 16 registers** in its instruction format,  
the assigned value range is **0 to 15**.  

Once values between `0` and `15` are assigned to `HWEncoding`, the **backend  
register number** is determined using **LLVM's register class functions**,  
as **TableGen** automatically sets this number.  

The **`def`** keyword is used to create instances of a class.  

In the following line, the `ZERO` register is defined as a member of the  
**Cpu0GPRReg** class:

.. code-block:: c++

  def ZERO : Cpu0GPRReg< 0, "ZERO">, DwarfRegNum<[0]>;

The **def ZERO** statement defines the name of this register.  
The parameters **<0, "ZERO">** are used to create this specific instance  
of the **Cpu0GPRReg** class.  

Thus, the field **Enc** is set to `0`, and the string **n** is set to `"ZERO"`.  

Since this register exists in the **Cpu0** namespace, it can be referenced in  
the backend C++ code using **Cpu0::ZERO**.  

### Overriding Values with `let` Expressions  

The **`let`** expressions allow overriding values initially defined in a  
superclass.  

For example, **`let Namespace = "Cpu0"`** in the **Cpu0Reg** class  
overrides the default namespace declared in the **Register** class.  

Additionally, `Cpu0RegisterInfo.td` defines **CPURegs** as an instance of  
the **RegisterClass**, a built-in **LLVM class**.  

A **RegisterClass** is essentially a **set of Register instances**,  
so **CPURegs** can be described as a set of registers.  

### Cpu0 Instruction Definition  

The **Cpu0 instruction** description file is named **Cpu0InstrInfo.td**.  
Its contents are as follows:  

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td  
.. literalinclude:: ../lbdex/chapters/Chapter2/Cpu0InstrInfo.td  

The `Cpu0InstrFormats.td` file is included in **Cpu0InstrInfo.td**, as shown:  

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrFormats.td  
.. literalinclude:: ../lbdex/chapters/Chapter2/Cpu0InstrFormats.td  

### Expanding `ADDiu`  

`ADDiu` is an instance of the **ArithLogicI** class, which inherits from `FL`.  
It can be further expanded to retrieve its member values as follows:

.. code:: text

  def ADDiu   : ArithLogicI<0x09, "addiu", add, simm16, immSExt16, CPURegs>;
  
  /// Arithmetic and logical instructions with 2 register operands.
  class ArithLogicI<bits<8> op, string instr_asm, SDNode OpNode,
            Operand Od, PatLeaf imm_type, RegisterClass RC> :
    FL<op, (outs GPROut:$ra), (ins RC:$rb, Od:$imm16),
     !strconcat(instr_asm, "\t$ra, $rb, $imm16"),
     [(set GPROut:$ra, (OpNode RC:$rb, imm_type:$imm16))], IIAlu> {
    let isReMaterializable = 1;
  }
  
So,  

.. code-block:: console  

  op = 0x09  
  instr_asm = "addiu"  
  OpNode = add  
  Od = simm16  
  imm_type = immSExt16  
  RC = CPURegs  

### Expanding `.td` Files: Key Principles  

- **`let`**: Overrides an existing field from the parent class.  

  - Example: `let isReMaterializable = 1;`  
    This overrides `isReMaterializable` from the `Instruction` class in `Target.td`.  

- **Declaration**: Defines a new field for the class.  

  - Example: `bits<4> ra;`  
    This declares the `ra` field in the `FL` class.  

### ADDiu Expansion Details  

The details of expansion are shown in the following table:  

.. table:: ADDiu Expansion Part I

  =========  ======================  ========================================================
  ADDiu      ArithLogicI             FL                                                      
  =========  ======================  ========================================================
  0x09       op = 0x09               Opcode = 0x09;
  addiu      instr_asm = “addiu”     (outs GPROut:$ra);
                                     !strconcat("addiu", "\t$ra, $rb, $imm16");
  add        OpNode = add            [(set GPROut:$ra, (add CPURegs:$rb, immSExt16:$imm16))]
  simm16     Od = simm16             (ins CPURegs:$rb, simm16:$imm16);
  immSExt16  imm_type = immSExt16    Inst{15-0} = imm16;
  CPURegs    RC = CPURegs
             isReMaterializable=1;   Inst{23-20} = ra;
                                     Inst{19-16} = rb;
  =========  ======================  ========================================================

.. table:: ADDiu Expansion part II

  =============================================================  =====================
  Cpu0Inst                                                       instruction
  =============================================================  =====================
  Namespace = "Cpu0"                                             Uses = []; ...
  Inst{31-24} = 0x09;                                            Size = 0; ...
  OutOperandList = GPROut:$ra;
  InOperandList  = CPURegs:$rb,simm16:$imm16;
  AsmString = "addiu\t$ra, $rb, $imm16"
  pattern = [(set GPROut:$ra, (add RC:$rb, immSExt16:$imm16))]
  Itinerary = IIAlu
  TSFlags{3-0} = FrmL.value
  DecoderNamespace = "Cpu0"
  =============================================================  =====================
  
The `.td` file expansion process can be cumbersome.  
Similarly, **LD** and **ST** instruction definitions can be expanded in the  
same manner.  

Note the **Pattern**:  

  `[(set GPROut:$ra, (add RC:$rb, immSExt16:$imm16))]`  

which includes the keyword **"add"**.  

The **ADDiu** instruction with **"add"** was used in the  
*Instruction Selection* subsection of the previous section.  

### Cpu0 Scheduling Information  

The `Cpu0Schedule.td` file includes details about **function units** and  
**pipeline stages**, as shown below:  

.. rubric:: lbdex/chapters/Chapter2/Cpu0Schedule.td  
.. literalinclude:: ../lbdex/chapters/Chapter2/Cpu0Schedule.td  

### Writing the CMake File  

The `Target/Cpu0` directory contains the `CMakeLists.txt` file.  
Its contents are as follows:  

.. rubric:: lbdex/chapters/Chapter2/CMakeLists.txt  
.. literalinclude:: ../lbdex/chapters/Chapter2/CMakeLists.txt  

**CMakeLists.txt** provides **build instructions** for **CMake**.  
Comments in this file are prefixed with **#**.  

The `"tablegen("` function in `CMakeLists.txt` is defined in:  

  `cmake/modules/TableGen.cmake`  

as shown below:

.. rubric:: llvm/cmake/modules/TableGen.cmake
.. code:: text

  function(tablegen project ofn)
    ...
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}.tmp
      # Generate tablegen output in a temporary file.
      COMMAND ${${project}_TABLEGEN_EXE} ${ARGN} -I ${CMAKE_CURRENT_SOURCE_DIR}
    ...
  endfunction()
  ...
  macro(add_tablegen target project)
    ...
    if(LLVM_USE_HOST_TOOLS)
      if( ${${project}_TABLEGEN} STREQUAL "${target}" )
        if (NOT CMAKE_CONFIGURATION_TYPES)
          set(${project}_TABLEGEN_EXE "${LLVM_NATIVE_BUILD}/bin/${target}")
        else()
          set(${project}_TABLEGEN_EXE "${LLVM_NATIVE_BUILD}/Release/bin/${target}")
        endif()
    ...
  endmacro()

.. rubric:: llvm/utils/TableGen/CMakeLists.txt
.. code-block:: cmake

  add_tablegen(llvm-tblgen LLVM
    ...
  )

The `"add_tablegen"` function in `llvm/utils/TableGen/CMakeLists.txt`  
makes `"tablegen("` in `Cpu0 CMakeLists.txt` an alias for **llvm-tblgen**  
(where `${project} = LLVM` and `${project}_TABLEGEN_EXE = llvm-tblgen`).  

The following elements define the **Cpu0CommonTableGen** target,  
which generates the output files **Cpu0Gen*.inc**:  

- `"tablegen("`  
- `"add_public_tablegen_target(Cpu0CommonTableGen)"`  
- The following additional code

.. rubric:: llvm/cmake/modules/TableGen.cmake
.. code:: text

  function(tablegen project ofn)
    ...
    set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)
    ...
  endfunction()

  # Creates a target for publicly exporting tablegen dependencies.
  function(add_public_tablegen_target target)
    ...
    add_custom_target(${target}
      DEPENDS ${TABLEGEN_OUTPUT})
    ...
  endfunction()

Since the **llvm-tblgen** executable is built before compiling any LLVM backend  
source code, it is always available to handle TableGen requests during the build  
process.  

This book introduces backend source code **incrementally**, adding code  
**chapter by chapter** based on function.  

### Understanding Source Code  

- **Don't try to understand everything from the text alone**—  
  the code added in each chapter serves as **learning material** too.  
- **Conceptual understanding** of computer-related knowledge can be gained  
  without reading source code.  
- However, **when implementing based on existing open-source software,  
  reading source code is essential**.  
- **Documentation cannot fully replace source code** in programming.  
- **Reading source code is a valuable skill in open-source development**.  

### CMakeLists.txt in Subdirectories  

The `CMakeLists.txt` files exist in the **MCTargetDesc** and **TargetInfo**  
subdirectories.  

These files instruct LLVM to generate the **Cpu0Desc** and **Cpu0Info**  
libraries, respectively.  

After building, you will find three libraries in the `lib/` directory  
of your build folder:  

- **libLLVMCpu0CodeGen.a**  
- **libLLVMCpu0Desc.a**  
- **libLLVMCpu0Info.a**  

For more details, refer to *"Building LLVM with CMake"* [#cmake]_.  

Target Registration  
********************

You must **register your target** with the **TargetRegistry**.  
After registration, LLVM tools can **identify and use** your target at runtime.  

Although the **TargetRegistry** can be used directly, most targets  
utilize **helper templates** to simplify the process.  

All targets should declare a **global Target object**, which represents the  
target during registration.  

Then, in the target's **TargetInfo library**, the target should define this  
object and use the **RegisterTarget** template to register it.  

For example, the file `TargetInfo/Cpu0TargetInfo.cpp` registers  
**TheCpu0Target** for **big-endian** and **TheCpu0elTarget** for **little-endian**,  
as shown below:

.. rubric:: lbdex/chapters/Chapter2/Cpu0.h
.. literalinclude:: ../lbdex/chapters/Chapter2/Cpu0.h

.. rubric:: lbdex/chapters/Chapter2/TargetInfo/Cpu0TargetInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/TargetInfo/Cpu0TargetInfo.cpp

.. rubric:: lbdex/chapters/Chapter2/TargetInfo/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/TargetInfo/CMakeLists.txt

The files **Cpu0TargetMachine.cpp** and **MCTargetDesc/Cpu0MCTargetDesc.cpp**  
currently define only an **empty initialization function**,  
as no components are being registered at this stage.

.. rubric:: lbdex/chapters/Chapter2/Cpu0TargetMachine.cpp
.. literalinclude:: ../lbdex/chapters/Chapter2/Cpu0TargetMachine.cpp

.. rubric:: lbdex/chapters/Chapter2/MCTargetDesc/Cpu0MCTargetDesc.h
.. literalinclude:: ../lbdex/chapters/Chapter2/MCTargetDesc/Cpu0MCTargetDesc.h

.. rubric:: lbdex/chapters/Chapter2/MCTargetDesc/Cpu0MCTargetDesc.cpp
.. literalinclude:: ../lbdex/chapters/Chapter2/MCTargetDesc/Cpu0MCTargetDesc.cpp

.. rubric:: lbdex/chapters/Chapter2/MCTargetDesc/CMakeLists.txt
.. literalinclude:: ../lbdex/chapters/Chapter2/MCTargetDesc/CMakeLists.txt


For reference, see *"Target Registration"* [#target-reg]_.  

Build Libraries and `.td` Files  
*********************************

Build steps: https://github.com/Jonathan2251/lbd/blob/master/README.md.
Illustrated in :ref:`Appendix A <sec-appendix-installing>`.  

We set the LLVM source code in:  

  `/Users/Jonathan/llvm/debug/llvm`  

and perform a debug build in:  

  `/Users/Jonathan/llvm/debug/build`  

.. note::
   
   Remember to build both clang and llvm(llc) in debug mode to generate the 
   destination registers in LLVM IR and to print DAGs with identifiers such as 
   t0, t1, ... (e.g. "t25: i32 = add t22, t28", 
   "t26: i32 = mul t25, Constant:i32<12>").

For details on how to build LLVM, refer to [#clang]_.  

In **Appendix A**, we create a copy of the LLVM source directory:  

  `/Users/Jonathan/llvm/debug/llvm`  

to:  

  `/Users/Jonathan/llvm/test/llvm`  

for developing the **Cpu0 target backend**.  

- The `llvm/` directory contains the **source code**.  
- The `build/` directory is used for the **debug build**.

### Modifying LLVM for Cpu0  

Beyond `llvm/lib/Target/Cpu0`, several files have been modified to support  
the **new Cpu0 target**. These modifications include:  

- **Adding the target's ID and name**  
- **Defining relocation records** (as discussed in an earlier section)  

To update your LLVM working copy and apply the modifications, use:  

.. code-block:: console  

   cp -rf lbdex/llvm/modify/llvm/* <yourllvm/workingcopy/sourcedir>/  

.. code-block:: console

  118-165-78-230:lbd Jonathan$ pwd
  /Users/Jonathan/git/lbd
  118-165-78-230:lbd Jonathan$ cp -rf lbdex/llvm/modify/llvm/* ~/llvm/test/llvm/.
  118-165-78-230:lbd Jonathan$ grep -R "cpu0" ~/llvm/test/llvm/include
  llvm/cmake/config-ix.cmake:elseif (LLVM_NATIVE_ARCH MATCHES "cpu0")
  llvm/include/llvm/ADT/Triple.h:#undef cpu0
  llvm/include/llvm/ADT/Triple.h:    cpu0,       // For Tutorial Backend Cpu0
  llvm/include/llvm/ADT/Triple.h:    cpu0el,
  llvm/include/llvm/Support/ELF.h:  EF_CPU0_ARCH_32R2 = 0x70000000, // cpu032r2
  llvm/include/llvm/Support/ELF.h:  EF_CPU0_ARCH_64R2 = 0x80000000, // cpu064r2
  ...

Next, configure the Cpu0 example code for **Chapter 2** as follows:

.. rubric:: ~/llvm/test/llvm/lib/Target/Cpu0/Cpu0SetChapter.h
.. code-block:: c++

  #define CH       CH2

In addition to configuring the chapter as shown above,  
I provide **gen-chapters.sh**, which allows you to retrieve  
the code for each chapter as follows:

.. code-block:: console

  118-165-78-230:lbdex Jonathan$ pwd
  /Users/Jonathan/git/lbd/lbdex
  118-165-78-230:lbdex Jonathan$ bash gen-chapters.sh
  118-165-78-230:lbdex Jonathan$ ls chapters
  Chapter10_1	Chapter11_2	Chapter2	Chapter3_2...
  Chapter11_1	Chapter12_1	Chapter3_1	Chapter3_3...


Now, run the ``cmake`` and ``make`` commands to build `.td` files.  
(The following `cmake` command is based on my setup.)

.. code-block:: console

  118-165-78-230:build Jonathan$ cmake -DCMAKE_CXX_COMPILER=clang++ 
  -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Debug -G "Unix Makefiles" ../llvm/
  
  -- Targeting Cpu0 
  ...
  -- Targeting XCore 
  -- Configuring done 
  -- Generating done 
  -- Build files have been written to: /Users/Jonathan/llvm/test/build
  
  118-165-78-230:build Jonathan$ make -j4
 
  118-165-78-230:build Jonathan$ 

After building, you can run the command ``llc --version``  
to verify that the Cpu0 backend is available.

.. code-block:: console

  118-165-78-230:build Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc --version
  LLVM (http://llvm.org/):
  ...
    Registered Targets: 
    arm      - ARM 
    ...
    cpp      - C++ backend 
    cpu0     - Cpu0 
    cpu0el   - Cpu0el 
  ...

The command ``llc --version`` will display the registered targets  
**"cpu0"** and **"cpu0el"**,  
as defined in `TargetInfo/Cpu0TargetInfo.cpp`  
from the previous section, *Target Registration* [#asadasd]_.  

Now, let's build the `lbdex/chapters/Chapter2` code as follows:

.. code-block:: console

  118-165-75-57:test Jonathan$ pwd
  /Users/Jonathan/test
  118-165-75-57:test Jonathan$ cp -rf lbdex/Cpu0 ~/llvm/test/llvm/lib/Target/.

  118-165-75-57:test Jonathan$ cd ~/llvm/test/build
  118-165-75-57:build Jonathan$ pwd
  /Users/Jonathan/llvm/test/build
  118-165-75-57:build Jonathan$ rm -rf *
  118-165-75-57:build Jonathan$ cmake -DCMAKE_CXX_COMPILER=clang++ 
  -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=Cpu0 
  -G "Unix Makefiles" ../llvm/
  ...
  -- Targeting Cpu0
  ...
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /Users/Jonathan/llvm/test/build

To save time, we build only the Cpu0 target using the option:  

``-DLLVM_TARGETS_TO_BUILD=Cpu0``  

After the build, you can find the generated `*.inc` files in:  

``/Users/Jonathan/llvm/test/build/lib/Target/Cpu0``  

as shown below:

.. rubric:: build/lib/Target/Cpu0/Cpu0GenRegisterInfo.inc
.. code-block:: c++

  namespace Cpu0 {
  enum {
    NoRegister,
    AT = 1,
    EPC = 2,
    FP = 3,
    GP = 4,
    HI = 5,
    LO = 6,
    LR = 7,
    PC = 8,
    SP = 9,
    SW = 10,
    ZERO = 11,
    A0 = 12,
    A1 = 13,
    S0 = 14,
    S1 = 15,
    T0 = 16,
    T1 = 17,
    T9 = 18,
    V0 = 19,
    V1 = 20,
    NUM_TARGET_REGS 	// 21
  };
  }
  ...

These `*.inc` files are generated by `llvm-tblgen` in the  
`build/lib/Target/Cpu0` directory, using the Cpu0 backend `*.td` files  
as input.  

The `llvm-tblgen` tool is invoked by **tablegen**  
in `/Users/Jonathan/llvm/test/llvm/lib/Target/Cpu0/CMakeLists.txt`.  

These `*.inc` files are later included in Cpu0 backend `.cpp` or `.h` files  
and compiled into `.o` files.  

**TableGen** is a crucial tool, as discussed earlier in the  
*".td: LLVM’s Target Description Files"* section of this chapter.  
For reference, the TableGen documentation is available here:  
[#tblgen]_ [#tblgen-langintro]_ [#tblgen-langref]_.  

Now, try running the `llc` command to compile the input file `ch3.cpp`:

.. rubric:: lbdex/input/ch3.cpp
.. literalinclude:: ../lbdex/input/ch3.cpp
    :start-after: /// start


First step, compile it with clang and get output ch3.bc as follows,

.. code-block:: console

  118-165-78-230:input Jonathan$ pwd
  /Users/Jonathan/git/lbd/lbdex/input
  118-165-78-230:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch3.cpp -emit-llvm -o ch3.bc

As shown above, compile C to `.bc` using:  

``clang -target mips-unknown-linux-gnu``  

since Cpu0 borrows its ABI from MIPS.  

Next, convert the bitcode (`.bc`) to a human-readable text format as follows:

.. code-block:: console

  118-165-78-230:test Jonathan$ llvm-dis ch3.bc -o -
  
  // ch3.ll
  ; ModuleID = 'ch3.bc' 
  target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f3
  2:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:6
  4-S128" 
  target triple = "mips-unknown-linux-gnu" 
  
  define i32 @main() nounwind uwtable { 
    %1 = alloca i32, align 4 
    store i32 0, i32* %1 
    ret i32 0 
  }

Now, compiling `ch3.bc` will result in the following error message:  

.. code-block:: console  

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/  
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc -o  
  ch3.cpu0.s  
  ...  
  ... Assertion `target.get() && "Could not allocate target machine!"' failed  
  ...  

At this point, we have completed *Target Registration* for the Cpu0 backend.  
The `llc` compiler command now recognizes the Cpu0 backend.  

Currently, we have only defined the target `.td` files (`Cpu0.td`,  
`Cpu0Other.td`, `Cpu0RegisterInfo.td`, etc.).  
According to the LLVM structure, we need to define our target machine  
and include these `.td` files.  

The error message indicates that the target machine is not yet defined.  
This book follows a step-by-step approach to backend development.  
You can review the **hundreds** of lines in the Chapter 2 example code  
to understand how *Target Registration* is implemented.  


Debug options
-------------

Options for `llc` Debugging  
****************************

Run the following command to see hidden `llc` options:  

``llc --help-hidden``  

The following `llc` options require an input `.bc` or `.ll` file:  

- `-debug`  
- `-debug-pass=Structure`  
- `-print-after-all`, `-print-before-all`  
- `-print-before="pass"` and `-print-after="pass"`  

  Example:  
  ``-print-before="postra-machine-sink" -print-after="postra-machine-sink"``  

  The pass name can be obtained as follows:

.. code-block:: console

  CodeGen % pwd
  ~/llvm/debug/llvm/lib/CodeGen
  CodeGen % grep -R "INITIALIZE_PASS" |grep sink
  ./MachineSink.cpp:INITIALIZE_PASS(PostRAMachineSinking, "postra-machine-sink",

- `-view-dag-combine1-dags`  
  Displays the DAG after being built, before the first optimization pass.  

- `-view-legalize-dags`  
  Displays the DAG before legalization.  

- `-view-dag-combine2-dags`  
  Displays the DAG before the second optimization pass.  

- `-view-isel-dags`  
  Displays the DAG before the Select phase.  

- `-view-sched-dags`  
  Displays the DAG before scheduling.  

- `-march=<string>`  
  Specifies the target architecture (e.g., `-march=mips`).  

- `-relocation-model=static/pic`  
  Sets the relocation model.  

- `-filetype=asm/obj`  
  Specifies the output file type (assembly or object).  

You can use `F.dump()` in the code, where `F` is an instance of the `Function`  
class, to inspect transformations in `llvm/lib/Transformation`.  


`opt` Debugging Options  
***********************

Check available options using:  

``opt --help-hidden``  

Refer to LLVM passes documentation [#llvm-passes]_. Examples:  

- `opt -dot-cfg input.ll`  
  Prints the CFG of a function to a `.dot` file.  

- `-dot-cfg-only`  
  Prints the CFG to a `.dot` file without function bodies.



.. [#java-cpp] https://en.wikipedia.org/wiki/Comparison_of_Java_and_C%2B%2B

.. [#cpu0-chinese] Original Cpu0 architecture and ISA details (Chinese). http://ccckmit.wikidot.com/ocs:cpu0

.. [#cpu0-english] English translation of Cpu0 description. http://translate.google.com.tw/translate?js=n&prev=_t&hl=zh-TW&ie=UTF-8&layout=2&eotf=1&sl=zh-CN&tl=en&u=http://ccckmit.wikidot.com/ocs:cpu0

.. [#lb-note] The difference between LB and LBu is signed and unsigned byte value expand to a word size. For example, After LB Ra, [Rb+Cx], Ra is 0xffffff80(= -128) if byte [Rb+Cx] is 0x80; Ra is 0x0000007f(= 127) if byte [Rb+Cx] is 0x7f. After LBu Ra, [Rb+Cx], Ra is 0x00000080(= 128) if byte [Rb+Cx] is 0x80; Ra is 0x0000007f(= 127) if byte [Rb+Cx] is 0x7f. Difference between LH and LHu is similar.

.. [#u-note] The only difference between ADDu instruction and the ADD instruction is that the ADDU instruction never causes an Integer Overflow exception. SUBu and SUB is similar.

.. [#cond-note] CMP is signed-compare while CMPu is unsigned. Conditions include the following comparisons: >, >=, ==, !=, <=, <. SW is actually set by the subtraction of the two register operands, and the flags indicate which conditions are present.

.. [#sra-note] Rb '>> Cx, Rb '>> Rc: Shift with signed bit remain.

.. [#call-note] jsub cx is direct call for 24 bits value of cx while jalr $rb is indirect call for 32 bits value of register $rb.

.. [#jr-note] Both JR and RET has same opcode (actually they are the same instruction for Cpu0 hardware). When user writes "jr $t9" meaning it jumps to address of register $t9; when user writes "jr $lr" meaning it jump back to the caller function (since $lr is the return address). For user read ability, Cpu0 prints "ret $lr" instead of "jr $lr".

.. [#aosa-book] Chris Lattner, **LLVM**. Published in The Architecture of Open Source Applications. http://www.aosabook.org/en/llvm.html

.. [#chapters-ex] http://jonathan2251.github.io/lbd/doc.html#generate-cpu0-document

.. [#codegen] http://llvm.org/docs/CodeGenerator.html

.. [#langref] http://llvm.org/docs/LangRef.html

.. [#cmu-rac] https://www.cs.cmu.edu/afs/cs/academic/class/15745-s16/www/lectures/L23-Register-Coalescing.pdf

.. [#ra-wiki] https://en.wikipedia.org/wiki/Register_allocation

.. [#dragonbooks-10.2.3] Refer section 10.2.3 of book Compilers: Principles, 
    Techniques, and Tools (2nd Edition) 

.. [#gnu] https://en.wikipedia.org/wiki/GNU_Compiler_Collection

.. [#gcc-frontend] https://en.wikipedia.org/wiki/GNU_Compiler_Collection#Front_ends

.. [#generic] https://gcc.gnu.org/onlinedocs/gccint/GENERIC.html

.. [#gimple] https://gcc.gnu.org/onlinedocs/gccint/GIMPLE.html

.. [#rtl] https://gcc.gnu.org/onlinedocs/gccint/RTL.html

.. [#md] https://gcc.gnu.org/onlinedocs/gccint/Machine-Desc.html#Machine-Desc

.. [#llvm-ir-vs-gimple] https://stackoverflow.com/questions/40799696/how-is-gcc-ir-different-from-llvm-ir/40802063

.. [#null_pointer_ex] https://github.com/Jonathan2251/lbd/tree/master/References/null_pointer.cpp is an example.

.. [#null_pointer]
    Dereferencing a NULL Pointer: 
    contrary to popular belief, dereferencing a null pointer in C is undefined. 
    It is not defined to trap, and if you mmap a page at 0, it is not defined to access that page. 
    This falls out of the rules that forbid dereferencing wild pointers and the use of NULL as a sentinel,
    from http://blog.llvm.org/2011/05/what-every-c-programmer-should-know.html.
    As link, https://blog.llvm.org/2011/05/what-every-c-programmer-should-know_14.html.
    In this case, the developer forgot to call "set", did not crash with a null pointer dereference, 
    and their code broke when someone else did a debug build.

.. [#cfg-wiki] https://en.wikipedia.org/wiki/Control-flow_graph

.. [#dragonbooks-8.5] Refer section 8.5 of book Compilers: Principles, 
    Techniques, and Tools (2nd Edition) 

.. [#cmake] http://llvm.org/docs/CMake.html

.. [#target-reg] http://llvm.org/docs/WritingAnLLVMBackend.html#target-registration

.. [#clang] http://clang.llvm.org/get_started.html

.. [#asadasd] http://jonathan2251.github.io/lbd/llvmstructure.html#target-registration

.. [#tblgen] http://llvm.org/docs/TableGen/index.html

.. [#tblgen-langintro] http://llvm.org/docs/TableGen/LangIntro.html

.. [#tblgen-langref] http://llvm.org/docs/TableGen/LangRef.html

.. [#llvm-passes] https://llvm.org/docs/Passes.html
