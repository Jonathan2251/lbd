.. _sec-asm:

Assembler
=========

.. contents::
   :local:
   :depth: 4

This chapter covers the assembly programming support in the Cpu0 backend.

When it comes to assembly language programming, there are two types of usage in
C/C++ as follows:

.. rubric:: ordinary assembly
.. code-block:: cpp-objdump

  asm("ld	$2, 8($sp)");

.. rubric:: inline assembly
.. code-block:: cpp-objdump

  int foo = 10;
  const int bar = 15;
  
  __asm__ __volatile__("addu %0,%1,%2"
                       :"=r"(foo) // 5
                       :"r"(foo), "r"(bar)
                       );

In LLVM, the first method is supported by the LLVM AsmParser, and the second by
the inline assembly handler.  

With AsmParser and inline assembly support in the Cpu0 backend, we can hand-code
assembly language in a C/C++ file and translate it into an object file (in ELF
format).

AsmParser support
-----------------

This section lists all the AsmParser code for the Cpu0 backend, with only brief
explanations. Please refer to [#llvmas]_ for a more detailed explanation of AsmParser.

Running Chapter10_1/ with ch11_1.cpp will produce the following error message.

Run Chapter10_1/ with ch11_1.cpp will get the following error message.

.. rubric:: lbdex/input/ch11_1.cpp
.. literalinclude:: ../lbdex/input/ch11_1.cpp
    :start-after: /// start

.. code-block:: console

  JonathantekiiMac:input Jonathan$ clang -c ch11_1.cpp -emit-llvm -o 
  ch11_1.bc
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/bin/llc
  -march=cpu0 -relocation-model=pic -filetype=obj ch11_1.bc 
  -o ch11_1.cpu0.o
  LLVM ERROR: Inline asm not supported by this streamer because we don't have 
  an asm parser for this target
  
Since we haven't implemented the Cpu0 assembler, the error message shown above
occurs. Cpu0 can translate LLVM IR into assembly and object files directly, but
it cannot translate hand-written assembly instructions into an object file.

Chapter11_1/ includes the AsmParser implementation as follows:

.. rubric:: lbdex/chapters/Chapter11_1/AsmParser/Cpu0AsmParser.cpp
.. literalinclude:: ../lbdex/Cpu0/AsmParser/Cpu0AsmParser.cpp

.. rubric:: lbdex/chapters/Chapter11_1/AsmParser/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/AsmParser/CMakeLists.txt

The `Cpu0AsmParser.cpp` file contains around one thousand lines of code that
handle assembly language parsing. With a little patience, you can understand it.

To enable building the files in the AsmParser directory, modify
`CMakeLists.txt` as follows:

.. rubric:: lbdex/chapters/Chapter11_1/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH11_1 1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH11_1 2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH11_1 3
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter11_1/Cpu0Asm.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0Asm.td

.. rubric:: lbdex/chapters/Chapter11_1/Cpu0RegisterInfoGPROutForAsm.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfoGPROutForAsm.td


The `CMakeLists.txt` modification shown above generates `Cpu0GenAsmMatcher.inc`,
which is used by `Cpu0AsmParser.cpp`.

`Cpu0Asm.td` includes `Cpu0RegisterInfoGPROutForAsm.td`, which defines `GPROut`
as mapping to `CPURegs`. Meanwhile, `Cpu0Other.td` includes
`Cpu0RegisterInfoGPROutForOther.td`, which defines `GPROut` to map to `CPURegs`
excluding `SW`.

`Cpu0Other.td` is used when translating LLVM IR to Cpu0 instructions. In this
case, the `SW` register is reserved for storing CPU status and must not be
allocated as a general-purpose register.

For example, if `GPROut` includes `SW`, compiling the C statement `a = (b & c);`
might generate the instruction `and $sw, $1, $2`. This would overwrite the
interrupt status in `$sw`.

However, when programming in assembly, instructions like `andi $sw, $sw, 0xffdf`
are allowed. This kind of assembly is accepted, and the Cpu0 backend considers
it safe. An assembler programmer may use `andi $sw, $sw, 0xffdf` to disable
trace debug messages, and `ori $sw, $sw, 0x0020` to enable them.

Additionally, interrupt bits can also be enabled or disabled using `ori` and
`andi` instructions.

The `EPC` must be set to `CPURegs` as shown below. Otherwise,
`MatchInstructionImpl()` in `MatchAndEmitInstruction()` will fail for the
instruction `asm("mfc0 $pc, $epc");`.

.. rubric:: lbdex/chapters/Chapter2/Cpu0RegisterInfo.td
.. code-block:: c++

  def CPURegs : RegisterClass<"Cpu0", [i32], 32, (add
    ...
    , PC, EPC)>;

.. rubric:: lbdex/chapters/Chapter11_1/Cpu0.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0.td
    :start-after: #if CH >= CH11_1 1
    :end-before: //#endif
    
.. code-block:: c++

  def Cpu0 : Target {
    ...

.. literalinclude:: ../lbdex/Cpu0/Cpu0.td
    :start-after: #if CH >= CH11_1 2
    :end-before: //#endif

.. code-block:: c++

  }
  
.. rubric:: lbdex/chapters/Chapter11_1/Cpu0InstrFormats.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrFormats.td
    :start-after: #if CH >= CH11_1
    :end-before: //#endif
  
.. rubric:: lbdex/chapters/Chapter11_1/Cpu0InstrInfo.td
.. code-block:: c++

  def Cpu0MemAsmOperand : AsmOperandClass {
    let Name = "Mem";
    let ParserMethod = "parseMemOperand";
  }
  
  // Address operand
  def mem : Operand<i32> {
    ...
    let ParserMatchClass = Cpu0MemAsmOperand;
  }
  ...

.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH11_1 3
    :end-before: //#endif

In the `Cpu0InstrInfo.td` file, the directive **let ParserMethod = 
"parseMemOperand"** declares that the method `parseMemOperand()` will be used
to handle the **mem** operand in Cpu0 instructions such as `ld` and `st`.

For example, in the instruction `ld $2, 4($sp)`, the **mem** operand is `4($sp)`.

Together with the directive **let ParserMatchClass = Cpu0MemAsmOperand;**, LLVM
will invoke the `parseMemOperand()` function in `Cpu0AsmParser.cpp` whenever it
encounters a **mem** operand like `4($sp)` in assembly code.

With the above **let** assignments, TableGen will generate the corresponding
structure and functions in `Cpu0GenAsmMatcher.inc`.

.. rubric:: build/lib/Target/Cpu0/Cpu0GenAsmMatcher.inc
.. code-block:: c++
  
    enum OperandMatchResultTy {
      MatchOperand_Success,    // operand matched successfully
      MatchOperand_NoMatch,    // operand did not match
      MatchOperand_ParseFail   // operand matched but had errors
    };
    OperandMatchResultTy MatchOperandParserImpl(
      OperandVector &Operands,
      StringRef Mnemonic);
    OperandMatchResultTy tryCustomParseOperand(
      OperandVector &Operands,
      unsigned MCK);
  ...
  Cpu0AsmParser::OperandMatchResultTy Cpu0AsmParser::
  tryCustomParseOperand(OperandVector &Operands,
              unsigned MCK) {
  
    switch(MCK) {
    case MCK_Mem:
      return parseMemOperand(Operands);
    default:
      return MatchOperand_NoMatch;
    }
    return MatchOperand_NoMatch;
  }
  
  Cpu0AsmParser::OperandMatchResultTy Cpu0AsmParser::
  MatchOperandParserImpl(OperandVector &Operands,
               StringRef Mnemonic) {
    ...
  }
  
  /// MatchClassKind - The kinds of classes which participate in
  /// instruction matching.
  enum MatchClassKind {
    ...
    MCK_Mem, // user defined class 'Cpu0MemAsmOperand'
    ...
  };


The three pseudo instruction definitions in `Cpu0InstrInfo.td`, such as
`LoadImm32Reg`, are handled by `Cpu0AsmParser.cpp` as follows:

.. rubric:: lbdex/chapters/Chapter11_1/AsmParser/Cpu0AsmParser.cpp
.. literalinclude:: ../lbdex/Cpu0/AsmParser/Cpu0AsmParser.cpp
    :start-after: //@1 {
    :end-before: //@1 }
.. literalinclude:: ../lbdex/Cpu0/AsmParser/Cpu0AsmParser.cpp
    :start-after: //@2 {
    :end-before: //@2 }

.. code-block:: c++

    ...
  }

Finally, remember that the `CPURegs` list below must follow the order of register
numbers, because the AsmParser uses this order when encoding register numbers.

.. rubric:: lbdex/chapters/Chapter11_1/Cpu0RegisterInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.td
    :start-after: //@Registers
    :end-before: //#if CH >= CH4_1 2


Run `Chapter11_1/` with `ch11_1.cpp` to get the correct result as shown below:

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/bin/llc
  -march=cpu0 -relocation-model=pic -filetype=obj ch11_1.bc -o 
  ch11_1.cpu0.o
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/bin/
  llvm-objdump -d ch11_1.cpu0.o
  
  ch11_1.cpu0.o:  file format ELF32-unknown
  
  Disassembly of section .text:
  .text:
         0:	01 2d 00 08                                  	ld	$2, 8($sp)
         4:	02 0d 00 04                                  	st	$zero, 4($sp)
         8:	09 30 00 00                                  	addiu	$3, $zero, 0
         c:	13 31 20 00                                  	add	$3, $1, $2
        10:	14 32 30 00                                  	sub	$3, $2, $3
        ...

The instructions `cmp` and `jeg` display the `$sw` register explicitly in both
assembly and disassembly. You can modify the code in the AsmParser and
Disassembler (discussed in the last chapter) to hide `$sw` in these instructions
—for example, displaying `jeq 20` instead of `jeq $sw, 20`.

Both `AsmParser` and `Cpu0AsmParser` inherit from `MCAsmParser` as follows:

.. rubric:: llvm/lib/MC/MCParser/AsmParser.cpp
.. code-block:: c++

  class AsmParser : public MCAsmParser {
    ...
  }
  ...

`AsmParser` will call the `ParseInstruction()` and `MatchAndEmitInstruction()`
functions of `Cpu0AsmParser` as follows:

.. rubric:: llvm/lib/MC/MCParser/AsmParser.cpp
.. code-block:: c++

  bool AsmParser::parseStatement(ParseStatementInfo &Info) {
    ...
    // Directives start with "."
    if (IDVal[0] == '.' && IDVal != ".") {
      // First query the target-specific parser. It will return 'true' if it
      // isn't interested in this directive.
      if (!getTargetParser().ParseDirective(ID))
        return false;
      ...
    }
    ...
    bool HadError = getTargetParser().ParseInstruction(IInfo, OpcodeStr, IDLoc,
                                                       Info.ParsedOperands);
    ...
    // If parsing succeeded, match the instruction.
    if (!HadError) {
      unsigned ErrorInfo;
      getTargetParser().MatchAndEmitInstruction(IDLoc, Info.Opcode,
                                                Info.ParsedOperands, Out,
                                                ErrorInfo, ParsingInlineAsm);
    }
    ...
  }


Assembler structure
-------------------

Run `llc` with the option `-debug-only=asm-matcher,cpu0-asm-parser` to observe
how the Cpu0 assembler works.

The `AsmParser` directory handles the translation from assembly to object files.

The assembling Data Flow Diagram (DFD) is shown in :numref:`asm-f1` and
:numref:`asm-f2`.

.. _asm-f1: 
.. graphviz:: ../Fig/asm/asmDfd.gv
   :caption: Assembly flow

.. _asm-f2: 
.. graphviz:: ../Fig/asm/asmDfdEx.gv
   :caption: Assembly flow, for instance: add $v1, $v0, $at
 
Given an example assembly instruction `add $v1, $v0, $at`, the LLVM AsmParser
core calls the backend `ParseInstruction()` function in `Cpu0AsmParser.cpp`
when it detects that the first token at the beginning of the line is an
identifier.

`ParseInstruction()` parses a single assembly instruction, creates the operand
objects, and returns them to the LLVM AsmParser. Then, the AsmParser calls the
backend `MatchAndEmitInstruction()` function to assign the opcode and operands
to an `MCInst`. The encoder then encodes the binary instruction from the `MCInst`
using information from `Cpu0InstrInfo.td`, which includes the binary values for
the opcode ID and operand IDs of the instruction.

Below is a list of the key functions and data structures in
`MatchAndEmitInstruction()` and `encodeInstruction()`, with explanations provided
in comments beginning with `///`.

.. rubric:: llvm/build/lib/Target/Cpu0/Cpu0GenAsmMatcher.inc
.. code-block:: c++
  
  enum InstructionConversionKind {
    Convert__Reg1_0__Reg1_1__Reg1_2,
    Convert__Reg1_0__Reg1_1__Imm1_2,
    ...
    CVT_NUM_SIGNATURES
  };
  
  } // end anonymous namespace
  
    struct MatchEntry {
      uint16_t Mnemonic;
      uint16_t Opcode;
      uint8_t ConvertFn;
      uint32_t RequiredFeatures;
      uint8_t Classes[3];
      StringRef getMnemonic() const {
        return StringRef(MnemonicTable + Mnemonic + 1,
                         MnemonicTable[Mnemonic]);
      }
    };
    
  static const MatchEntry MatchTable0[] = {
    { 0 /* add */, Cpu0::ADD, Convert__Reg1_0__Reg1_1__Reg1_2, 0, { MCK_CPURegs, MCK_CPURegs, MCK_CPURegs }, },
    { 4 /* addiu */, Cpu0::ADDiu, Convert__Reg1_0__Reg1_1__Imm1_2, 0, { MCK_CPURegs, MCK_CPURegs, MCK_Imm }, },
    ...
  };
  
  unsigned Cpu0AsmParser::
  MatchInstructionImpl(const OperandVector &Operands,
                       MCInst &Inst, uint64_t &ErrorInfo,
                       bool matchingInlineAsm, unsigned VariantID) {
    ...
    // Find the appropriate table for this asm variant.
    const MatchEntry *Start, *End;
    switch (VariantID) {
    default: llvm_unreachable("invalid variant!");
    case 0: Start = std::begin(MatchTable0); End = std::end(MatchTable0); break;
    }
    // Search the table.
    auto MnemonicRange = std::equal_range(Start, End, Mnemonic, LessOpcode());
    ...
    for (const MatchEntry *it = MnemonicRange.first, *ie = MnemonicRange.second;
         it != ie; ++it) {
      ...
      // We have selected a definite instruction, convert the parsed
      // operands into the appropriate MCInst.
      
      /// For instance ADD , V1, AT, V0
      /// MnemonicRange.first = &MatchTable0[0]
      /// MnemonicRange.second = &MatchTable0[1]
      /// it.ConvertFn = Convert__Reg1_0__Reg1_1__Reg1_2
  
      convertToMCInst(it->ConvertFn, Inst, it->Opcode, Operands);
      ...
    }
    ...
  }
  
  static const uint8_t ConversionTable[CVT_NUM_SIGNATURES][9] = {
    // Convert__Reg1_0__Reg1_1__Reg1_2
    { CVT_95_Reg, 1, CVT_95_Reg, 2, CVT_95_Reg, 3, CVT_Done },
    // Convert__Reg1_0__Reg1_1__Imm1_2
    { CVT_95_Reg, 1, CVT_95_Reg, 2, CVT_95_addImmOperands, 3, CVT_Done },
    ...
  };
  
  /// When kind = Convert__Reg1_0__Reg1_1__Reg1_2, ConversionTable[Kind] is equal to CVT_95_Reg
  /// For Operands[1], Operands[2], Operands[3] do the following:
  ///   static_cast<Cpu0Operand&>(*Operands[OpIdx]).addRegOperands(Inst, 1);
  /// Since p = 0, 2, 4, then OpIdx = 1, 2, 3 when OpIdx=*(p+1).
  /// Since, Operands[1] = V1, Operands[2] = AT, Operands[3] = V0, 
  ///   for "ADD , V1, AT, V0" which created by ParseInstruction().
  /// Inst.Opcode = ADD, Inst.Operand[0] = V1, Inst.Operand[1] = AT, 
  ///   Inst.Operand[2] = V0.
  void Cpu0AsmParser::
  convertToMCInst(unsigned Kind, MCInst &Inst, unsigned Opcode,
                  const OperandVector &Operands) {
    assert(Kind < CVT_NUM_SIGNATURES && "Invalid signature!");
    const uint8_t *Converter = ConversionTable[Kind];
    unsigned OpIdx;
    Inst.setOpcode(Opcode);
    for (const uint8_t *p = Converter; *p; p+= 2) {
      OpIdx = *(p + 1);
      switch (*p) {
      default: llvm_unreachable("invalid conversion entry!");
      case CVT_Reg:
        static_cast<Cpu0Operand&>(*Operands[OpIdx]).addRegOperands(Inst, 1);
        break;
      ...
      }
    }
  }

.. rubric:: lbdex/chapters/Chapter11_1/AsmParser/Cpu0AsmParser.cpp
.. code-block:: c++

  /// For "ADD , V1, AT, V0", ParseInstruction() set Operands[1].Reg.RegNum = V1, 
  ///   Operands[2].Reg.RegNum = AT, ..., by Cpu0Operand::CreateReg(RegNo, S,
  ///   Parser.getTok().getLoc()) in calling ParseOperand().
  /// So, after (*Operands[1..3]).addRegOperands(Inst, 1), 
  ///   Inst.Opcode = ADD, Inst.Operand[0] = V1, Inst.Operand[1] = AT, 
  ///   Inst.Operand[2] = V0.
  class Cpu0Operand : public MCParsedAsmOperand {
    ...
    void addRegOperands(MCInst &Inst, unsigned N) const {
      assert(N == 1 && "Invalid number of operands!");
      Inst.addOperand(MCOperand::createReg(getReg()));
    }
    ...    
    unsigned getReg() const override {
      assert((Kind == k_Register) && "Invalid access!");
      return Reg.RegNum;
    }
    ...
  }

.. rubric:: lbdex/chapters/Chapter11_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp
.. code-block:: c++
  
  void Cpu0MCCodeEmitter::
  encodeInstruction(const MCInst &MI, raw_ostream &OS,
                    SmallVectorImpl<MCFixup> &Fixups,
                    const MCSubtargetInfo &STI) const
  {
    uint32_t Binary = getBinaryCodeForInstr(MI, Fixups, STI);
    ...
    EmitInstruction(Binary, Size, OS);
  }

.. rubric:: llvm/build/lib/Target/Cpu0/Cpu0GenMCCodeEmitter.inc
.. code-block:: c++
  
  uint64_t Cpu0MCCodeEmitter::getBinaryCodeForInstr(const MCInst &MI,
      SmallVectorImpl<MCFixup> &Fixups,
      const MCSubtargetInfo &STI) const {
    static const uint64_t InstBits[] = {
      ...
      UINT64_C(318767104),	// ADD  /// 318767104=0x13000000
      ...
    };
    ...
    
    const unsigned opcode = MI.getOpcode();
    ...
    switch (opcode) {
      case Cpu0::ADD:
      ...
        // op: ra
        op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
        Value |= (op & UINT64_C(15)) << 20;
        // op: rb
        op = getMachineOpValue(MI, MI.getOperand(1), Fixups, STI);
        Value |= (op & UINT64_C(15)) << 16;
        // op: rc
        op = getMachineOpValue(MI, MI.getOperand(2), Fixups, STI);
        Value |= (op & UINT64_C(15)) << 12;
        break;
      }
      ...
    }
    return Value;
  }


.. _asm-f3: 
.. graphviz:: ../Fig/asm/asmDfdEx2.gv
   :caption: Data flow in MatchAndEmitInstruction(), for instance: add $v1, 
             $v0, $at"

.. _asm-f4: 
.. graphviz:: ../Fig/asm/asmDfdEx3.gv
   :caption: Data flow between MatchAndEmitInstruction() and encodeInstruction(), 
             for instance: add $v1, $v0, $at

`MatchTable0` includes all possible combinations of opcodes and operand types.

Even if a user’s assembly instruction passes the syntax check in
`Cpu0AsmParser`, `MatchAndEmitInstruction()` can still fail. For example, the
instruction `asm("move $3, $2");` may succeed, but `asm("move $3, $2, $1");`
will fail.

The flow of function calls for `Cpu0AsmParser` is shown in :numref:`asm-flow`.
  
.. _asm-flow: 
.. graphviz:: ../Fig/asm/asmFlow.gv
   :caption: Flow of calling functions for Cpu0AsmParser. 

- After `ParseInstruction()` and `MatchAndEmitInstruction()` are called, an
  `MCInst` object is produced.

  - In `MatchAndEmitInstruction()`, the assembler calls
    `MCObjectStreamer::emitInstruction()` to encode the instruction into binary.
    See :numref:`genobj-f11` for reference.

- Run `llc` with the option `-debug` or
  `-debug-only=asm-matcher,cpu0-asm-parser` to display debug messages and trace
  the assembler flow, as shown below.

- For Cpu0, only memory operands (used in L-type or J-type instructions) will
  trigger a call to `tryCustomParseOperand()`.

.. code-block:: console

  input % ~/llvm/test/build/bin/clang -target mips-unknown-linux-gnu -c 
  ch11_1.cpp -emit-llvm -o ch11_1.bc
  input % ~/llvm/test/build/bin/llc -march=cpu0 -relocation-model=pic 
  -filetype=obj -debug-only=asm-matcher,cpu0-asm-parser ch11_1.bc -o 
  ch11_1.cpu0.o

  ParseOperand
  .. Generic Parser
  ParseOperand
  <ld><Register<19>><Mem<9, 8>>
  AsmMatcher: found 1 encodings with mnemonic 'ld'
  Trying to match opcode LD
    Matching formal operand class MCK_CPURegs against actual operand at index 1 
    (Register<19>): match success using generic matcher
    Matching formal operand class MCK_Mem against actual operand at index 2 
    (Mem<9, 8>): match success using generic matcher
    Matching formal operand class InvalidMatchClass against actual operand at 
    index 3: actual operand index out of range Opcode result: complete match, 
    selecting this opcode

The other functions in `Cpu0AsmParser` are called in the following flow:

- `ParseDirective()`  
  → `parseDirectiveSet()`  
  → `parseSetReorderDirective()`,  
  `parseSetNoReorderDirective()`,  
  `parseSetMacroDirective()`,  
  `parseSetNoMacroDirective()`  
  → `reportParseError()`

- `ParseInstruction()`  
  → `ParseOperand()`  
  → `MatchOperandParserImpl()` (in `Cpu0GenAsmMatcher.inc`)  
  → `tryCustomParseOperand()` (in `Cpu0GenAsmMatcher.inc`)  
  → `parseMemOperand()`  
  → `parseMemOffset()`, `tryParseRegisterOperand()`

- `MatchAndEmitInstruction()`  
  → `MatchInstructionImpl()` (in `Cpu0GenAsmMatcher.inc`)  
  → `needsExpansion()`  
  → `expandInstruction()`

- `parseMemOffset()`  
  → `parseRelocOperand()`  
  → `getVariantKind()`

- `tryParseRegisterOperand()`  
  → `tryParseRegister()`  
  → `matchRegisterName()`  
  → `getReg()`, `matchRegisterByNumber()`

- `expandInstruction()`  
  → `expandLoadImm()`, `expandLoadAddressImm()`, `expandLoadAddressReg()`  
  → `EmitInstruction()` (in `Cpu0AsmPrint.cpp`)

Inline assembly
----------------

Run Chapter11_1 with ch11_2 will get the following error.

.. rubric:: lbdex/input/ch11_2.cpp
.. literalinclude:: ../lbdex/input/ch11_2.cpp
    :start-after: /// start

.. code-block:: console
  
  1-160-129-73:input Jonathan$ ~/llvm/test/build/bin/llc 
  -march=cpu0 -relocation-model=static -filetype=asm ch11_2.bc -o -
    .section .mdebug.abi32
    .previous
    .file "ch11_2.bc"
  error: couldn't allocate output register for constraint 'r'

The `ch11_2.cpp` file is an inline assembly example. Clang supports inline
assembly similarly to GCC.

Inline assembly is used in C/C++ when a program needs to access a specific
allocated register or memory location for a C/C++ variable. For example, the
variable `foo` in `ch11_2.cpp` may be allocated by the compiler to register `$2`,
`$3`, or another register.

Inline assembly helps bridge the gap between high-level languages and assembly
language. See reference [#inlineas]_.

`Chapter11_2` adds support for inline assembly as follows:

.. rubric:: lbdex/chapters/Chapter11_2/Cpu0AsmPrinter.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.h
    :start-after: #if CH >= CH11_2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter11_2/Cpu0AsmPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH11_2
    :end-before: #endif // #if CH >= CH11_2

.. rubric:: lbdex/chapters/Chapter11_2/Cpu0InstrInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.cpp
    :start-after: //@GetInstSizeInBytes {
    :end-before: //@GetInstSizeInBytes - body
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.cpp
    :start-after: #if CH >= CH11_2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter11_2/Cpu0ISelDAGToDAG.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.h
    :start-after: #if CH >= CH11_2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter11_2/Cpu0ISelDAGToDAG.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: #if CH >= CH11_2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter11_2/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH11_2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter11_2/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH11_2
    :end-before: #endif // #if CH >= CH11_2

Similar to the backend structure, the structure of inline assembly can also be
organized by file name, as shown in the table titled *"The structure of inline
assembly"*.

.. table:: inline assembly functions

  =============================  ================================== 
  File                           Function 
  =============================  ================================== 
  Cpu0ISelLowering.cpp           inline asm DAG node create
  Cpu0ISelDAGToDAG.cpp           save OP code 
  Cpu0AsmPrinter.cpp,            inline asm instructions printing    
  Cpu0InstrInfo.cpp              -                              
  =============================  ================================== 

Except for `Cpu0ISelDAGToDAG.cpp`, the other functions are the same as those
used in the backend's normal code compilation.

The inline assembly handling in `Cpu0ISelLowering.cpp` is explained after showing
the result of running with `ch11_2.cpp`.

`Cpu0ISelDAGToDAG.cpp` simply saves the opcode in
`SelectInlineAsmMemoryOperand()`. Since the opcode represents a Cpu0 inline
assembly instruction, no further LLVM IR DAG translation is needed. The function
just saves the opcode directly and returns `false` to notify the LLVM system that
the Cpu0 backend has finished processing this inline assembly instruction.

Run `Chapter11_2` with `ch11_2.cpp` to get the following result:

.. code-block:: console
  
  1-160-129-73:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch11_2.cpp -emit-llvm -o ch11_2.bc

  1-160-129-73:input Jonathan$ ~/llvm/test/build/bin/
  llvm-dis ch11_2.bc -o -
  ...
  target triple = "mips-unknown-linux-gnu"

  @g = global [3 x i32] [i32 1, i32 2, i32 3], align 4

  ; Function Attrs: nounwind
  define i32 @_Z14inlineasm_adduv() #0 {
    %foo = alloca i32, align 4
    %bar = alloca i32, align 4
    store i32 10, i32* %foo, align 4
    store i32 15, i32* %bar, align 4
    %1 = load i32* %foo, align 4
    %2 = call i32 asm sideeffect "addu $0,$1,$2", "=r,r,r"(i32 %1, i32 15) #1, 
    !srcloc !1
    store i32 %2, i32* %foo, align 4
    %3 = load i32* %foo, align 4
    ret i32 %3
  }

  ; Function Attrs: nounwind
  define i32 @_Z18inlineasm_longlongv() #0 {
    %a = alloca i32, align 4
    %b = alloca i32, align 4
    %bar = alloca i64, align 8
    %p = alloca i32*, align 4
    %q = alloca i32*, align 4
    store i64 21474836486, i64* %bar, align 8
    %1 = bitcast i64* %bar to i32*
    store i32* %1, i32** %p, align 4
    %2 = load i32** %p, align 4
    %3 = call i32 asm sideeffect "ld $0,$1", "=r,*m"(i32* %2) #1, !srcloc !2
    store i32 %3, i32* %a, align 4
    %4 = load i32** %p, align 4
    %5 = getelementptr inbounds i32* %4, i32 1
    store i32* %5, i32** %q, align 4
    %6 = load i32** %q, align 4
    %7 = call i32 asm sideeffect "ld $0,$1", "=r,*m"(i32* %6) #1, !srcloc !3
    store i32 %7, i32* %b, align 4
    %8 = load i32* %a, align 4
    %9 = load i32* %b, align 4
    %10 = add nsw i32 %8, %9
    ret i32 %10
  }

  ; Function Attrs: nounwind
  define i32 @_Z20inlineasm_constraintv() #0 {
    %foo = alloca i32, align 4
    %n_5 = alloca i32, align 4
    %n5 = alloca i32, align 4
    %n0 = alloca i32, align 4
    %un5 = alloca i32, align 4
    %n65536 = alloca i32, align 4
    %n_65531 = alloca i32, align 4
    store i32 10, i32* %foo, align 4
    store i32 -5, i32* %n_5, align 4
    store i32 5, i32* %n5, align 4
    store i32 0, i32* %n0, align 4
    store i32 5, i32* %un5, align 4
    store i32 65536, i32* %n65536, align 4
    store i32 -65531, i32* %n_65531, align 4
    %1 = load i32* %foo, align 4
    %2 = call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,I"(i32 %1, i32 -5) #1, 
    !srcloc !4
    store i32 %2, i32* %foo, align 4
    %3 = load i32* %foo, align 4
    %4 = call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,J"(i32 %3, i32 0) #1, 
    !srcloc !5
    store i32 %4, i32* %foo, align 4
    %5 = load i32* %foo, align 4
    %6 = call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,K"(i32 %5, i32 5) #1, 
    !srcloc !6
    store i32 %6, i32* %foo, align 4
    %7 = load i32* %foo, align 4
    %8 = call i32 asm sideeffect "ori $0,$1,$2", "=r,r,L"(i32 %7, i32 65536) #1, 
    !srcloc !7
    store i32 %8, i32* %foo, align 4
    %9 = load i32* %foo, align 4
    %10 = call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,N"(i32 %9, i32 -65531) 
    #1, !srcloc !8
    store i32 %10, i32* %foo, align 4
    %11 = load i32* %foo, align 4
    %12 = call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,O"(i32 %11, i32 -5) #1, 
    !srcloc !9
    store i32 %12, i32* %foo, align 4
    %13 = load i32* %foo, align 4
    %14 = call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,P"(i32 %13, i32 5) #1, 
    !srcloc !10
    store i32 %14, i32* %foo, align 4
    %15 = load i32* %foo, align 4
    ret i32 %15
  }

  ; Function Attrs: nounwind
  define i32 @_Z13inlineasm_argii(i32 %u, i32 %v) #0 {
    %1 = alloca i32, align 4
    %2 = alloca i32, align 4
    %w = alloca i32, align 4
    store i32 %u, i32* %1, align 4
    store i32 %v, i32* %2, align 4
    %3 = load i32* %1, align 4
    %4 = load i32* %2, align 4
    %5 = call i32 asm sideeffect "subu $0,$1,$2", "=r,r,r"(i32 %3, i32 %4) #1, 
    !srcloc !11
    store i32 %5, i32* %w, align 4
    %6 = load i32* %w, align 4
    ret i32 %6
  }

  ; Function Attrs: nounwind
  define i32 @_Z16inlineasm_globalv() #0 {
    %c = alloca i32, align 4
    %d = alloca i32, align 4
    %1 = call i32 asm sideeffect "ld $0,$1", "=r,*m"(i32* getelementptr inbounds 
    ([3 x i32]* @g, i32 0, i32 2)) #1, !srcloc !12
    store i32 %1, i32* %c, align 4
    %2 = load i32* %c, align 4
    %3 = call i32 asm sideeffect "addiu $0,$1,1", "=r,r"(i32 %2) #1, !srcloc !13
    store i32 %3, i32* %d, align 4
    %4 = load i32* %d, align 4
    ret i32 %4
  }

  ; Function Attrs: nounwind
  define i32 @_Z14test_inlineasmv() #0 {
    %a = alloca i32, align 4
    %b = alloca i32, align 4
    %c = alloca i32, align 4
    %d = alloca i32, align 4
    %e = alloca i32, align 4
    %f = alloca i32, align 4
    %g = alloca i32, align 4
    %1 = call i32 @_Z14inlineasm_adduv()
    store i32 %1, i32* %a, align 4
    %2 = call i32 @_Z18inlineasm_longlongv()
    store i32 %2, i32* %b, align 4
    %3 = call i32 @_Z20inlineasm_constraintv()
    store i32 %3, i32* %c, align 4
    %4 = call i32 @_Z13inlineasm_argii(i32 1, i32 10)
    store i32 %4, i32* %d, align 4
    %5 = call i32 @_Z13inlineasm_argii(i32 6, i32 3)
    store i32 %5, i32* %e, align 4
    %6 = load i32* %e, align 4
    %7 = call i32 asm sideeffect "addiu $0,$1,1", "=r,r"(i32 %6) #1, !srcloc !14
    store i32 %7, i32* %f, align 4
    %8 = call i32 @_Z16inlineasm_globalv()
    store i32 %8, i32* %g, align 4
    %9 = load i32* %a, align 4
    %10 = load i32* %b, align 4
    %11 = add nsw i32 %9, %10
    %12 = load i32* %c, align 4
    %13 = add nsw i32 %11, %12
    %14 = load i32* %d, align 4
    %15 = add nsw i32 %13, %14
    %16 = load i32* %e, align 4
    %17 = add nsw i32 %15, %16
    %18 = load i32* %f, align 4
    %19 = add nsw i32 %17, %18
    %20 = load i32* %g, align 4
    %21 = add nsw i32 %19, %20
    ret i32 %21
  }
  ...
  1-160-129-73:input Jonathan$ ~/llvm/test/build/bin/llc 
    -march=cpu0 -relocation-model=static -filetype=asm ch11_2.bc -o -
    .section .mdebug.abi32
    .previous
    .file "ch11_2.bc"
    .text
    .globl  _Z14inlineasm_adduv
    .align  2
    .type _Z14inlineasm_adduv,@function
    .ent  _Z14inlineasm_adduv     # @_Z14inlineasm_adduv
  _Z14inlineasm_adduv:
    .frame  $fp,16,$lr
    .mask   0x00001000,-4
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -16
    st  $fp, 12($sp)            # 4-byte Folded Spill
    addu  $fp, $sp, $zero
    addiu $2, $zero, 10
    st  $2, 8($fp)
    addiu $2, $zero, 15
    st  $2, 4($fp)
    ld  $3, 8($fp)
    #APP
    addu $2,$3,$2
    #NO_APP
    st  $2, 8($fp)
    addu  $sp, $fp, $zero
    ld  $fp, 12($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 16
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z14inlineasm_adduv
  $tmp3:
    .size _Z14inlineasm_adduv, ($tmp3)-_Z14inlineasm_adduv

    .globl  _Z18inlineasm_longlongv
    .align  2
    .type _Z18inlineasm_longlongv,@function
    .ent  _Z18inlineasm_longlongv # @_Z18inlineasm_longlongv
  _Z18inlineasm_longlongv:
    .frame  $fp,32,$lr
    .mask   0x00001000,-4
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -32
    st  $fp, 28($sp)            # 4-byte Folded Spill
    addu  $fp, $sp, $zero
    addiu $2, $zero, 6
    st  $2, 12($fp)
    addiu $2, $zero, 5
    st  $2, 8($fp)
    addiu $2, $fp, 8
    st  $2, 4($fp)
    #APP
    ld $2,0($2)
    #NO_APP
    st  $2, 24($fp)
    ld  $2, 4($fp)
    addiu $2, $2, 4
    st  $2, 0($fp)
    #APP
    ld $2,0($2)
    #NO_APP
    st  $2, 20($fp)
    ld  $3, 24($fp)
    addu  $2, $3, $2
    addu  $sp, $fp, $zero
    ld  $fp, 28($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 32
    ret $lr
    .set  macro
    .set  reorder
    .end  _Z18inlineasm_longlongv
  $tmp7:
    .size _Z18inlineasm_longlongv, ($tmp7)-_Z18inlineasm_longlongv

    .globl  _Z20inlineasm_constraintv
    .align  2
    .type _Z20inlineasm_constraintv,@function
    .ent  _Z20inlineasm_constraintv # @_Z20inlineasm_constraintv
  _Z20inlineasm_constraintv:
    .frame  $fp,32,$lr
    .mask   0x00001000,-4
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -32
    st  $fp, 28($sp)            # 4-byte Folded Spill
    addu  $fp, $sp, $zero
    addiu $2, $zero, 10
    st  $2, 24($fp)
    addiu $2, $zero, -5
    st  $2, 20($fp)
    addiu $2, $zero, 5
    st  $2, 16($fp)
    addiu $3, $zero, 0
    st  $3, 12($fp)
    st  $2, 8($fp)
    lui $2, 1
    st  $2, 4($fp)
    lui $2, 65535
    ori $2, $2, 5
    st  $2, 0($fp)
    ld  $2, 24($fp)
    #APP
    addiu $2,$2,-5
    #NO_APP
    st  $2, 24($fp)
    #APP
    addiu $2,$2,0
    #NO_APP
    st  $2, 24($fp)
    #APP
    addiu $2,$2,5
    #NO_APP
    st  $2, 24($fp)
    #APP
    ori $2,$2,65536
    #NO_APP
    st  $2, 24($fp)
    #APP
    addiu $2,$2,-65531
    #NO_APP
    st  $2, 24($fp)
    #APP
    addiu $2,$2,-5
    #NO_APP
    st  $2, 24($fp)
    #APP
    addiu $2,$2,5
    #NO_APP
    st  $2, 24($fp)
    addu  $sp, $fp, $zero
    ld  $fp, 28($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 32
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z20inlineasm_constraintv
  $tmp11:
    .size _Z20inlineasm_constraintv, ($tmp11)-_Z20inlineasm_constraintv

    .globl  _Z13inlineasm_argii
    .align  2
    .type _Z13inlineasm_argii,@function
    .ent  _Z13inlineasm_argii     # @_Z13inlineasm_argii
  _Z13inlineasm_argii:
    .frame  $fp,16,$lr
    .mask   0x00001000,-4
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -16
    st  $fp, 12($sp)            # 4-byte Folded Spill
    addu  $fp, $sp, $zero
    ld  $2, 16($fp)
    st  $2, 8($fp)
    ld  $2, 20($fp)
    st  $2, 4($fp)
    ld  $3, 8($fp)
    #APP
    subu $2,$3,$2
    #NO_APP
    st  $2, 0($fp)
    addu  $sp, $fp, $zero
    ld  $fp, 12($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 16
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z13inlineasm_argii
  $tmp15:
    .size _Z13inlineasm_argii, ($tmp15)-_Z13inlineasm_argii

    .globl  _Z16inlineasm_globalv
    .align  2
    .type _Z16inlineasm_globalv,@function
    .ent  _Z16inlineasm_globalv   # @_Z16inlineasm_globalv
  _Z16inlineasm_globalv:
    .frame  $fp,16,$lr
    .mask   0x00001000,-4
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -16
    st  $fp, 12($sp)            # 4-byte Folded Spill
    addu  $fp, $sp, $zero
    lui $2, %hi(g)
    ori $2, $2, %lo(g)
    addiu $2, $2, 8
    #APP
    ld $2,0($2)
    #NO_APP
    st  $2, 8($fp)
    #APP
    addiu $2,$2,1
    #NO_APP
    st  $2, 4($fp)
    addu  $sp, $fp, $zero
    ld  $fp, 12($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 16
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z16inlineasm_globalv
  $tmp19:
    .size _Z16inlineasm_globalv, ($tmp19)-_Z16inlineasm_globalv

    .globl  _Z14test_inlineasmv
    .align  2
    .type _Z14test_inlineasmv,@function
    .ent  _Z14test_inlineasmv     # @_Z14test_inlineasmv
  _Z14test_inlineasmv:
    .frame  $fp,48,$lr
    .mask   0x00005000,-4
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -48
    st  $lr, 44($sp)            # 4-byte Folded Spill
    st  $fp, 40($sp)            # 4-byte Folded Spill
    addu  $fp, $sp, $zero
    jsub  _Z14inlineasm_adduv
    nop
    st  $2, 36($fp)
    jsub  _Z18inlineasm_longlongv
    nop
    st  $2, 32($fp)
    jsub  _Z20inlineasm_constraintv
    nop
    st  $2, 28($fp)
    addiu $2, $zero, 10
    st  $2, 4($sp)
    addiu $2, $zero, 1
    st  $2, 0($sp)
    jsub  _Z13inlineasm_argii
    nop
    st  $2, 24($fp)
    addiu $2, $zero, 3
    st  $2, 4($sp)
    addiu $2, $zero, 6
    st  $2, 0($sp)
    jsub  _Z13inlineasm_argii
    nop
    st  $2, 20($fp)
    #APP
    addiu $2,$2,1
    #NO_APP
    st  $2, 16($fp)
    jsub  _Z16inlineasm_globalv
    nop
    st  $2, 12($fp)
    ld  $3, 32($fp)
    ld  $4, 36($fp)
    addu  $3, $4, $3
    ld  $4, 28($fp)
    addu  $3, $3, $4
    ld  $4, 24($fp)
    addu  $3, $3, $4
    ld  $4, 20($fp)
    addu  $3, $3, $4
    ld  $4, 16($fp)
    addu  $3, $3, $4
    addu  $2, $3, $2
    addu  $sp, $fp, $zero
    ld  $fp, 40($sp)            # 4-byte Folded Reload
    ld  $lr, 44($sp)            # 4-byte Folded Reload
    addiu $sp, $sp, 48
    ret $lr
    nop
    .set  macro
    .set  reorder
    .end  _Z14test_inlineasmv
  $tmp23:
    .size _Z14test_inlineasmv, ($tmp23)-_Z14test_inlineasmv

    .type g,@object               # @g
    .data
    .globl  g
    .align  2
  g:
    .4byte  1                       # 0x1
    .4byte  2                       # 0x2
    .4byte  3                       # 0x3
    .size g, 12


Clang first translates GCC-style inline assembly (`__asm__`) into LLVM IR
Inline Assembler Expressions [#cia]_. Then, during the `llc` register allocation
stage, it replaces the SSA-form variable registers with physical registers.

In the above example, the functions `LowerAsmOperandForConstraint()` and
`getSingleConstraintMatchWeight()` in `Cpu0ISelLowering.cpp` generate different
ranges of constant operands based on constraint codes `I`, `J`, `K`, `L`, `N`,
`O`, or `P`, and register operands based on `r`.

For instance, the following `__asm__` generates the corresponding LLVM inline
assembly immediately after it:

.. code-block:: cpp-objdump

  __asm__ __volatile__("addiu %0,%1,%2"
                       :"=r"(foo) // 15
                       :"r"(foo), "I"(n_5)
                       );

.. code-block:: llvm

  %2 = call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,I"(i32 %1, i32 -5) #0, !srcloc !1

.. code-block:: cpp-objdump

  __asm__ __volatile__("addiu %0,%1,%2"
                       :"=r"(foo) // 15
                       :"r"(foo), "N"(n_65531)
                       );

.. code-block:: llvm

  %10 = call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,N"(i32 %9, i32 -65531) #0, !srcloc !5
  
.. code-block:: cpp-objdump

  __asm__ __volatile__("addiu %0,%1,%2"
                       :"=r"(foo) // 15
                       :"r"(foo), "P"(un5)
                       );

.. code-block:: llvm

  %14 = call i32 asm sideeffect "addiu $0,$1,$2", "=r,r,P"(i32 %13, i32 5) #0, !srcloc !7

The `r` constraint in `__asm__` will generate a register operand (e.g., `%1`) in
LLVM IR inline assembly. The `I` constraint will generate a constant operand
(e.g., `-5`) in LLVM IR inline assembly.

Note that `LowerAsmOperandForConstraint()` limits the range of positive and
negative constant operand values to 16 bits, since the FL-type immediate operand
in Cpu0 instructions is 16 bits.

As a result:

- The range of `N` is from `-65535` to `-1`.
- The range of `P` is from `1` to `65535`.

Any value outside of these ranges is treated as an error in
`LowerAsmOperandForConstraint()`, due to the 16-bit limitation of the FL
instruction format.

.. [#llvmas] http://www.embecosm.com/appnotes/ean10/ean10-howto-llvmas-1.0.html

.. [#inlineas] http://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html

.. [#cia] http://llvm.org/docs/LangRef.html#inline-assembler-expressions

