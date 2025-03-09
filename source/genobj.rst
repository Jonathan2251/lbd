.. _sec-genobjfiles:

Generating object files
=======================

.. contents::
   :local:
   :depth: 4

The previous chapters focused solely on **assembly code generation**.  
This chapter extends that by **adding ELF object file support**  
and verifying the generated object files using the `objdump` utility.

With LLVM support, the **Cpu0 backend** can generate both **big-endian**  
and **little-endian** object files with minimal additional code.

Additionally, this chapter introduces the **Target Registration mechanism**  
and its structure.

Similar to :numref:`asm-emit` from the previous chapter :ref:`sec-backendstructure`, but 
this chapter focuses on emitting **binary object instructions** as shown in :numref:`obj-emit`.

.. _obj-emit:
.. graphviz:: ../Fig/genobj/obj-emit.gv
   :caption: When "llc -filetype=obj", Cpu0AsmPrinter extract MCInst from MachineInstr for obj encoding

Translate into obj file
------------------------

Currently, the backend **only supports translating LLVM IR code into assembly code**.  
If you attempt to compile an LLVM IR input file into an **object file** using the  
`Chapter4_2/` compiler code, you will encounter the following error message:

.. code-block:: console

  bin$ pwd
  $HOME/llvm/test/build/bin/
  bin$ llc -march=cpu0 -relocation-model=pic -filetype=obj ch4_1_math_math.bc -o 
  ch4_1_math.cpu0.o
  ~/llvm/test/build/bin/llc: target does not 
  support generation of this file type! 
	
The `Chapter5_1/` implementation **supports object file generation**.  
It can produce object files for both **big-endian** and **little-endian** architectures  
using the following commands:  

- **Big-endian:** `llc -march=cpu0`  
- **Little-endian:** `llc -march=cpu0el`  

Running these commands will generate the corresponding object files as shown below:

.. code-block:: console

  input$ cat ch4_1_math.cpu0.s 
  ...
    .set  nomacro
  # BB#0:                                 # %entry
    addiu $sp, $sp, -40
  $tmp1:
    .cfi_def_cfa_offset 40
    addiu $2, $zero, 5
    st  $2, 36($fp)
    addiu $2, $zero, 2
    st  $2, 32($fp)
    addiu $2, $zero, 0
    st  $2, 28($fp)
  ...
  
  bin$ pwd
  $HOME/llvm/test/build/bin/
  bin$ llc -march=cpu0 -relocation-model=pic -filetype=obj ch4_1_math.bc -o 
  ch4_1_math.cpu0.o
  input$ objdump -s ch4_1_math.cpu0.o 
  
  ch4_1_math.cpu0.o:     file format elf32-big 
  
  Contents of section .text: 
   0000 09ddffc8 09200005 022d0034 09200002  ..... ...-.4. ..
   0010 022d0030 0920fffb 022d002c 012d0030  .-.0. ...-.,.-.0
   0020 013d0034 11232000 022d0028 012d0030  .=.4.# ..-.(.-.0
   0030 013d0034 12232000 022d0024 012d0030  .=.4.# ..-.$.-.0
   0040 013d0034 17232000 022d0020 012d0034  .=.4.# ..-. .-.4
   0050 1e220002 022d001c 012d002c 1e220001  ."...-...-.,."..
   0060 022d000c 012d0034 1d220002 022d0018  .-...-.4."...-..
   0070 012d002c 1f22001e 022d0008 09200001  .-.,."...-... ..
   0080 013d0034 21323000 023d0014 013d0030  .=.4!20..=...=.0
   0090 21223000 022d0004 09200080 013d0034  !"0..-... ...=.4
   00a0 22223000 022d0010 012d0034 013d0030  ""0..-...-.4.=.0
   00b0 20232000 022d0000 09dd0038 3ce00000   # ..-.....8<...     
   
  input$ ~/llvm/test/
  build/bin/llc -march=cpu0el -relocation-model=pic -filetype=obj 
  ch4_1_math.bc -o ch4_1_math.cpu0el.o 
  input$ objdump -s ch4_1_math.cpu0el.o 
  
  ch4_1_math.cpu0el.o:     file format elf32-little 
  
  Contents of section .text: 
   0000 c8ffdd09 05002009 34002d02 02002009  ...... .4.-... .
   0010 30002d02 fbff2009 2c002d02 30002d01  0.-... .,.-.0.-.
   0020 34003d01 00202311 28002d02 30002d01  4.=.. #.(.-.0.-.
   0030 34003d01 00202312 24002d02 30002d01  4.=.. #.$.-.0.-.
   0040 34003d01 00202317 20002d02 34002d01  4.=.. #. .-.4.-.
   0050 0200221e 1c002d02 2c002d01 0100221e  .."...-.,.-...".
   0060 0c002d02 34002d01 0200221d 18002d02  ..-.4.-..."...-.
   0070 2c002d01 1e00221f 08002d02 01002009  ,.-..."...-... .
   0080 34003d01 00303221 14003d02 30003d01  4.=..02!..=.0.=.
   0090 00302221 04002d02 80002009 34003d01  .0"!..-... .4.=.
   00a0 00302222 10002d02 34002d01 30003d01  .0""..-.4.-.0.=.
   00b0 00202320 00002d02 3800dd09 0000e03c  . # ..-.8......<      
         
The first instruction is **"addiu $sp, -56"**, and its corresponding obj is  
0x09ddffc8. The opcode of addiu is 0x09 (8 bits); the $sp register number  
is 13 (0xd) (4 bits); and the immediate value is -56 (0xffc8) (16 bits),  
so it is correct.

The third instruction **"st $2, 52($fp)"**, and its corresponding obj is  
0x022b0034. The **st** opcode is 0x02, $2 is 0x2, $fp is 0xb, and the  
immediate value is 52 (0x0034).

Thanks to the Cpu0 instruction format, where the opcode, register operand,  
and offset (immediate value) sizes are multiples of 4 bits, the obj format  
is easy to verify manually.

For big-endian format: (B0, B1, B2, B3) = (09, dd, ff, c8), objdump from  
B0 to B3 is 0x09ddffc8.  

For little-endian format: (B3, B2, B1, B0) = (09, dd, ff, c8), objdump from  
B0 to B3 is 0xc8ffdd09.


ELF obj related code
----------------------

To support ELF object file generation, the following code was modified and  
added to Chapter5_1.

.. rubric:: lbdex/chapters/Chapter5_1/InstPrinter/Cpu0InstPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/InstPrinter/Cpu0InstPrinter.cpp
    :start-after: #if CH >= CH5_1 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/CMakeLists.txt
    :start-after: #if CH >= CH5_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0AsmBackend.h
.. literalinclude:: ../lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0AsmBackend.h

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0AsmBackend.cpp
.. literalinclude:: ../lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0AsmBackend.cpp

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0BaseInfo.h
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0BaseInfo.h
    :start-after: #if CH >= CH5_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0ELFObjectWriter.cpp
.. literalinclude:: ../lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0ELFObjectWriter.cpp

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0FixupKinds.h
.. literalinclude:: ../lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0FixupKinds.h

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCCodeEmitter.h
.. literalinclude:: ../lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCCodeEmitter.h

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp
.. literalinclude:: ../lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCExpr.h
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCExpr.h

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCExpr.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCExpr.cpp

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCTargetDesc.h
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCTargetDesc.h
    :start-after: #if CH >= CH5_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCTargetDesc.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCTargetDesc.cpp
    :start-after: #if CH >= CH5_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCTargetDesc.cpp
    :start-after: //@2 {
    :end-before: #if CH >= CH3_2 //3
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCTargetDesc.cpp
    :start-after: #if CH >= CH5_1 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCTargetDesc.cpp
    :start-after: #if CH >= CH5_1 //3
    :end-before: #endif

.. code-block:: c++

  }

.. rubric:: lbdex/chapters/Chapter5_1/Cpu0MCInstLower.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.h
    :start-after: #if CH >= CH5_1
    :end-before: #endif


Work flow
---------

In Chapter 3_2, `OutStreamer->emitInstruction` prints the assembly code.  
To support ELF object file generation, this chapter creates  
`MCELFObjectStreamer`, which inherits from `OutStreamer` by calling  
`createELFStreamer` in `Cpu0MCTargetDesc.cpp` above.  

Once `MCELFObjectStreamer` is created, `OutStreamer->emitInstruction` will  
work with other code added in the `MCTargetDesc` directory of this chapter.  
The details of the explanation are as follows:

.. rubric:: llvm/include/llvm/CodeGen/AsmPrinter.h
.. code-block:: c++

  class AsmPrinter : public MachineFunctionPass {
  public:
    ...
    std::unique_ptr<MCStreamer> OutStreamer;
    ...
  }

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0AsmPrinter.h
.. code-block:: c++

  class LLVM_LIBRARY_VISIBILITY Cpu0AsmPrinter : public AsmPrinter {

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0AsmPrinter.cpp
.. code-block:: c++

  void Cpu0AsmPrinter::emitInstruction(const MachineInstr *MI) {
    ...
    do {
      ...
      OutStreamer->emitInstruction(TmpInst0, getSubtargetInfo());
      ...
    } while ((++I != E) && I->isInsideBundle()); // Delay slot check
  }

.. _genobj-f11: 
.. graphviz:: ../Fig/genobj/callFunctions.gv
   :caption: Calling Functions of elf encoder

- AsmPrinter::OutStreamer is \nMCObjectStreamer if llc -filetype=obj;
  AsmPrinter::OutStreamer is \nMCAsmStreamer if llc -filetype=asm as 
  :numref:`print-asm`.

The ELF encoder calling functions are shown in :numref:`genobj-f11` above.  
`AsmPrinter::OutStreamer` is set to `MCObjectStreamer` by the `llc` driver  
when the user inputs ``llc -filetype=obj``.

.. _genobj-f12: 
.. graphviz:: ../Fig/genobj/instEncodeDfd.gv
   :caption: DFD flow for instruction encode

The instruction operand information for the encoder is obtained as shown in  
:numref:`genobj-f12` above. The steps are as follows:

  1. The function `encodeInstruction()` passes `MI.Opcode` to  
     `getBinaryCodeForInstr()`.

  2. `getBinaryCodeForInstr()` passes `MI.Operand[n]` to `getMachineOpValue()`,  
     and then,

  3. It retrieves the register number by calling `getMachineOpValue()`.

  4. `getBinaryCodeForInstr()` returns `MI` with all register numbers to  
     `encodeInstruction()`.

The `MI.Opcode` is set in the Instruction Selection Stage.  
The table generation function `getBinaryCodeForInstr()` gathers all operand  
information from the `.td` files defined by the programmer, as shown in  
:numref:`genobj-f13`.

.. _genobj-f13: 
.. graphviz:: ../Fig/genobj/getBinaryCodeForInstr.gv
   :caption: Instruction encode, for instance:  addu $v0, $at, $v1\n  v0:MI.getOperand(0), at:MI.getOperand(1), v1:MI.getOperand(2)

For instance, the Cpu0 backend will generate `"addu $v0, $at, $v1"` for the IR  
`"%0 = add %1, %2"` once LLVM allocates registers `$v0`, `$at`, and `$v1` for  
operands `%0`, `%1`, and `%2` individually. The `MCOperand` structure for  
`MI.Operands[]` includes the register number set in the pass where LLVM  
allocates registers, which can be retrieved in `getMachineOpValue()`.

The function `getEncodingValue(Reg)` in `getMachineOpValue()`, as shown below,  
retrieves the `RegNo` for encoding from register names such as `AT`, `V0`, or  
`V1`, using table generation information from `Cpu0RegisterInfo.td`.  
My comments are after `"///"`.

.. rubric:: include//llvm/MC/MCRegisterInfo.h
.. code-block:: c++

  void InitMCRegisterInfo(...,
                          const uint16_t *RET) {
    ...
    RegEncodingTable = RET;
  }
  
  unsigned Cpu0MCCodeEmitter::
  getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                    SmallVectorImpl<MCFixup> &Fixups,
                    const MCSubtargetInfo &STI) const {
    if (MO.isReg()) {
      unsigned Reg = MO.getReg();
      unsigned RegNo = Ctx.getRegisterInfo()->getEncodingValue(Reg);
      return RegNo;
    ...
  }

.. rubric:: include/llvm/MC/MCRegisterInfo.h
.. code-block:: c++

  void InitMCRegisterInfo(...,
                          const uint16_t *RET) {
    ...
    RegEncodingTable = RET;
  }

   /// \brief Returns the encoding for RegNo
  uint16_t getEncodingValue(unsigned RegNo) const {
    assert(RegNo < NumRegs &&
           "Attempting to get encoding for invalid register number!");
    return RegEncodingTable[RegNo];
  }

.. rubric:: lbdex/chapters/Chapter5_1/Cpu0RegisterInfo.td
.. code-block:: c++
  
  let Namespace = "Cpu0" in {
    ...
    def AT   : Cpu0GPRReg<1,  "1">,    DwarfRegNum<[1]>;
    def V0   : Cpu0GPRReg<2,  "2">,    DwarfRegNum<[2]>;
    def V1   : Cpu0GPRReg<3,  "3">,    DwarfRegNum<[3]>;
    ...
  }

.. rubric:: build/lib/Target/Cpu0/Cpu0GenRegisterInfo.inc
.. code-block:: c++
  
  namespace Cpu0 {
  enum {
    NoRegister,
    AT = 1,
    ...
    V0 = 19,
    V1 = 20,
    NUM_TARGET_REGS       // 21
  };
  } // end namespace Cpu0
  
  extern const uint16_t Cpu0RegEncodingTable[] = {
    0,
    1,     /// 1, AT
    1,
    12,
    11,
    0,
    0,
    14,
    0,
    13,
    15,
    0,
    4,
    5,
    9,
    10,
    7,
    8,
    6,
    2,    /// 19, V0
    3,    /// 20, V1
  };
  
  static inline void InitCpu0MCRegisterInfo(MCRegisterInfo *RI, ...) {
    RI->InitMCRegisterInfo(..., Cpu0RegEncodingTable);
  
The `applyFixup()` function in `Cpu0AsmBackend.cpp` will fix up the **jeq**,  
**jub**, and other instructions related to "address control flow statements" or  
"function call statements" used in later chapters.  

The setting of `true` or `false` for each relocation record in  
`needsRelocateWithSymbol()` of `Cpu0ELFObjectWriter.cpp` depends on whether  
this relocation record needs to adjust the address value during linking.  

- If set to `true`, the linker has the opportunity to adjust this address value  
  with the correct information.  
- If set to `false`, the linker lacks the correct information to adjust this  
  relocation record.  

The concept of relocation records will be introduced in a later chapter on  
**ELF Support**.  

When emitting ELF object format instructions, the `EncodeInstruction()`  
function in `Cpu0MCCodeEmitter.cpp` will be called, as it overrides the function  
of the same name in the parent class `MCCodeEmitter`.  

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 5
    :end-before: //#if CH >= CH11_1
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#endif //#if CH >= CH11_1 2
    :end-before: //#endif //#if CH >= CH2 5
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@ 32-bit load.
    :end-before: //#endif

As shown in :numref:`llvmstructure-f2`, **ld** and **st** follow the L-Type  
format, whereas **ADD** and similar instructions follow the R-Type format.  

The statement `"let EncoderMethod = "getMemEncoding";"` in `Cpu0InstrInfo.td`,  
as mentioned above, ensures that LLVM calls the function `getMemEncoding()`  
whenever an **ld** or **st** instruction is issued in the ELF object file.  
This happens because both of these instructions use the **mem** operand.  

Below is the implementation and the corresponding TableGen code for them.  

.. rubric:: lbdex/chapters/Chapter5_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/chapters/Chapter5_1/Cpu0InstrInfo.td
    :start-after: // Address operand
    :end-before: // Transformation Function - get the lower 16 bits.

.. rubric:: lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp
.. literalinclude:: ../lbdex/chapters/Chapter5_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: /// getMemEncoding

.. rubric:: build/lib/Target/Cpu0/Cpu0GenMCCodeEmitter.inc
.. code-block:: c++

    case Cpu0::LD
    case Cpu0::ST: {
      // op: ra
      op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
      op &= UINT64_C(15);
      op <<= 20;
      Value |= op;
      // op: addr
      op = getMemEncoding(MI, 1, Fixups, STI);
      op &= UINT64_C(1048575);
      Value |= op;
      break;
    } 


The other functions in `Cpu0MCCodeEmitter.cpp` are called by these two functions.  

After encoding, the following code will write the encoded instructions to the buffer.  

.. rubric:: src/lib/MC/MCELFStreamer.cpp
.. code-block:: c++
  
  void MCELFStreamer::EmitInstToData(const MCInst &Inst,
                                     const MCSubtargetInfo &STI) {
    ...
    DF->setHasInstructions(true);
    DF->getContents().append(Code.begin(), Code.end());
    ...
  }

Then, `ELFObjectWriter::writeObject()` will write the buffer to the ELF file.  

Backend Target Registration Structure
--------------------------------------

Now, let's examine `Cpu0MCTargetDesc.cpp`.  
`Cpu0MCTargetDesc.cpp` performs target registration as mentioned in  
the previous chapter here [#target-registration]_. The assembly output  
has been explained here [#add-asmprinter]_.  

The register functions for ELF object output are listed as follows:

.. rubric:: Register function of elf streamer
.. code-block:: c++

  // Register the elf streamer.
  TargetRegistry::RegisterELFStreamer(*T, createMCStreamer);

    static MCStreamer *createMCStreamer(const Triple &TT, MCContext &Context, 
                                        MCAsmBackend &MAB, raw_pwrite_stream &OS, 
                                        MCCodeEmitter *Emitter, bool RelaxAll) {
      return createELFStreamer(Context, MAB, OS, Emitter, RelaxAll);
    }

    // MCELFStreamer.cpp
    MCStreamer *llvm::createELFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                        raw_pwrite_stream &OS, MCCodeEmitter *CE,
                                        bool RelaxAll) {
      MCELFStreamer *S = new MCELFStreamer(Context, MAB, OS, CE);
      if (RelaxAll)
        S->getAssembler().setRelaxAll(true);
      return S;
    }

Above, `createELFStreamer` handles the ELF object streamer.  
:numref:`genobj-f10` below shows the `MCELFStreamer` inheritance tree.  
You can find many operations within that inheritance tree.

.. _genobj-f10:
.. figure:: ../Fig/genobj/10.png
	:height: 596 px
	:width: 783 px
	:scale: 100 %
	:align: center

	MCELFStreamer inherit tree

.. rubric:: Register function of asm target streamer
.. code-block:: c++

  // Register the asm target streamer.
  TargetRegistry::RegisterAsmTargetStreamer(*T, createCpu0AsmTargetStreamer);

    static MCTargetStreamer *createCpu0AsmTargetStreamer(MCStreamer &S,
                                                         formatted_raw_ostream &OS,
                                                         MCInstPrinter *InstPrint,
                                                         bool isVerboseAsm) {
      return new Cpu0TargetAsmStreamer(S, OS);
    }

      // Cpu0TargetStreamer.h
      class Cpu0TargetStreamer : public MCTargetStreamer {
      public:
        Cpu0TargetStreamer(MCStreamer &S);
      };

      // This part is for ascii assembly output
      class Cpu0TargetAsmStreamer : public Cpu0TargetStreamer {
        formatted_raw_ostream &OS;

      public:
        Cpu0TargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
      };

Above instancing MCTargetStreamer instance.

.. rubric:: Register function of MC Code Emitter
.. code-block:: c++

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheCpu0Target,
                                        createCpu0MCCodeEmitterEB);
  TargetRegistry::RegisterMCCodeEmitter(TheCpu0elTarget,
                                        createCpu0MCCodeEmitterEL);

    // Cpu0MCCodeEmitter.cpp
    MCCodeEmitter *llvm::createCpu0MCCodeEmitterEB(const MCInstrInfo &MCII,
                                                   const MCRegisterInfo &MRI,
                                                   MCContext &Ctx) {
      return new Cpu0MCCodeEmitter(MCII, Ctx, false);
    }

    MCCodeEmitter *llvm::createCpu0MCCodeEmitterEL(const MCInstrInfo &MCII,
                                                   const MCRegisterInfo &MRI,
                                                   MCContext &Ctx) {
      return new Cpu0MCCodeEmitter(MCII, Ctx, true);
    }

Above, two `Cpu0MCCodeEmitter` objects are instantiated. One for big-endian  
and the other for little-endian. They handle the object format generation,  
while `RegisterELFStreamer()` reuses the ELF streamer class.

Readers may wonder:  
"What are the actual arguments in  
`createCpu0MCCodeEmitterEB(const MCInstrInfo &MCII,  
const MCSubtargetInfo &STI, MCContext &Ctx)`?"  
and "When are they assigned?"

At this point, we have not assigned them yet. Instead, we register the  
`createXXX()` function using a function pointer (according to C,  
`TargetRegistry::RegisterXXX(TheCpu0Target, createXXX())`, where  
`createXXX` is a function pointer).

LLVM stores a function pointer to `createXXX()` when we call target  
registration. Later, during the target registration process (`RegisterXXX()`),  
LLVM invokes these `createXXX()` functions with the necessary arguments (LLVM
invokes these arguments during run time in LLVM structure)`.

.. rubric:: Register function of asm backend
.. code-block:: c++

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheCpu0Target,
                                       createCpu0AsmBackendEB32);
  TargetRegistry::RegisterMCAsmBackend(TheCpu0elTarget,
                                       createCpu0AsmBackendEL32);

    // Cpu0AsmBackend.cpp
    MCAsmBackend *llvm::createCpu0AsmBackendEL32(const Target &T,
                                                 const MCRegisterInfo &MRI,
                                                 const Triple &TT, StringRef CPU) {
      return new Cpu0AsmBackend(T, TT.getOS(), /*IsLittle*/true);
    }

    MCAsmBackend *llvm::createCpu0AsmBackendEB32(const Target &T,
                                                 const MCRegisterInfo &MRI,
                                                 const Triple &TT, StringRef CPU) {
      return new Cpu0AsmBackend(T, TT.getOS(), /*IsLittle*/false);
    }

      // Cpu0AsmBackend.h
      class Cpu0AsmBackend : public MCAsmBackend {
      ...
      }

The `Cpu0AsmBackend` class serves as the bridge between assembly and object  
file generation. Two instances handle big-endian and little-endian formats,  
respectively.  

This class is derived from `MCAsmBackend`.  
Most of the code responsible for object file generation is implemented in  
`MCELFStreamer` and its parent class, `MCAsmBackend`.


.. [#target-registration] http://jonathan2251.github.io/lbd/llvmstructure.html#target-registration

.. [#add-asmprinter] http://jonathan2251.github.io/lbd/backendstructure.html#add-asmprinter
