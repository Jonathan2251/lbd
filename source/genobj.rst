.. _sec-genobjfiles:

Generating object files
=======================

.. contents::
   :local:
   :depth: 4

The previous chapters introducing the assembly code generation only. 
This chapter adding the elf obj support and verify the generated obj by 
objdump utility. With LLVM support, the Cpu0 backend can generate both big 
endian and little endian obj files with only a few code added.  
The Target Registration mechanism and their structure are introduced in 
this chapter.

Translate into obj file
------------------------

Currently, we only support translation of llvm IR code into assembly code. 
If you try to run Chapter4_2/ to translate it into obj code will get the error 
message as follows,

.. code-block:: bash

  [Gamma@localhost 3]$ ~/llvm/test/cmake_debug_build/bin/
  llc -march=cpu0 -relocation-model=pic -filetype=obj ch4_1.bc -o ch4_1.cpu0.o
  ~/llvm/test/cmake_debug_build/bin/llc: target does not 
  support generation of this file type! 
	
Chapter5_1/ support obj file generation. 
It produces obj files both for big endian and little endian with command 
``llc -march=cpu0`` and ``llc -march=cpu0el``, respectively. 
Run with them will get the obj files as follows,

.. code-block:: bash

  [Gamma@localhost input]$ cat ch4_1.cpu0.s 
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
  
  [Gamma@localhost 3]$ ~/llvm/test/cmake_debug_build/bin/
  llc -march=cpu0 -relocation-model=pic -filetype=obj ch4_1.bc -o ch4_1.cpu0.o
  [Gamma@localhost input]$ objdump -s ch4_1.cpu0.o 
  
  ch4_1.cpu0.o:     file format elf32-big 
  
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
   
  [Gamma@localhost input]$ ~/llvm/test/
  cmake_debug_build/bin/llc -march=cpu0el -relocation-model=pic -filetype=obj 
  ch4_1.bc -o ch4_1.cpu0el.o 
  [Gamma@localhost input]$ objdump -s ch4_1.cpu0el.o 
  
  ch4_1.cpu0el.o:     file format elf32-little 
  
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
         

The first instruction is **“addiu  $sp, -56”** and its corresponding obj is 
0x09ddffc8. 
The opcode of addiu is 0x09, 8 bits; $sp register number is 13(0xd), 4bits; and 
the immediate is 16 bits -56(=0xffc8), so it is correct. 
The third instruction **“st  $2, 52($fp)”** and it's corresponding obj 
is 0x022b0034. The **st** opcode is **0x02**, $2 is 0x2, $fp is 0xb and 
immediate is 52(0x0034). 
Thanks to Cpu0 instruction format which opcode, register operand and 
offset(imediate value) size are multiple of 4 bits. 
Base on the 4 bits multiple, the obj format is easy to check by eyes. 
The big endian (B0, B1, B2, B3) = (09, dd, ff, c8), objdump from B0 to B3 is 
0x09ddffc8 and the little endian is (B3, B2, B1, B0) = (09, dd, ff, c8), 
objdump from B0 to B3 is 0xc8ffdd09. 


ELF obj related code
----------------------

To support elf obj generation, the following code changed and added to 
Chapter5_1.

.. rubric:: lbdex/chapters/Chapter5_1/InstPrinter/Cpu0InstPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/InstPrinter/Cpu0InstPrinter.cpp
    :start-after: #if CH >= CH5_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/InstPrinter/Cpu0InstPrinter.cpp
    :start-after: #if CH >= CH5_1 //2
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


The applyFixup() of Cpu0AsmBackend.cpp will fix up the **jeq**, **jub**, ... 
instructions of "address control flow statements" or "function call statements" 
used in later chapters.
The setting of true or false for each relocation record in 
needsRelocateWithSymbol() of Cpu0ELFObjectWriter.cpp depends on whethor this 
relocation record is needed to adjust address value during link or not.
If set true, then linker has chance to adjust this address value with correct
information. On the other hand, if set false, then linker has no correct 
information to adjust this relocation record. About relocation record, it will
be introduced in later chapter ELF Support.

When emit elf obj format instruction, the EncodeInstruction() of 
Cpu0MCCodeEmitter.cpp will be called since it override the same name of 
function in parent class MCCodeEmitter. 

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


The "let EncoderMethod = "getMemEncoding";" in Cpu0InstrInfo.td as above will 
making llvm call function getMemEncoding() when either **ld** or **st** 
instruction is issued in elf obj since these two instructions use **mem** 
Operand.

The other functions in Cpu0MCCodeEmitter.cpp are called by these two functions.


Backend Target Registration Structure
--------------------------------------

Now, let's examine Cpu0MCTargetDesc.cpp.
Cpu0MCTargetDesc.cpp do the target registration as mentioned in 
the previous chapter here [#target-registration]_, and the assembly
output has explained here [#add-asmprinter]_.
List the register functions of ELF obj output as follows,

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

Above createELFStreamer takes care the elf obj streamer. 
:num:`Figure #genobj-f10` as follow is MCELFStreamer inheritance tree. 
You can find a lot of operations in that inheritance tree.

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

Above instancing two objects Cpu0MCCodeEmitter, one is for 
big endian and the other is for little endian. 
They take care the obj format generated while RegisterELFStreamer() reuse the
elf streamer class.

Reader maybe has the question: "What are the actual arguments in 
createCpu0MCCodeEmitterEB(const MCInstrInfo &MCII,  const MCSubtargetInfo &STI, 
MCContext &Ctx)?" and "When they are assigned?"
Yes, we didn't assign it at this point, we register the createXXX() function by 
function pointer only (according C, TargetRegistry::RegisterXXX(TheCpu0Target, 
createXXX()) where createXXX is function pointer). 
LLVM keeps a function pointer to createXXX() when we call target registry, and 
will call these createXXX() function back at proper time with arguments 
assigned during the target registration process, RegisterXXX().

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

Above Cpu0AsmBackend class is the bridge for asm to obj. 
Two objects take care big endian and little endian, respectively. 
It derived from MCAsmBackend. 
Most of code for object file generated is implemented by MCELFStreamer and it's 
parent, MCAsmBackend.



.. [#target-registration] http://jonathan2251.github.io/lbd/llvmstructure.html#target-registration

.. [#add-asmprinter] http://jonathan2251.github.io/lbd/backendstructure.html#add-asmprinter
