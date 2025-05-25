.. _sec-backendstructure:

Backend structure
==================

.. contents::
   :local:
   :depth: 4

From :numref:`llvmstructure-f9`, llvm compiler transfer from llvm-ir to assembly
or binary object with the following data structure as 
:numref:`backendstructure-lds`.

.. _backendstructure-lds:
.. graphviz:: ../Fig/backendstructure/llvm-data-structure.gv
  :caption: LLVM data structure used in different stages

Cpu0 backend supports the following backend compiler, assembler and disassembler 
as :numref:`backendstructure-function` and :numref:`backendstructure-enc-struct`.
However only the green part for printing assembly is implemented in this chapter.
Others are implemented in later chapters :numref:`genobj-f11`, :numref:`asm-flow` 
and :numref:`disas`.

.. _backendstructure-function:
.. graphviz:: ../Fig/backendstructure/cpu0-function.gv
  :caption: Backend compiler, assembler and disassembler of Cpu0

.. _backendstructure-enc-struct:
.. graphviz:: ../Fig/backendstructure/cpu0-enc-struct.gv
  :caption: The structure for backend compiler, assembler and disassembler of Cpu0

- Bytes: 4-byte (32-bits) for Cpu0.

- The emitInstruction() of Cpu0MCCodeEmitter.cpp: encode binary for an 
  instruction reused for both llc (compiler) and llc (assembler).

- The printInst() of Cpu0InstPrinter.cpp: print assembly code for an instruction
  reused for both llc (compiler) and llvm-objdump (disassembler).

This chapter first introduces the backend class inheritance tree and its class 
members. Then, following the backend structure, we add individual class 
implementations in each section. By the end of this chapter, we will have a 
backend capable of compiling LLVM intermediate code into Cpu0 assembly code.

Many lines of code are introduced in this chapter. Most of them are common 
across different backends, except for the backend name (e.g., Cpu0 or Mips). 
In fact, we copy almost all the code from Mips and replace the name with Cpu0. 
Beyond understanding DAG pattern matching in theoretical compilers and the LLVM 
code generation phase, please focus on the relationships between classes in 
this backend structure. Once you grasp the structure, you will be able to 
create your backend as quickly as we did, even though this chapter introduces 
around 5000 lines of code.

TargetMachine structure
-----------------------

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0TargetObjectFile.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0TargetObjectFile.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0TargetObjectFile.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0TargetObjectFile.cpp

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0TargetMachine.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0TargetMachine.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0TargetMachine.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0TargetMachine.cpp
  
.. rubric:: include/llvm/Target/TargetInstInfo.h
.. code-block:: c++

  class TargetInstrInfo : public MCInstrInfo { 
    TargetInstrInfo(const TargetInstrInfo &) = delete;
    void operator=(const TargetInstrInfo &) = delete;
  public: 
    ... 
  }
  ...
  class TargetInstrInfoImpl : public TargetInstrInfo { 
  protected: 
    TargetInstrInfoImpl(int CallFrameSetupOpcode = -1, 
              int CallFrameDestroyOpcode = -1) 
    : TargetInstrInfo(CallFrameSetupOpcode, CallFrameDestroyOpcode) {} 
  public: 
    ... 
  } 
  

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0.td
    :start-after: #if CH >= CH3_1 2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0CallingConv.td
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0CallingConv.td

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0FrameLowering.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0FrameLowering.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0FrameLowering.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0FrameLowering.cpp

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0SEFrameLowering.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0SEFrameLowering.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0SEFrameLowering.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0SEFrameLowering.cpp

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0InstrInfo.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0InstrInfo.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0InstrInfo.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0InstrInfo.cpp

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0ISelLowering.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0ISelLowering.cpp

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0SEISelLowering.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0SEISelLowering.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0SEISelLowering.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0SEISelLowering.cpp

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0MachineFunction.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0MachineFunction.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0MachineFunction.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0MachineFunction.cpp

.. rubric:: lbdex/chapters/Chapter3_1/MCTargetDesc/Cpu0ABIInfo.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/MCTargetDesc/Cpu0ABIInfo.h

.. rubric:: lbdex/chapters/Chapter3_1/MCTargetDesc/Cpu0ABIInfo.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/MCTargetDesc/Cpu0ABIInfo.cpp

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0Subtarget.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.h
    :start-after: #if CH >= CH3_1
    :end-before: #if CH >= CH6_1
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.h
    :start-after: //@1
    :end-before: #if CH >= CH6_1 //RM
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.h
    :start-after: #endif //TM
    :end-before: #if CH >= CH6_1 //hasSlt
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.h
    :start-after: #endif //abiUsesSoftFloat
    :end-before: #endif // #if CH >= CH3_1

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0Subtarget.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0Subtarget.cpp

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0RegisterInfo.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0RegisterInfo.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0RegisterInfo.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0RegisterInfo.cpp

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0SERegisterInfo.h
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0SERegisterInfo.h

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0SERegisterInfo.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_1/Cpu0SERegisterInfo.cpp

.. rubric:: build/lib/Target/Cpu0/Cpu0GenInstInfo.inc
.. code-block:: c++

  //- Cpu0GenInstInfo.inc which generate from Cpu0InstrInfo.td 
  #ifdef GET_INSTRINFO_HEADER 
  #undef GET_INSTRINFO_HEADER 
  namespace llvm { 
  struct Cpu0GenInstrInfo : public TargetInstrInfoImpl { 
    explicit Cpu0GenInstrInfo(int SO = -1, int DO = -1); 
  }; 
  } // End llvm namespace 
  #endif // GET_INSTRINFO_HEADER 
  
  #define GET_INSTRINFO_HEADER 
  #include "Cpu0GenInstrInfo.inc" 
  //- Cpu0InstInfo.h 
  class Cpu0InstrInfo : public Cpu0GenInstrInfo { 
    Cpu0TargetMachine &TM; 
  public: 
    explicit Cpu0InstrInfo(Cpu0TargetMachine &TM); 
  };

.. _backendstructure-f1: 
.. figure:: ../Fig/backendstructure/1.png
  :align: center

  Cpu0 backend class access link

Chapter3_1 adds most of the Cpu0 backend classes. The code in Chapter3_1 can be 
summarized in :numref:`backendstructure-f1`.  

The **Cpu0Subtarget** class provides interfaces such as `getInstrInfo()`, 
`getFrameLowering()`, etc., to access other Cpu0 classes. 
Most classes (like **Cpu0InstrInfo**, **Cpu0RegisterInfo**, etc.) contain a 
**Subtarget** reference member, allowing them to access other classes through 
the **Cpu0Subtarget** interface.  

If a backend module does not have a **Subtarget** reference, these classes can 
still access the **Subtarget** class through **Cpu0TargetMachine** (typically 
referred to as `TM`) using:  

.. code-block:: cpp

    static_cast<Cpu0TargetMachine &>(TM).getSubtargetImpl()

Once the **Subtarget** class is obtained, the backend code can access other 
classes through it.  

For classes named **Cpu0SExx**, they represent the standard 32-bit class. 
This naming convention follows the **LLVM 3.5 Mips backend** style. 
The Mips backend uses `Mips16`, `MipsSE`, and `Mips64` file/class names to 
define classes for 16-bit, 32-bit, and 64-bit architectures, respectively.  

Since **Cpu0Subtarget** creates **Cpu0InstrInfo**, **Cpu0RegisterInfo**, etc., 
in its constructor, it can provide class references through the interfaces 
shown in :numref:`backendstructure-f1`.  

:numref:`backendstructure-f2` below illustrates the **Cpu0 TableGen** 
inheritance relationship.  

In the previous chapter, we mentioned that backend classes can include 
TableGen-generated classes and inherit from them. 
All TableGen-generated classes for the Cpu0 backend are located in:  

.. code-block:: none

    build/lib/Target/Cpu0/*.inc

Through **C++ inheritance**, TableGen provides backend developers with a
flexible way to utilize its generated code. Developers also have the
opportunity to override functions if needed.

.. _backendstructure-f2:

.. figure:: ../Fig/backendstructure/2.png
   :align: center

   Cpu0 classes inherited from TableGen-generated files

Since LLVM has a deep inheritance tree, it is not fully explored here.
Benefiting from the inheritance tree structure, minimal code needs to be
implemented in instruction, frame/stack, and DAG selection classes, as much
of the functionality is already provided by their parent classes.

The `llvm-tblgen` tool generates **Cpu0GenInstrInfo.inc** based on the
information in **Cpu0InstrInfo.td**.

**Cpu0InstrInfo.h** extracts the necessary code from **Cpu0GenInstrInfo.inc**
by defining:

.. code-block:: c

    #define GET_INSTRINFO_HEADER

With TableGen, the backend code size is further reduced through the pattern
matching theory of compiler development. This is explained in the "DAG" and
"Instruction Selection" sections of the previous chapter.

The following is a code fragment from **Cpu0GenInstrInfo.inc**. Code between
`#ifdef GET_INSTRINFO_HEADER` and `#endif // GET_INSTRINFO_HEADER` is extracted
into **Cpu0InstrInfo.h**.

.. rubric:: build/lib/Target/Cpu0/Cpu0GenInstInfo.inc
.. code-block:: c++

  //- Cpu0GenInstInfo.inc which generate from Cpu0InstrInfo.td 
  #ifdef GET_INSTRINFO_HEADER 
  #undef GET_INSTRINFO_HEADER 
  namespace llvm { 
  struct Cpu0GenInstrInfo : public TargetInstrInfoImpl { 
    explicit Cpu0GenInstrInfo(int SO = -1, int DO = -1); 
  }; 
  } // End llvm namespace 
  #endif // GET_INSTRINFO_HEADER 

Reference web sites are here [#targetmachine]_ [#datalayout]_. 

Chapter3_1/CMakeLists.txt is modified with these new added \*.cpp as follows,

.. rubric:: lbdex/chapters/Chapter3_1/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH3_1 1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH3_1 2
    :end-before: #endif


Please take a look for Chapter3_1 code. 
After that, building Chapter3_1 by **"#define CH  CH3_1"** in Cpu0Config.h as 
follows, and do building with cmake and make again.

.. rubric:: ~/llvm/test/llvm/lib/Target/Cpu0SetChapter.h
.. code-block:: c++

  #define CH       CH3_1
  
.. code-block:: console

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc -o 
  ch3.cpu0.s
  ... Assertion failed: (MRI && "Unable to create reg info"), function initAsmInfo
  ...

With Chapter3_1 implementation, the Chapter2 error message 
"Could not allocate target machine!" has gone.
The new error say that we have not Target AsmPrinter and asm register info. 
We will add it in next section.

With the implementation of **Chapter3_1**, the **Chapter2** error message  
*"Could not allocate target machine!"* has been resolved.  
The new error indicates that we lack **Target AsmPrinter** and  
**ASM register info**. We will add these in the next section.  

**Chapter3_1** creates **FeatureCpu032I** and **FeatureCpu032II** for CPUs  
**cpu032I** and **cpu032II**, respectively.  
Additionally, it defines two more features: **FeatureCmp** and **FeatureSlt**.  

To demonstrate **instruction set design choices**, this book introduces two CPUs.  
Readers will understand why **MIPS CPUs** use the **SLT** instruction instead of  
**CMP** after reading the later chapter, *"Control Flow Statement"*.  

With the added support for **cpu032I** and **cpu032II** in `Cpu0.td` and  
`Cpu0InstrInfo.td` from **Chapter3_1**, running the command:  

.. code-block:: bash  

   llc -march=cpu0 -mcpu=help  

will display messages as follows:

.. code-block:: console
  
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/
  build/bin/llc -march=cpu0 -mcpu=help
  Available CPUs for this target:

    cpu032I  - Select the cpu032I processor.
    cpu032II - Select the cpu032II processor.

  Available features for this target:

    ch10_1   - Enable Chapter instructions..
    ch11_1   - Enable Chapter instructions..
    ch11_2   - Enable Chapter instructions..
    ch14_1   - Enable Chapter instructions..
    ch3_1    - Enable Chapter instructions..
    ch3_2    - Enable Chapter instructions..
    ch3_3    - Enable Chapter instructions..
    ch3_4    - Enable Chapter instructions..
    ch3_5    - Enable Chapter instructions..
    ch4_1    - Enable Chapter instructions..
    ch4_2    - Enable Chapter instructions..
    ch5_1    - Enable Chapter instructions..
    ch6_1    - Enable Chapter instructions..
    ch7_1    - Enable Chapter instructions..
    ch8_1    - Enable Chapter instructions..
    ch8_2    - Enable Chapter instructions..
    ch9_1    - Enable Chapter instructions..
    ch9_2    - Enable Chapter instructions..
    ch9_3    - Enable Chapter instructions..
    chall    - Enable Chapter instructions..
    cmp      - Enable 'cmp' instructions..
    cpu032I  - Cpu032I ISA Support.
    cpu032II - Cpu032II ISA Support (slt).
    o32      - Enable o32 ABI.
    s32      - Enable s32 ABI.
    slt      - Enable 'slt' instructions..

  Use +feature to enable a feature, or -feature to disable it.
  For example, llc -mcpu=mycpu -mattr=+feature1,-feature2
  ...

When the user inputs ``-mcpu=cpu032I``, the variable **IsCpu032I** from  
**Cpu0InstrInfo.td** will be **true** since the function **isCpu032I()** defined  
in **Cpu0Subtarget.h** returns **true**. This happens because **Cpu0ArchVersion**  
is set to **cpu032I** in **initializeSubtargetDependencies()**, which is called  
in the constructor. The variable **CPU** in the constructor is `"cpu032I"` when  
the user inputs ``-mcpu=cpu032I``.  

Please note that the variable **Cpu0ArchVersion** must be initialized in  
**Cpu0Subtarget.cpp**. Otherwise, **Cpu0ArchVersion** may hold an undefined value,  
causing issues with **isCpu032I()** and **isCpu032II()**, which support  
``llc -mcpu=cpu032I`` and ``llc -mcpu=cpu032II``, respectively.  

The values of the variables **HasCmp** and **HasSlt** depend on **Cpu0ArchVersion**.  
The instructions **slt**, **beq**, etc., are supported only if **HasSlt** is true.  
Furthermore, **HasSlt** is **true** only when **Cpu0ArchVersion** is **Cpu032II**.  

Similarly, **Ch4_1**, **Ch4_2**, etc., control the enabling or disabling of  
instruction definitions. Through **Subtarget->hasChapter4_1()**, which exists in  
both **Cpu0.td** and **Cpu0Subtarget.h**, predicates such as **Ch4_1**, defined in  
**Cpu0InstrInfo.td**, can be enabled or disabled.  

For example, the **shift-rotate** instructions can be enabled by defining **CH**  
to be greater than or equal to **CH4_1**, as shown below:

.. rubric:: lbdex/Cpu0/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH4_1 3
    :end-before: #endif

.. rubric:: ~/llvm/test/llvm/lib/Target/Cpu0SetChapter.h
.. code-block:: c++

  #define CH       CH4_1

On the contrary, it can be disabled by define it to less than CH4_1, for 
instance CH3_5, as follows,

.. rubric:: ~/llvm/test/llvm/lib/Target/Cpu0SetChapter.h
.. code-block:: c++

  #define CH       CH3_5


Add AsmPrinter
--------------

.. _asm-emit:
.. graphviz:: ../Fig/backendstructure/asm-emit.gv
   :caption: When "llc -filetype=asm", Cpu0AsmPrinter extract MCInst from MachineInstr for asm encoding

As :numref:`asm-emit`, because MachineInstr is a big class for opitmization and
convertion in many passes. LLVM creates MCInst for encoding purpose in assembly
and binary object.

Chapter3_2/ contains the Cpu0AsmPrinter definition. 

.. rubric:: lbdex/chapters/Chapter2/Cpu0.td
.. code-block:: c++

  def Cpu0InstrInfo : InstrInfo;

  // Will generate Cpu0GenAsmWrite.inc included by Cpu0InstPrinter.cpp, contents 
  //  as follows,
  // void Cpu0InstPrinter::printInstruction(const MCInst *MI, raw_ostream &O) {...}
  // const char *Cpu0InstPrinter::getRegisterName(unsigned RegNo) {...}
  def Cpu0 : Target {
  // def Cpu0InstrInfo : InstrInfo as before.
    let InstructionSet = Cpu0InstrInfo;
  }

As mentioned in the comments of **Chapter2/Cpu0.td**, it will generate  
**Cpu0GenAsmWriter.inc**, which is included by **Cpu0InstPrinter.cpp** as 
follows:

.. rubric:: lbdex/chapters/Chapter3_2/InstPrinter/Cpu0InstPrinter.h
.. literalinclude:: ../lbdex/chapters/Chapter3_2/InstPrinter/Cpu0InstPrinter.h

.. rubric:: lbdex/chapters/Chapter3_2/InstPrinter/Cpu0InstPrinter.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_2/InstPrinter/Cpu0InstPrinter.cpp

.. rubric:: lbdex/chapters/Chapter3_2/InstPrinter/CMakeLists.txt
.. literalinclude:: ../lbdex/chapters/Chapter3_2/InstPrinter/CMakeLists.txt

**Cpu0GenAsmWriter.inc** contains the implementations of  
**Cpu0InstPrinter::printInstruction()** and  
**Cpu0InstPrinter::getRegisterName()**. Both of these functions are  
auto-generated based on the information defined in **Cpu0InstrInfo.td**  
and **Cpu0RegisterInfo.td**.  

To enable these functions in our code, we only need to add a class  
**Cpu0InstPrinter** and include them, as demonstrated in **Chapter3_2**.  

The file **Chapter3_2/Cpu0/InstPrinter/Cpu0InstPrinter.cpp** includes  
**Cpu0GenAsmWriter.inc** and invokes the auto-generated functions from  
TableGen.  

The process of printing assembly and the interaction between  
**Cpu0InstPrinter.cpp** and **Cpu0GenAsmWriter.inc** is illustrated in  
:numref:`print-asm`.  

**Cpu0AsmPrinter::emitInstruction()** calls  
**Cpu0MCInstLower::Lower(const MachineInstr \*MI, MCInst &OutMI)**  
to extract **MCInst** from **MachineInstr**.  

.. _print-asm:
.. graphviz:: ../Fig/backendstructure/printAsm.gv
   :caption: The flow of printing assembly and calling between 
             Cpu0InstPrinter.cpp and Cpu0GenAsmWrite.inc

- AsmPrinter::OutStreamer is \nMCAsmStreamer if llc -filetype=asm; 
  AsmPrinter::OutStreamer is \nMCObjectStreamer if llc -filetype=obj as 
  :numref:`genobj-f11`.

- **Bits** is the format of instruction for **Opcode**, used to print ","  
  between operands.  

The function **Cpu0InstPrinter::printMemOperand()** is defined in  
**Chapter3_2/InstPrinter/Cpu0InstPrinter.cpp**, as shown above.  
This function is triggered because **Cpu0InstrInfo.td** defines  
**'let PrintMethod = "printMemOperand";'**, as shown below.  

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td
.. code:: text

  // Address operand
  def mem : Operand<i32> {
    let PrintMethod = "printMemOperand";
    let MIOperandInfo = (ops CPURegs, simm16);
    let EncoderMethod = "getMemEncoding";
  }
  ...
  // 32-bit load.
  multiclass LoadM32<bits<8> op, string instr_asm, PatFrag OpNode,
                     bit Pseudo = 0> {
    def #NAME# : LoadM<op, instr_asm, OpNode, GPROut, mem, Pseudo>;
  }

  // 32-bit store.
  multiclass StoreM32<bits<8> op, string instr_asm, PatFrag OpNode,
                      bit Pseudo = 0> {
    def #NAME# : StoreM<op, instr_asm, OpNode, CPURegs, mem, Pseudo>;
  }

  defm LD     : LoadM32<0x01,  "ld",  load_a>;
  defm ST     : StoreM32<0x02, "st",  store_a>;


Cpu0InstPrinter::printMemOperand() will print backend operands for "local 
variable access", which is like the following,

.. code-block:: console

	  ld	$2, 16($fp)
	  st	$2, 8($fp)

Next, add **Cpu0MCInstLower** (**Cpu0MCInstLower.h**, **Cpu0MCInstLower.cpp**)  
as well as **Cpu0BaseInfo.h**, **Cpu0FixupKinds.h**, and **Cpu0MCAsmInfo**  
(**Cpu0MCAsmInfo.h**, **Cpu0MCAsmInfo.cpp**) in the sub-directory  
**MCTargetDesc**, as shown below.  

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0MCInstLower.h
.. literalinclude:: ../lbdex/chapters/Chapter3_2/Cpu0MCInstLower.h

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0MCInstLower.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_2/Cpu0MCInstLower.cpp

.. rubric:: lbdex/chapters/Chapter3_2/MCTargetDesc/Cpu0BaseInfo.h
.. literalinclude:: ../lbdex/chapters/Chapter3_2/MCTargetDesc/Cpu0BaseInfo.h

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0MCAsmInfo.h
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCAsmInfo.h

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0MCAsmInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCAsmInfo.cpp

Finally, add code in **Cpu0MCTargetDesc.cpp** to register **Cpu0InstPrinter**  
as shown below. It also registers other classes (register, instruction,  
and subtarget) that were defined in **Chapter3_1** at this point.

.. rubric:: lbdex/chapters/Chapter3_2/MCTargetDesc/Cpu0MCTargetDesc.h
.. code-block:: c++

  namespace llvm {
  class MCAsmBackend;
  class MCCodeEmitter;
  class MCContext;
  class MCInstrInfo;
  class MCObjectWriter;
  class MCRegisterInfo;
  class MCSubtargetInfo;
  class StringRef;
  ...
  class raw_ostream;
  ...
  }

.. rubric:: lbdex/chapters/Chapter3_2/MCTargetDesc/Cpu0MCTargetDesc.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCTargetDesc.cpp
    :start-after: #if CH >= CH3_2 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/chapters/Chapter3_2/MCTargetDesc/Cpu0MCTargetDesc.cpp
    :start-after: //@1 {

.. rubric:: lbdex/chapters/Chapter3_2/MCTargetDesc/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/CMakeLists.txt
    :start-after: #if CH >= CH3_2
    :end-before: #endif


To make the registration clearly, summary as the following diagram, :numref:`backendstructure-f3`.

.. _backendstructure-f3: 
.. figure:: ../Fig/backendstructure/dyn_reg.png
  :align: center

  Tblgen generate files for Cpu0 backend


Above createCpu0MCAsmInfo() registering the object of class Cpu0MCAsmInfo for 
target TheCpu0Target and TheCpu0elTarget. 
TheCpu0Target is for big endian and TheCpu0elTarget is for little endian. 
Cpu0MCAsmInfo is derived from MCAsmInfo which is an llvm built-in class. 
Most code is implemented in it's parent, backend reuses those code by 
inheritance.

Above `createCpu0MCAsmInfo()` registers the `Cpu0MCAsmInfo` object for the  
targets `TheCpu0Target` (big-endian) and `TheCpu0elTarget` (little-endian).  
`Cpu0MCAsmInfo` is derived from `MCAsmInfo`, an LLVM built-in class.  
Most of its functionality is inherited from its parent class.  

Above `createCpu0MCInstrInfo()` instantiates an `MCInstrInfo` object `X` and  
initializes it using `InitCpu0MCInstrInfo(X)`.  
Since `InitCpu0MCInstrInfo(X)` is defined in `Cpu0GenInstrInfo.inc`,  
this function incorporates the instruction details specified in `Cpu0InstrInfo.td`.  

Above `createCpu0MCInstPrinter()` instantiates `Cpu0InstPrinter`,  
which handles instruction printing.  

Above `createCpu0MCRegisterInfo()` follows a similar approach to  
"Register function of MC instruction info" but initializes  
register information from `Cpu0RegisterInfo.td`.  
It reuses values from the instruction/register TableGen descriptions,  
eliminating redundancy in the initialization routine if they remain consistent.  

Above `createCpu0MCSubtargetInfo()` instantiates an `MCSubtargetInfo` object  
and initializes it with details from `Cpu0.td`.  

According to "section Target Registration" [#registration]_,  
we can register **Cpu0 backend** classes in `LLVMInitializeCpu0TargetMC()`  
using LLVM's dynamic registration mechanism.  

Now, it's time to work with **AsmPrinter**, as shown below.  


.. rubric:: lbdex/chapters/Chapter3_2/Cpu0AsmPrinter.h
.. literalinclude:: ../lbdex/chapters/Chapter3_2/Cpu0AsmPrinter.h

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0AsmPrinter.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_2/Cpu0AsmPrinter.cpp


When an instruction is ready to be printed, the function  
`Cpu0AsmPrinter::EmitInstruction()` is triggered first.  
It then calls `OutStreamer.EmitInstruction()` to print the opcode and  
register names based on the information from `Cpu0GenInstrInfo.inc`  
and `Cpu0GenRegisterInfo.inc`.  
Both files are registered dynamically in `LLVMInitializeCpu0TargetMC()`.  

Note that `Cpu0InstPrinter.cpp` only prints the operands,  
while the opcode information comes from `Cpu0InstrInfo.td`.  

Add the following code to `Cpu0ISelLowering.cpp`.

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0ISelLowering.cpp
.. code-block:: c++

  Cpu0TargetLowering::
  Cpu0TargetLowering(Cpu0TargetMachine &TM)
    : TargetLowering(TM, new Cpu0TargetObjectFile()),
      Subtarget(&TM.getSubtarget<Cpu0Subtarget>()) {

  //- Set .align 2
  // It will emit .align 2 later
    setMinFunctionAlignment(2);

  // must, computeRegisterProperties - Once all of the register classes are 
  //  added, this allows us to compute derived properties we expose.
    computeRegisterProperties();
  }

Add the following code to `Cpu0MachineFunction.h`  
since `Cpu0AsmPrinter.cpp` will call `getEmitNOAT()`.  

.. rubric:: lbdex/chapters/Chapter3_2/Cpu0MachineFunction.h
.. code-block:: c++

  class Cpu0FunctionInfo : public MachineFunctionInfo {
  public:
    Cpu0FunctionInfo(MachineFunction& MF)
    : ...
      , EmitNOAT(false)
      {}

    ...
    bool getEmitNOAT() const { return EmitNOAT; }
    void setEmitNOAT() { EmitNOAT = true; }
  private:
    ...
    bool EmitNOAT;
  };


.. rubric:: lbdex/chapters/Chapter3_2/CMakeLists.txt
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH3_2 1
    :end-before: #endif

.. code-block:: c++

  ...
  add_llvm_target(Cpu0CodeGen

.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH3_2 2
    :end-before: #endif

.. code-block:: c++

    LINK_COMPONENTS

    ...
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH3_2 3
    :end-before: #endif

.. code-block:: c++

    ...
    )
  ...
 
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH3_2 4
    :end-before: #endif

Now, run `Chapter3_2/Cpu0` for AsmPrinter support,  
and you will get a new error message as follows,  

.. code-block:: console

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc -o 
  ch3.cpu0.s
  /Users/Jonathan/llvm/test/build/bin/llc: target does not 
  support generation of this file type!

The ``llc`` fails to compile IR code into machine code  
since we haven't implemented the class `Cpu0DAGToDAGISel`.


Add Cpu0DAGToDAGISel class
--------------------------

The IR DAG to machine instruction DAG transformation  
was introduced in the previous chapter.  

Now, let's check what IR DAG nodes the file `ch3.bc` contains.  
List `ch3.ll` as follows:

.. code:: text

  // ch3.ll
  define i32 @main() nounwind uwtable { 
  %1 = alloca i32, align 4 
  store i32 0, i32* %1 
  ret i32 0 
  } 

As above, `ch3.ll` uses the IR DAG node **store**, **ret**.  
So, the definitions in `Cpu0InstrInfo.td` as below are enough.  
The `ADDiu` is used for stack adjustment, which will be needed  
in the later section *"Add Prologue/Epilogue functions"* of this chapter.  

IR DAG is defined in the file `include/llvm/Target/TargetSelectionDAG.td`.

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td
.. code-block:: c++

  //===----------------------------------------------------------------------===//

  /// Load and Store Instructions
  ///  aligned
  defm LD     : LoadM32<0x01,  "ld",  load_a>;
  defm ST     : StoreM32<0x02, "st",  store_a>;

  /// Arithmetic Instructions (ALU Immediate)
  // IR "add" defined in include/llvm/Target/TargetSelectionDAG.td, line 315 (def add).
  def ADDiu   : ArithLogicI<0x09, "addiu", add, simm16, immSExt16, CPURegs>;

  let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1, hasCtrlDep=1 in
  def RetLR : Cpu0Pseudo<(outs), (ins), "", [(Cpu0Ret)]>;
  
  def RET     : RetBase<GPROut>;

Add class `Cpu0DAGToDAGISel` (`Cpu0ISelDAGToDAG.cpp`) to `CMakeLists.txt`,  
and add the following fragment to `Cpu0TargetMachine.cpp`,

.. rubric:: lbdex/chapters/Chapter3_3/CMakeLists.txt
.. code-block:: c++

  add_llvm_target(
    ...
  
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH3_3
    :end-before: #endif
  
.. code-block:: c++
  
    ...
  )

The following code in `Cpu0TargetMachine.cpp` will create a pass in the  
instruction selection stage.

.. rubric:: lbdex/chapters/Chapter3_3/Cpu0TargetMachine.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH3_3 //0.5
    :end-before: #endif

.. code-block:: c++

  ...
  class Cpu0PassConfig : public TargetPassConfig {
  public:
    ...

.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH3_3 //1
    :end-before: #endif
	
.. code-block:: c++

  };
  ...

.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH3_3 //2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter3_3/Cpu0ISelDAGToDAG.h
.. literalinclude:: ../lbdex/chapters/Chapter3_3/Cpu0ISelDAGToDAG.h

.. rubric:: lbdex/chapters/Chapter3_3/Cpu0ISelDAGToDAG.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_3/Cpu0ISelDAGToDAG.cpp

.. rubric:: lbdex/chapters/Chapter3_3/Cpu0SEISelDAGToDAG.h
.. literalinclude:: ../lbdex/chapters/Chapter3_3/Cpu0SEISelDAGToDAG.h

.. rubric:: lbdex/chapters/Chapter3_3/Cpu0ISelDAGToDAG.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_3/Cpu0SEISelDAGToDAG.cpp

Function `Cpu0DAGToDAGISel::Select()` of `Cpu0ISelDAGToDAG.cpp` is for the  
selection of "OP code DAG node," while `Cpu0DAGToDAGISel::SelectAddr()` is for  
the selection of "DATA DAG node with **addr** type," which is defined in  
`Chapter2/Cpu0InstrInfo.td`. This method's name corresponds to  
`Chapter2/Cpu0InstrInfo.td` as follows:

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td
.. code-block:: c++

  def addr : ComplexPattern<iPTR, 2, "SelectAddr", [frameindex], [SDNPWantParent]>;

The `iPTR`, `ComplexPattern`, `frameindex`, and `SDNPWantParent` are defined  
as follows:

.. rubric:: llvm/include/llvm/Target/TargetSelection.td
.. code-block:: c++

  def SDNPWantParent  : SDNodeProperty;   // ComplexPattern gets the parent
  ...
  def frameindex  : SDNode<"ISD::FrameIndex",           SDTPtrLeaf, [],
                           "FrameIndexSDNode">;
  ...
  // Complex patterns, e.g. X86 addressing mode, requires pattern matching code
  // in C++. NumOperands is the number of operands returned by the select function;
  // SelectFunc is the name of the function used to pattern match the max. pattern;
  // RootNodes are the list of possible root nodes of the sub-dags to match.
  // e.g. X86 addressing mode - def addr : ComplexPattern<4, "SelectAddr", [add]>;
  //
  class ComplexPattern<ValueType ty, int numops, string fn,
                       list<SDNode> roots = [], list<SDNodeProperty> props = []> {
    ValueType Ty = ty;
    int NumOperands = numops;
    string SelectFunc = fn;
    list<SDNode> RootNodes = roots;
    list<SDNodeProperty> Properties = props;
  }


.. rubric:: llvm/include/llvm/CodeGen/ValueTypes.td
.. code-block:: c++

  // Pseudo valuetype mapped to the current pointer size.
  def iPTR   : ValueType<0  , 255>;

Build Chapter3_3 and run it. The error message from Chapter3_2 is gone.  
The new error message for Chapter3_3 is as follows:

.. code-block:: console

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc -o 
  ch3.cpu0.s
  ...
  LLVM ERROR: Cannot select: t6: ch = Cpu0ISD::Ret t4, Register:i32 $lr
    t5: i32 = Register $lr
  ...

The above can display the error message for the DAG node "Cpu0ISD::Ret"  
because the following code was added in Chapter3_1/Cpu0ISelLowering.cpp.

.. rubric:: lbdex/chapters/Chapter3_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@3_1 1 {
    :end-before: #if CH >= CH12_1 //0.5
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #endif //#if CH >= CH12_1 //0.5
    :end-before: //@3_1 1 }


Handle return register \$lr 
-----------------------------

The following code is the result of running the Mips backend with ch3.cpp.

.. code-block:: console
  
  JonathantekiiMac:input Jonathan$ ~/llvm/debug/build/bin/llc 
  -march=mips -relocation-model=pic -filetype=asm ch3.bc -o -
    .text
    .abicalls
    .section  .mdebug.abi32,"",@progbits
    .nan  legacy
    .file "ch3.bc"
    .text
    .globl  main
    .align  2
    .type main,@function
    .set  nomicromips
    .set  nomips16
    .ent  main
  main:                                   # @main
    .frame  $fp,8,$ra
    .mask   0x40000000,-4
    .fmask  0x00000000,0
    .set  noreorder
    .set  nomacro
    .set  noat
  # BB#0:
    addiu $sp, $sp, -8
    sw  $fp, 4($sp)             # 4-byte Folded Spill
    move   $fp, $sp
    sw  $zero, 0($fp)
    addiu $2, $zero, 0
    move   $sp, $fp
    lw  $fp, 4($sp)             # 4-byte Folded Reload
    jr  $ra
    addiu $sp, $sp, 8
    .set  at
    .set  macro
    .set  reorder
    .end  main
  $func_end0:
    .size main, ($func_end0)-main

As you can see, Mips returns to the caller by using "jr $ra", where $ra is a 
specific register that holds the caller's next instruction address. It also 
stores the return value in register $2. 

If we only create DAGs directly, we encounter the following two problems:

1. LLVM can allocate any register for the return value, such as $3, rather 
   than keeping it in $2.

2. LLVM may randomly allocate a register for "jr" since "jr" requires one 
   operand. For example, it might generate "jr $8" instead of "jr $ra".

If the backend strictly uses the "jal sub-routine" and "jr" while always 
storing the return address in the specific register $ra, the second problem 
does not occur. However, in Mips, programmers are allowed to use "jal $rx, 
sub-routine" and "jr $rx", where $rx is not necessarily $ra. 

Allowing programmers to use registers other than $ra provides more flexibility 
for high-level languages such as C when integrating assembly. 

The following file, `ch8_2_longbranch.cpp`, demonstrates this concept. It uses 
"jr $1" without spilling the $ra register. This optimization can significantly 
improve performance, especially in hot functions.

.. rubric:: lbdex/input/ch8_2_longbranch.cpp
.. literalinclude:: ../lbdex/input/ch8_2_longbranch.cpp
    :start-after: /// start

.. code-block:: console
  
  JonathantekiiMac:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch8_2_longbranch.cpp -emit-llvm -o ch8_2_longbranch.bc
  JonathantekiiMac:input Jonathan$ ~/llvm/debug/build/bin/llc 
  -march=mips -relocation-model=pic -filetype=asm -force-mips-long-branch 
  ch8_2_longbranch.bc -o -
    ...
    .ent  _Z15test_longbranchv
  _Z15test_longbranchv:                   # @_Z15test_longbranchv
    .frame  $fp,16,$ra
    .mask   0x40000000,-4
    .fmask  0x00000000,0
    .set  noreorder
    .set  nomacro
    .set  noat
  # BB#0:
    addiu $sp, $sp, -16
    sw  $fp, 12($sp)            # 4-byte Folded Spill
    move   $fp, $sp
    addiu $1, $zero, 2
    sw  $1, 8($fp)
    addiu $2, $zero, 1
    sw  $2, 4($fp)
    sw  $zero, 0($fp)
    lw  $1, 8($fp)
    lw  $3, 4($fp)
    slt $1, $1, $3
    bnez  $1, $BB0_3
    nop
  # BB#1:
    addiu $sp, $sp, -8
    sw  $ra, 0($sp)
    lui $1, %hi(($BB0_4)-($BB0_2))
    bal $BB0_2
    addiu $1, $1, %lo(($BB0_4)-($BB0_2))
  $BB0_2:
    addu  $1, $ra, $1
    lw  $ra, 0($sp)
    jr  $1
    addiu $sp, $sp, 8
  $BB0_3:
    sw  $2, 0($fp)
  $BB0_4:
    lw  $2, 0($fp)
    move   $sp, $fp
    lw  $fp, 12($sp)            # 4-byte Folded Reload
    jr  $ra
    addiu $sp, $sp, 16
    .set  at
    .set  macro
    .set  reorder
    .end  _Z15test_longbranchv
  $func_end0:
    .size _Z15test_longbranchv, ($func_end0)-_Z15test_longbranchv

The following code handles the return register $lr.

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0CallingConv.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0CallingConv.td
    :start-after: #if CH >= CH3_4 1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0CallingConv.td
    :start-after: #if CH >= CH3_4 2
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0InstrFormats.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrFormats.td
    :start-after: #if CH >= CH3_4
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_4 12
    :end-before: #endif


.. rubric:: lbdex/chapters/Chapter3_4/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/chapters/Chapter3_4/Cpu0ISelLowering.h
    :start-after: //@CH3_4 1 {
    :end-before: //@CH3_4 1 }

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH3_1 //LowerReturn
    :end-before: #if CH >= CH3_4 //in LowerReturn
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH3_4 //in LowerReturn
    :end-before: #else // #if CH >= CH3_4

.. code-block:: c++

  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH3_4 //analyzeReturn
    :end-before: #endif // #if CH >= CH3_4 //analyzeReturn

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH3_4 //reservedArgArea
    :end-before: #endif

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH3_4 //getRegVT
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0MachineFunction.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: //@1 {
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH3_4 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH3_4 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH3_4 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH3_4 //4
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH3_4 //5
    :end-before: #endif

.. code-block:: c++

  }

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0SEInstrInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.h
    :start-after: #if CH >= CH3_4 //1
    :end-before: #endif //#if CH >= CH3_4 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.h
    :start-after: #if CH >= CH3_4 //2
    :end-before: #endif //#if CH >= CH3_4 //2

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0SEInstrInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: #if CH >= CH3_4 //1
    :end-before: #if CH >= CH9_3 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: #endif //#if CH >= CH9_3 //1
    :end-before: #endif //#if CH >= CH3_4 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: #if CH >= CH3_4 //2
    :end-before: #endif //#if CH >= CH3_4 //2
    
Build Chapter3_4 and run with it, finding the error message in Chapter3_3 is
gone. The compilation result will hang, and please press "Ctrl+C" to abort 
as follows,

.. code-block:: console
  
  118-165-78-230:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch3.cpp -emit-llvm -o ch3.bc
  118-165-78-230:input Jonathan$ ~/llvm/test/build/bin/llvm-dis 
  ch3.bc -o -
  ...
  define i32 @main() #0 {
    %1 = alloca i32, align 4
    store i32 0, i32* %1
    ret i32 0
  }

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc -o -
    ...
    .text
    .section .mdebug.abiO32
    .previous
    .file "ch3.bc"
  ^C
  
It hangs because the Cpu0 backend has not handled stack slots for local 
variables. The instruction "store i32 0, i32* %1" in the above IR requires 
Cpu0 to allocate a stack slot and save to it.

However, ch3.cpp can be run with the option ``clang -O2`` as follows,

.. code-block:: console
  
  118-165-78-230:input Jonathan$ clang -O2 -target mips-unknown-linux-gnu -c 
  ch3.cpp -emit-llvm -o ch3.bc
  118-165-78-230:input Jonathan$ ~/llvm/test/build/bin/llvm-dis 
  ch3.bc -o -
  ...
  define i32 @main() #0 {
    ret i32 0
  }

  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc -o -
    .text
    .section .mdebug.abiO32
    .previous
    .file "ch3.bc"
    .globl  main
    .align  2
    .type main,@function
    .ent  main                    # @main
  main:
    .frame  $sp,0,$lr
    .mask   0x00000000,0
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $2, $zero, 0
    ret $lr
    .set  macro
    .set  reorder
    .end  main
  $func_end0:
    .size main, ($func_end0)-main

To see how the **'DAG->DAG Pattern Instruction Selection'** works in llc, 
let's compile with the option ``llc -print-before-all -print-after-all`` and 
get the following result. 

The DAGs before and after the instruction selection stage are shown below,

.. code-block:: console

  118-165-78-230:input Jonathan$ clang -O2 -target mips-unknown-linux-gnu -c 
  ch3.cpp -emit-llvm -o ch3.bc
  118-165-78-12:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm 
  -print-before-all -print-after-all ch3.bc -o -
  ...
  *** IR Dump After Module Verifier ***
  ; Function Attrs: nounwind readnone
  define i32 @main() #0 {
    ret i32 0
  }
  ...
  Initial selection DAG: BB#0 'main:'
  SelectionDAG has 5 nodes:
      t0: ch = EntryToken
    t3: ch,glue = CopyToReg t0, Register:i32 %V0, Constant:i32<0>
    t4: ch = Cpu0ISD::Ret t3, Register:i32 %V0, t3:1
  ...
  ===== Instruction selection begins: BB#0 ''
  Selecting: t4: ch = Cpu0ISD::Ret t3, Register:i32 %V0, t3:1
  
  ISEL: Starting pattern match on root node: t4: ch = Cpu0ISD::Ret t3, Register:i32 %V0, t3:1
  
    Morphed node: t4: ch = RetLR Register:i32 %V0, t3, t3:1
  
  ISEL: Match complete!
  Selecting: t3: ch,glue = CopyToReg t0, Register:i32 %V0, Constant:i32<0>
  
  Selecting: t2: i32 = Register %V0
  
  Selecting: t1: i32 = Constant<0>
  
  ISEL: Starting pattern match on root node: t1: i32 = Constant<0>
  
    Initial Opcode index to 3158
    Morphed node: t1: i32 = ADDiu Register:i32 %ZERO, TargetConstant:i32<0>
  
  ISEL: Match complete!
  Selecting: t0: ch = EntryToken
  
  ===== Instruction selection ends:
  Selected selection DAG: BB#0 'main:'
  SelectionDAG has 7 nodes:
      t0: ch = EntryToken
      t1: i32 = ADDiu Register:i32 %ZERO, TargetConstant:i32<0>
    t3: ch,glue = CopyToReg t0, Register:i32 %V0, t1
    t4: ch = RetLR Register:i32 %V0, t3, t3:1
  ...
  ********** REWRITE VIRTUAL REGISTERS **********
  ********** Function: main
  ********** REGISTER MAP **********
  [%vreg0 -> %V0] GPROut
  
  0B  BB#0: derived from LLVM BB %0
  16B   %vreg0<def> = ADDiu %ZERO, 0; GPROut:%vreg0
  32B   %V0<def> = COPY %vreg0<kill>; GPROut:%vreg0
  48B   RetLR %V0<imp-use>
  > %V0<def> = ADDiu %ZERO, 0
  > %V0<def> = COPY %V0<kill>
  Identity copy: %V0<def> = COPY %V0<kill>
    deleted.
  > RetLR %V0<imp-use>
  # *** IR Dump After Virtual Register Rewriter ***:
  # Machine code for function main: Properties: <Post SSA, tracking liveness, AllVRegsAllocated>
  
  0B  BB#0: derived from LLVM BB %0
  16B   %V0<def> = ADDiu %ZERO, 0
  48B   RetLR %V0<imp-use>
  ...
  ********** EXPANDING POST-RA PSEUDO INSTRS **********
  ********** Function: main
  # *** IR Dump After Post-RA pseudo instruction expansion pass ***:
  # Machine code for function main: Properties: <Post SSA, tracking liveness, AllVRegsAllocated>
  
  BB#0: derived from LLVM BB %0
    %V0<def> = ADDiu %ZERO, 0
    RET %LR
  ...
  
    .globl  main
    .p2align  2
    .type main,@function
    .ent  main                    # @main
  main:
    .frame  $sp,0,$lr
    .mask   0x00000000,0
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $2, $zero, 0
    ret $lr
    .set  macro
    .set  reorder
    .end  main
  $func_end0:
    .size main, ($func_end0)-main
  
  
    .ident  "Apple LLVM version 7.0.0 (clang-700.1.76)"
    .section  ".note.GNU-stack","",@progbits


Summary above translation into Table: Chapter 3 .bc IR instructions.

.. table:: Chapter 3 .bc IR instructions

  =============================  ==================================  ===============  =======  =======  ==========
  .bc                            Lower                               ISel             RVR      Post-RA  AsmP
  =============================  ==================================  ===============  =======  =======  ==========
  constant 0                     constant 0                          ADDiu            ADDiu    ADDiu    addiu  
  ret                            Cpu0ISD::Ret                        CopyToReg,RetLR  RetLR    RET      ret
  =============================  ==================================  ===============  =======  =======  ==========

- Lower: Initial selection DAG (Cpu0ISelLowering.cpp, LowerReturn(...))

- ISel: Instruction selection

- RVR: REWRITE VIRTUAL REGISTERS, remove CopyToReg

- AsmP: Cpu0 Asm Printer

- Post-RA: Post-RA pseudo instruction expansion pass

From the above ``llc -print-before-all -print-after-all`` display, we observe 
that **ret** is translated into **Cpu0ISD::Ret** in the stage of Optimized 
Legalized Selection DAG, and then finally translated into the Cpu0 instruction 
**ret**.

Since **ret** uses **constant 0** (**ret i32 0** in this example), the constant 
0 is translated into **"addiu $2, $zero, 0"** via the following pattern defined 
in **Cpu0InstrInfo.td**.

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 17
    :end-before: //#endif

In order to handle the IR **ret**, the following codes in **Cpu0InstrInfo.td** 
perform the following tasks:

1. Declare a pseudo node **Cpu0::RetLR** to handle the IR **Cpu0ISD::Ret** 
   with the following code:

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 2
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH3_4 12
    :end-before: //#endif

2. Create the **Cpu0ISD::Ret** node in **LowerReturn()** of 
   **Cpu0ISelLowering.cpp**, which is called when encountering the **return** 
   keyword in C. Reminder: In **LowerReturn()**, the return value is placed 
   in register **$2 ($v0)**.
   
If we use the following code in Cpu0, then V0 won't **live out**.

.. rubric:: lbdex/chapters/Chapter3_4/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH3_1 //LowerReturn
    :end-before: #if CH >= CH3_4 //in LowerReturn
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #else // #if CH >= CH3_4
    :end-before: #endif

.. code-block:: c++

  }


3. After instruction selection, the **Cpu0ISD::Ret** node is replaced by 
   **Cpu0::RetLR** as shown below. This effect comes from the **"def RetLR"** 
   declaration in step 1.

.. code-block:: console

  ===== Instruction selection begins: BB#0 'entry'
  Selecting: 0x1ea4050: ch = Cpu0ISD::Ret 0x1ea3f50, 0x1ea3e50, 
  0x1ea3f50:1 [ID=27]

  ISEL: Starting pattern match on root node: 0x1ea4050: ch = Cpu0ISD::Ret 
  0x1ea3f50, 0x1ea3e50, 0x1ea3f50:1 [ID=27]

    Morphed node: 0x1ea4050: ch = RetLR 0x1ea3e50, 0x1ea3f50, 0x1ea3f50:1
  ...
  ISEL: Match complete!
  => 0x1ea4050: ch = RetLR 0x1ea3e50, 0x1ea3f50, 0x1ea3f50:1
  ...
  ===== Instruction selection ends:
  Selected selection DAG: BB#0 'main:entry'
  SelectionDAG has 28 nodes:
  ...
      0x1ea3e50: <multiple use>
      0x1ea3f50: <multiple use>
      0x1ea3f50: <multiple use>
    0x1ea4050: ch = RetLR 0x1ea3e50, 0x1ea3f50, 0x1ea3f50:1

4. Expand the **Cpu0ISD::RetLR** into the instruction **Cpu0::RET $lr** during 
   the "Post-RA pseudo instruction expansion pass" stage using the code in 
   **Chapter3_4/Cpu0SEInstrInfo.cpp** as mentioned above. This stage occurs 
   after register allocation, so we can replace **V0 ($r2)** with **LR ($lr)** 
   without any side effects.

5. Print assembly or object code based on the information from **.inc** files 
   generated by TableGen from **.td** files during the "Cpu0 Assembly Print" 
   stage.

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 12
    :end-before: //#if CH >= CH10_1 1.5

.. code-block:: c++

  }
  
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 15
    :end-before: //#endif


.. table:: Handle return register lr

  ===================================================  ========================================
  Stage                                                Function   
  ===================================================  ========================================
  Write Code                                           Declare a pseudo node Cpu0::RetLR
  -                                                    for IR Cpu0::Ret;
  Before CPU0 DAG->DAG Pattern Instruction Selection   Create Cpu0ISD::Ret DAG
  Instruction selection                                Cpu0::Ret is replaced by Cpu0::RetLR
  Post-RA pseudo instruction expansion pass            Cpu0::RetLR -> Cpu0::RET \$lr
  Cpu0 Assembly Printer                                Print according "def RET"
  ===================================================  ========================================

The function **LowerReturn()** in *Cpu0ISelLowering.cpp* correctly handles the 
return variable. 

In **Chapter3_4/Cpu0ISelLowering.cpp**, the function **LowerReturn()** creates 
the **Cpu0ISD::Ret** node, which is called by the LLVM system when it encounters 
the `return` keyword in C.



More specifically, it constructs the DAG as follows:

   **Cpu0ISD::Ret (CopyToReg %X, %V0, %Y), %V0, Flag**


Add Prologue/Epilogue functions
-------------------------------

Concept
+++++++

Following come from tricore_llvm.pdf section “4.4.2 Non-static Register 
Information ”.

For some target architectures, some aspects of the target architecture’s 
register set are dependent upon variable factors and have to be determined at 
runtime. 
As a consequence, they cannot be generated statically from a TableGen 
description – although that would be possible for the bulk of them in the case 
of the TriCore backend. 
Among them are the following points:

- Callee-saved registers. Normally, the ABI specifies a set of registers that a 
  function must save on entry and restore on return if their contents are 
  possibly modified during execution.

- Reserved registers. Although the set of unavailable registers is already 
  defined in the TableGen file, TriCoreRegisterInfo contains a method that marks 
  all non-allocatable register numbers in a bit vector. 

The following methods are implemented:

- emitPrologue() inserts prologue code at the beginning of a function. Thanks 
  to TriCore’s context model, this is a trivial task as it is not required to 
  save any registers manually. The only thing that has to be done is reserving 
  space for the function’s stack frame by decrementing the stack pointer. 
  In addition, if the function needs a frame pointer, the frame register %a14 is 
  set to the old value of the stack pointer beforehand.

- emitEpilogue() is intended to emit instructions to destroy the stack frame 
  and restore all previously saved registers before returning from a function. 
  However, as %a10 (stack pointer), %a11 (return address), and %a14 (frame 
  pointer, if any) are all part of the upper context, no epilogue code is needed 
  at all. All cleanup operations are performed implicitly by the ret instruction. 

- eliminateFrameIndex() is called for each instruction that references a word 
  of data in a stack slot. All previous passes of the code generator have been 
  addressing stack slots through an abstract frame index and an immediate offset. 
  The purpose of this function is to translate such a reference into a 
  register–offset pair. Depending on whether the machine function that contains 
  the instruction has a fixed or a variable stack frame, either the stack pointer 
  %a10 or the frame pointer %a14 is used as the base register. 
  The offset is computed accordingly. 
  :numref:`backendstructure-f10` demonstrates for both cases how a stack slot 
  is addressed. 

If the addressing mode of the affected instruction cannot handle the address 
because the offset is too large (the offset field has 10 bits for the BO 
addressing mode and 16 bits for the BOL mode), a sequence of instructions is 
emitted that explicitly computes the effective address. 
Interim results are put into an unused address register. 
If none is available, an already occupied address register is scavenged. 
For this purpose, LLVM’s framework offers a class named RegScavenger that 
takes care of all the details.

.. _backendstructure-f10: 
.. figure:: ../Fig/backendstructure/10.png
  :align: center

  Addressing of a variable a located on the stack. 
  If the stack frame has a variable size, slot must be addressed relative to 
  the frame pointer


Prologue and Epilogue functions
++++++++++++++++++++++++++++++++

The Prologue and Epilogue functions as follows,
    
.. rubric:: lbdex/chapters/Chapter3_5/Cpu0SEFrameLowering.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_5/Cpu0SEFrameLowering.cpp
    :start-after: //@emitPrologue {
    :end-before: //}
.. literalinclude:: ../lbdex/chapters/Chapter3_5/Cpu0SEFrameLowering.cpp
    :start-after: //@emitEpilogue {
    :end-before: //}
.. literalinclude:: ../lbdex/chapters/Chapter3_5/Cpu0SEFrameLowering.cpp
    :start-after: //@hasReservedCallFrame {
    :end-before: //}

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0MachineFunction.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH3_5
    :end-before: #endif

Now, we further explain the **Prologue** and **Epilogue** with an example.

For the following LLVM IR code of *ch3.cpp*, **Chapter3_5** of the **Cpu0** 
backend will emit the corresponding machine instructions as follows:

.. code-block:: console

  118-165-78-230:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch3.cpp -emit-llvm -o ch3.bc
  118-165-78-230:input Jonathan$ ~/llvm/test/build/bin/llvm-dis 
  ch3.bc -o -
  ...
  define i32 @main() #0 {
    %1 = alloca i32, align 4
    store i32 0, i32* %1
    ret i32 0
  }
  
  118-165-78-230:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc -o -
    ...
    .section .mdebug.abi32
    .previous
    .file "ch3.bc"
    .text
    .globl  main//static void expandLargeImm\\n
    .align  2
    .type main,@function
    .ent  main                    # @main
  main:
    .cfi_startproc
    .frame  $sp,8,$lr
    .mask   0x00000000,0
    .set  noreorder
    .set  nomacro
  # BB#0:
    addiu $sp, $sp, -8
  $tmp1:
    .cfi_def_cfa_offset 8
    addiu $2, $zero, 0
    st  $2, 4($sp)
    addiu $sp, $sp, 8
    ret $lr
    .set  macro
    .set  reorder
    .end  main
  $tmp2:
    .size main, ($tmp2)-main
    .cfi_endproc

LLVM get the stack size by counting how many virtual registers is assigned to 
local variables. 
After that, it calls emitPrologue(). 

.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.h
    :start-after: #if CH >= CH3_5 //3
    :end-before: #endif
    
.. rubric:: lbdex/chapters/Chapter3_5/Cpu0SEInstrInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.h
    :start-after: #if CH >= CH3_5 //2
    :end-before: #endif //#if CH >= CH3_5 //2

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0SEInstrInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: #if CH >= CH3_5 //2
    :end-before: #endif //#if CH >= CH3_5 //2
    
In `emitPrologue()`, it emits machine instructions to adjust the **sp** 
(stack pointer register) for local variables.
For our example, it will emit the instruction:

.. code:: text

  addiu $sp, $sp, -8

In the above `ch3.cpp` assembly output, it generates:

.. code:: text

  addiu $2, $zero, 0

rather than:

.. code:: text

  ori $2, $zero, 0

because **ADDiu** is defined before **ORi** as shown below, so it takes 
priority. Of course, if **ORi** were defined first, it would translate 
into the **ori** instruction.

.. rubric:: lbdex/chapters/Chapter2/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 17
    :end-before: //#endif

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH3_5 14
    :end-before: //#endif


Handle stack slot for local variables
+++++++++++++++++++++++++++++++++++++

The following code handle the stack slot for local variables.

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0RegisterInfo.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_5/Cpu0RegisterInfo.cpp
    :start-after: //@eliminateFrameIndex {
    :end-before: //}

The `eliminateFrameIndex()` function in `Cpu0RegisterInfo.cpp` is called after 
the stages of **instruction selection** and **register allocation**.
It translates the frame index to the correct offset of the stack pointer using:

.. code:: cpp

  spOffset = MF.getFrameInfo()->getObjectOffset(FrameIndex);

For instance, in `ch3.cpp`, the offset calculation is displayed as follows:

.. code:: text
  
  Spilling live registers at end of block.
  BB#0: derived from LLVM BB %0
  	%V0<def> = ADDiu %ZERO, 0
  	ST %V0, <fi#0>, 0; mem:ST4[%1]
  	RetLR %V0<imp-use,kill>
  alloc FI(0) at SP[-4]
  
  Function : main
  <--------->
  ST %V0, <fi#0>, 0; mem:ST4[%1]
  FrameIndex : 0
  spOffset   : -4
  stackSize  : 8
  Offset     : 4
  <--------->
    ...
    .file "ch3.bc"
    ...
    .frame  $sp,8,$lr
    ...
  # BB#0:
    addiu $sp, $sp, -8
  $tmp1:
    .cfi_def_cfa_offset 8
    addiu $2, $zero, 0
    st  $2, 4($sp)
    ...


.. rubric:: lbdex/chapters/Chapter3_5/Cpu0SEFrameLowering.cpp
.. literalinclude:: ../lbdex/chapters/Chapter3_5/Cpu0SEFrameLowering.cpp
    :start-after: //@determineCalleeSaves {
    :end-before: //}

The `determineCalleeSaves()` function in `Cpu0SEFrameLowering.cpp` determines 
the spill registers. Once the spill registers are identified, the function 
`SpillCalleeSavedRegisters()` will save/restore registers to/from stack slots 
via the following code:

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0InstrInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.h
    :start-after: #if CH >= CH3_5 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.h
    :start-after: #if CH >= CH3_5 //4
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0InstrInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.cpp
    :start-after: #if CH >= CH3_5 //1
    :end-before: #endif
    
.. rubric:: lbdex/chapters/Chapter3_5/Cpu0SEInstrInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.h
    :start-after: #if CH >= CH3_5 //1
    :end-before: #endif //#if CH >= CH3_5 //1

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0SEInstrInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEInstrInfo.cpp
    :start-after: #if CH >= CH3_5 //1
    :end-before: #endif //#if CH >= CH3_5 //1

Functions `storeRegToStack()` in `Cpu0SEInstrInfo.cpp` and 
`storeRegToStackSlot()` in `Cpu0InstrInfo.cpp` handle register spilling during 
the register allocation process. 

Each local variable is associated with a frame index. The code 
`.addFrameIndex(FI).addImm(Offset).addMemOperand(MMO);` in `storeRegToStack()` 
(where `Offset` is `0`) is added for each virtual register.

The functions `loadRegFromStackSlot()` and `loadRegFromStack()` are used when 
registers need to be reloaded from stack slots.

If `V0` is added to `Cpu0CallingConv.td` as shown below and both 
`storeRegToStack()` and `storeRegToStackSlot()` are absent in 
`Cpu0SEInstrInfo.cpp`, `Cpu0SEInstrInfo.h`, and `Cpu0InstrInfo.h`, the following 
error will occur.

.. rubric:: lbdex/Cpu0/Cpu0CallingConv.td
.. code-block:: c++

    def CSR_O32 : CalleeSavedRegs<(add LR, FP, V0,
                                     (sequence "S%u", 1, 0))>;

.. code-block:: console
  
  114-43-191-19:input Jonathan$ ~/llvm/test/build/bin/llc 
  -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc -o -
    .text
    .section .mdebug.abiO32
    .previous
    .file "ch3.bc"
  Target didn't implement TargetInstrInfo::storeRegToStackSlot!
  ...
  Stack dump:
  ...
  Abort trap: 6

.. table:: Backend functions called in PrologEpilogInserter.cpp

  ===================================================  ===========================================
  Stage                                                Function   
  ===================================================  ===========================================
  Prologue/Epilogue Insertion & Frame Finalization
    - Determine spill callee saved registers           - Cpu0SEFrameLowering::determineCalleeSaves
    - Spill callee saved registers                     - Cpu0SEFrameLowering::spillCalleeSavedRegisters
    - Prolog                                           - Cpu0SEFrameLowering::emitPrologue
    - Epilog                                           - Cpu0SEFrameLowering::emitEpilogue
    - Handle stack slot for local variables            - Cpu0RegisterInfo::eliminateFrameIndex
  ===================================================  ===========================================

File PrologEpilogInserter.cpp will call backend functions 
spillCalleeSavedRegisters(), emitProlog(), emitEpilog() and eliminateFrameIndex()
as follows,

.. rubric:: lib/CodeGen/PrologEpilogInserter.cpp
.. code-block:: c++
  
  class PEI : public MachineFunctionPass {
  public:
    static char ID;
    explicit PEI(const TargetMachine *TM = nullptr) : MachineFunctionPass(ID) {
      initializePEIPass(*PassRegistry::getPassRegistry());
  
      if (TM && (!TM->usesPhysRegsForPEI())) {
        ...
      } else {
        SpillCalleeSavedRegisters = doSpillCalleeSavedRegs;
        ...
      }    
    }
    ...
  }
    
  /// insertCSRSpillsAndRestores - Insert spill and restore code for
  /// callee saved registers used in the function.
  ///
  static void insertCSRSpillsAndRestores(MachineFunction &Fn,
                                         const MBBVector &SaveBlocks,
                                         const MBBVector &RestoreBlocks) {
    ...
    // Spill using target interface.
    for (MachineBasicBlock *SaveBlock : SaveBlocks) {
      ...
      if (!TFI->spillCalleeSavedRegisters(*SaveBlock, I, CSI, TRI)) {
        for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
          // Insert the spill to the stack frame.
          ...
          TII.storeRegToStackSlot(*SaveBlock, I, Reg, true, CSI[i].getFrameIdx(),
                                  RC, TRI);
        }
      }
      ...
    }
  
    // Restore using target interface.
    for (MachineBasicBlock *MBB : RestoreBlocks) {
      ...
      // Restore all registers immediately before the return and any
      // terminators that precede it.
      if (!TFI->restoreCalleeSavedRegisters(*MBB, I, CSI, TRI)) {
        for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
          ...
          TII.loadRegFromStackSlot(*MBB, I, Reg, CSI[i].getFrameIdx(), RC, TRI);
          ...
        }
      ...
    }
    ...
  }
  
  static void doSpillCalleeSavedRegs(MachineFunction &Fn, RegScavenger *RS, 
                                     unsigned &MinCSFrameIndex,
                                     unsigned &MaxCSFrameIndex,
                                     const MBBVector &SaveBlocks,
                                     const MBBVector &RestoreBlocks) {
    const Function *F = Fn.getFunction();
    const TargetFrameLowering *TFI = Fn.getSubtarget().getFrameLowering();
    MinCSFrameIndex = std::numeric_limits<unsigned>::max();
    MaxCSFrameIndex = 0; 
  
    // Determine which of the registers in the callee save list should be saved.
    BitVector SavedRegs;
    TFI->determineCalleeSaves(Fn, SavedRegs, RS); 
  
    // Assign stack slots for any callee-saved registers that must be spilled.
    assignCalleeSavedSpillSlots(Fn, SavedRegs, MinCSFrameIndex, MaxCSFrameIndex);
  
    // Add the code to save and restore the callee saved registers.
    if (!F->hasFnAttribute(Attribute::Naked))
      insertCSRSpillsAndRestores(Fn, SaveBlocks, RestoreBlocks);
  }

  void PEI::insertPrologEpilogCode(MachineFunction &Fn) {
    const TargetFrameLowering &TFI = *Fn.getSubtarget().getFrameLowering();
  
    // Add prologue to the function...
    for (MachineBasicBlock *SaveBlock : SaveBlocks)
      TFI.emitPrologue(Fn, *SaveBlock);
  
    // Add epilogue to restore the callee-save registers in each exiting block.
    for (MachineBasicBlock *RestoreBlock : RestoreBlocks)
      TFI.emitEpilogue(Fn, *RestoreBlock);
    ...
  }
  
  void PEI::replaceFrameIndices(MachineBasicBlock *BB, MachineFunction &Fn, 
                                int &SPAdj) {
    ...
      for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
        ...
        // If this instruction has a FrameIndex operand, we need to
        // use that target machine register info object to eliminate
        // it.
        TRI.eliminateFrameIndex(MI, SPAdj, i,
                                FrameIndexVirtualScavenging ?  nullptr : RS);
        ...
      }
    ...
  }

  /// replaceFrameIndices - Replace all MO_FrameIndex operands with physical
  /// register references and actual offsets.
  ///
  void PEI::replaceFrameIndices(MachineFunction &Fn) {
    ...
    // Iterate over the reachable blocks in DFS order.
    for (auto DFI = df_ext_begin(&Fn, Reachable), DFE = df_ext_end(&Fn, Reachable);
         DFI != DFE; ++DFI) {
      ...
      replaceFrameIndices(BB, Fn, SPAdj);
      ...
    }
  
    // Handle the unreachable blocks.
    for (auto &BB : Fn) {
      ...
      replaceFrameIndices(&BB, Fn, SPAdj);
    }
  }
  
  bool PEI::runOnMachineFunction(MachineFunction &Fn) {
    const TargetFrameLowering &TFI = *Fn.getSubtarget().getFrameLowering();
    ...
    FrameIndexVirtualScavenging = TRI->requiresFrameIndexScavenging(Fn);
    ...
    // Handle CSR spilling and restoring, for targets that need it.
    SpillCalleeSavedRegisters(Fn, RS, MinCSFrameIndex, MaxCSFrameIndex,
                              SaveBlocks, RestoreBlocks);
    ...
    // Calculate actual frame offsets for all abstract stack objects...
    calculateFrameObjectOffsets(Fn);
  
    // Add prolog and epilog code to the function.  This function is required
    // to align the stack frame as necessary for any stack variables or
    // called functions.  Because of this, calculateCalleeSavedRegisters()
    // must be called before this function in order to set the AdjustsStack
    // and MaxCallFrameSize variables.
    if (!F->hasFnAttribute(Attribute::Naked))
      insertPrologEpilogCode(Fn);
  
    // Replace all MO_FrameIndex operands with physical register references
    // and actual offsets.
    //
    replaceFrameIndices(Fn);
  
    // If register scavenging is needed, as we've enabled doing it as a 
    // post-pass, scavenge the virtual registers that frame index elimination
    // inserted.
    if (TRI->requiresRegisterScavenging(Fn) && FrameIndexVirtualScavenging) {
      ScavengeFrameVirtualRegs(Fn, RS);
  
      // Clear any vregs created by virtual scavenging.
      Fn.getRegInfo().clearVirtRegs();
    }
    ...
  }


Large stack
+++++++++++

At this stage, we have successfully translated a simple ``main()`` 
function containing only ``return 0;``. However, the stack size 
adjustments for 32-bit values are handled by ``Cpu0AnalyzeImmediate.cpp`` 
and the instruction definitions in ``Cpu0InstrInfo.td`` from Chapter3_5.

Later, in **CH9_3**, dynamic stack allocation support introduces the 
instruction:

.. code:: nasm

   move $fp, $sp

This addition makes the implementation more complex, potentially 
diverging from the tutorial’s primary focus on the **Cpu0** architecture.

Thus, for simplicity, we avoid handling large stack frames at this stage.
[#nolargeframe]_

.. rubric:: lbdex/chapters/Chapter3_5/CMakeLists.txt
.. code-block:: c++

  add_llvm_target(
    ...
  
.. literalinclude:: ../lbdex/Cpu0/CMakeLists.txt
    :start-after: #if CH >= CH3_5
    :end-before: #endif
  
.. code-block:: c++

    ...
  )

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0AnalyzeImmediate.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0AnalyzeImmediate.h

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0AnalyzeImmediate.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0AnalyzeImmediate.cpp

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0InstrInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.h
    :start-after: #if CH >= CH3_5 //1
    :end-before: #endif


.. rubric:: lbdex/chapters/Chapter3_5/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 4
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 5
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 6
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 7
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 8
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 9
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 10
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 11
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 13
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH3_5 14
    :end-before: #endif
    
The ``Cpu0AnalyzeImmediate.cpp`` implementation is recursive and has a 
somewhat complex logic. However, recursive techniques are commonly 
covered in compiler frontend books, so you should already be familiar 
with them.

Instead of tracing the code directly, we list the stack size and the 
corresponding instructions generated in:

**Table: Cpu0 stack adjustment instructions before replacing ``addiu`` 
and ``shl`` with ``lui`` instructions** 

as follows, and in:

**Table: Cpu0 stack adjustment instructions after replacing ``addiu`` 
and ``shl`` with ``lui`` instructions** 

in the next section.

.. table:: Cpu0 stack adjustment instructions before replace addiu and shl with lui instruction

  ====================  ================  ==================================  ==================================
  stack size range      ex. stack size    Cpu0 Prologue instructions          Cpu0 Epilogue instructions
  ====================  ================  ==================================  ==================================
  0 ~ 0x7ff8            - 0x7ff8          - addiu $sp, $sp, -32760;           - addiu $sp, $sp, 32760;
  0x8000 ~ 0xfff8       - 0x8000          - addiu $sp, $sp, -32768;           - addiu $1, $zero, 1;
                                                                              - shl $1, $1, 16;
                                                                              - addiu $1, $1, -32768;
                                                                              - addu $sp, $sp, $1;
  x10000 ~ 0xfffffff8   - 0x7ffffff8      - addiu $1, $zero, 8;               - addiu $1, $zero, 8;
                                          - shl $1, $1, 28;                   - shl $1, $1, 28;
                                          - addiu $1, $1, 8;                  - addiu $1, $1, -8;
                                          - addu $sp, $sp, $1;                - addu $sp, $sp, $1;
  x10000 ~ 0xfffffff8   - 0x90008000      - addiu $1, $zero, -9;              - addiu $1, $zero, -28671;
                                          - shl $1, $1, 28;                   - shl $1, $1, 16
                                          - addiu $1, $1, -32768;             - addiu $1, $1, -32768;
                                          - addu $sp, $sp, $1;                - addu $sp, $sp, $1;
  ====================  ================  ==================================  ==================================


Since the Cpu0 stack is 8-byte aligned, addresses from 0x7FF9 to 0x7FFF  
cannot exist.  

Assume ``sp = 0xA0008000`` and ``stack size = 0x90008000``, then  
``(0xA0008000 - 0x90008000) => 0x10000000``.  
Verify with the Cpu0 prologue instructions as follows,  

1. "addiu	$1, $zero, -9" => ($1 = 0 + 0xfffffff7) => $1 = 0xfffffff7.
2. "shl	$1, $1, 28;" => $1 = 0x70000000.
3. "addiu	$1, $1, -32768" => $1 = (0x70000000 + 0xffff8000) => $1 = 0x6fff8000.
4. "addu	$sp, $sp, $1" => $sp = (0xa0008000 + 0x6fff8000) => $sp = 0x10000000.

Verify with the Cpu0 epilogue instructions with ``sp = 0x10000000`` and  
``stack size = 0x90008000`` as follows,  

1. "addiu	$1, $zero, -28671" => ($1 = 0 + 0xffff9001) => $1 = 0xffff9001.
2. "shl	$1, $1, 16;" => $1 = 0x90010000.
3. "addiu	$1, $1, -32768" => $1 = (0x90010000 + 0xffff8000) => $1 = 0x90008000.
4. "addu	$sp, $sp, $1" => $sp = (0x10000000 + 0x90008000) => $sp = 0xa0008000.


The ``Cpu0AnalyzeImmediate::GetShortestSeq()`` will call  
``Cpu0AnalyzeImmediate::ReplaceADDiuSHLWithLUi()`` to replace ``addiu`` and  
``shl`` with a single instruction ``lui`` only.  
The effect is shown in the following table.  

.. table:: Cpu0 stack adjustment instructions after replace addiu and shl with lui instruction

  ====================  ================  ==================================  ==================================
  stack size range      ex. stack size    Cpu0 Prologue instructions          Cpu0 Epilogue instructions
  ====================  ================  ==================================  ==================================
  0x8000 ~ 0xfff8       - 0x8000          - addiu $sp, $sp, -32768;           - ori	$1, $zero, 32768;
                                                                              - addu $sp, $sp, $1;
  x10000 ~ 0xfffffff8   - 0x7ffffff8      - lui	$1, 32768;                    - lui	$1, 32767;
                                          - addiu $1, $1, 8;                  - ori	$1, $1, 65528
                                          - addu $sp, $sp, $1;                - addu $sp, $sp, $1;
  x10000 ~ 0xfffffff8   - 0x90008000      - lui $1, 28671;                    - lui $1, 36865;
                                          - ori	$1, $1, 32768;                - addiu $1, $1, -32768;
                                          - addu $sp, $sp, $1;                - addu $sp, $sp, $1;
  ====================  ================  ==================================  ==================================


Assume ``sp = 0xA0008000`` and ``stack size = 0x90008000``, then  
``(0xA0008000 - 0x90008000) => 0x10000000``.  
Verify with the Cpu0 prologue instructions as follows,  

1. "lui	$1, 28671" => $1 = 0x6fff0000.
2. "ori	$1, $1, 32768" => $1 = (0x6fff0000 + 0x00008000) => $1 = 0x6fff8000.
3. "addu	$sp, $sp, $1" => $sp = (0xa0008000 + 0x6fff8000) => $sp = 0x10000000.

Verify with the Cpu0 epilogue instructions with ``sp = 0x10000000`` and  
``stack size = 0x90008000`` as follows,  

1. "lui	$1, 36865" => $1 = 0x90010000.
2. "addiu $1, $1, -32768" => $1 = (0x90010000 + 0xffff8000) => $1 = 0x90008000.
3. "addu $sp, $sp, $1" => $sp = (0x10000000 + 0x90008000) => $sp = 0xa0008000.

The file ``ch3_largeframe.cpp`` includes the large frame test.  

Running Chapter3_5 with ``ch3_largeframe.cpp`` will produce the following result.  

.. rubric:: lbdex/input/ch3_largeframe.cpp
.. literalinclude:: ../lbdex/input/ch3_largeframe.cpp
    :start-after: /// start

.. code-block:: console

  118-165-78-12:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch3_largeframe.cpp -emit-llvm -o ch3_largeframe.bc
  118-165-78-12:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm 
  ch3_largeframe.bc.bc -o -
    ...
    .section .mdebug.abiO32
    .previous
    .file "ch3_largeframe.bc"
    .globl  _Z16test_largegframev
    .align  2
    .type _Z16test_largegframev,@function
    .ent  _Z16test_largegframev   # @_Z16test_largegframev
  _Z16test_largegframev:
    .frame  $fp,1879015424,$lr
    .mask   0x00000000,0
    .set  noreorder
    .set  nomacro
    .set  noat
  # BB#0:
    lui $1, 36865
    addiu $1, $1, -32768
    addu  $sp, $sp, $1
    addiu $2, $zero, 0
    lui $1, 28672
    addiu $1, $1, -32768
    addu  $sp, $sp, $1
    ret $lr
    .set  at
    .set  macro
    .set  reorder
    .end  _Z16test_largegframev
  $func_end0:
    .size _Z16test_largegframev, ($func_end0)-_Z16test_largegframev


Data operands DAGs
---------------------

From the above or the compiler book, you can see that all the OP codes are  
the internal nodes in DAG graphs, and operands are the leaves of DAGs.  

To develop your backend, you can copy the related data operand DAG nodes from  
other backends since the IR data nodes are handled by all backends.  

Regarding data DAG nodes, you can understand some of them through  
``Cpu0InstrInfo.td`` and find them using the following command:  

grep -R "<datadag>"  \`find llvm/include/llvm\`,

By spending a little more time thinking or making educated guesses, you  
can identify them. Some data DAGs are well understood, some are partially  
understood, and some remain unknown—but that is acceptable.  

Here is a list of some data DAGs we understand and have encountered so far:  

.. rubric:: include/llvm/Target/TargetSelectionDAG.td
.. code-block:: c++
  
  // PatLeaf's are pattern fragments that have no operands.  This is just a helper
  // to define immediates and other common things concisely.
  class PatLeaf<dag frag, code pred = [{}], SDNodeXForm xform = NOOP_SDNodeXForm>
   : PatFrag<(ops), frag, pred, xform>;

  // ImmLeaf is a pattern fragment with a constraint on the immediate.  The
  // constraint is a function that is run on the immediate (always with the value
  // sign extended out to an int64_t) as Imm.  For example:
  //
  //  def immSExt8 : ImmLeaf<i16, [{ return (char)Imm == Imm; }]>;
  //
  // this is a more convenient form to match 'imm' nodes in than PatLeaf and also
  // is preferred over using PatLeaf because it allows the code generator to
  // reason more about the constraint.
  //
  // If FastIsel should ignore all instructions that have an operand of this type,
  // the FastIselShouldIgnore flag can be set.  This is an optimization to reduce
  // the code size of the generated fast instruction selector.
  class ImmLeaf<ValueType vt, code pred, SDNodeXForm xform = NOOP_SDNodeXForm>
    : PatFrag<(ops), (vt imm), [{}], xform> {
    let ImmediateCode = pred;
    bit FastIselShouldIgnore = 0;
  }

.. rubric:: lbdex/chapters/Chapter3_5/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 4
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH3_5 2
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 5
    :end-before: //#if CH >= CH11_1 2

.. code-block:: c++

  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH3_5 3
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 6
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH3_5 4
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 7
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 8
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH2 9
    :end-before: //#endif


As mentioned in the subsection "Instruction Selection" of the last chapter,  
``immSExt16`` is a data leaf DAG node that returns ``true`` if its value is  
within the range of a signed 16-bit integer. The ``load_a``, ``store_a``,  
and others are similar, but they check for alignment.  

The ``mem`` is explained in Chapter3_2 for printing operands, and ``addr``  
is explained in Chapter3_3 for data DAG selection.  

The ``simm16``, ..., are inherited from ``Operand<i32>`` because Cpu0 is  
a 32-bit architecture. It may exceed 16 bits, so the ``immSExt16`` pattern  
leaf is used to constrain it, as seen in the ``ADDiu`` example mentioned  
in the last chapter.  

The ``PatLeaf immZExt16``, ``immLow16Zero``, and ``ImmLeaf immZExt5`` are  
similar to ``immSExt16``.  
Summary of this Chapter
-----------------------

Summary the functions for llvm backend stages as the following table.

.. code-block:: console

  118-165-79-200:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch3.bc 
  -debug-pass=Structure -o -
  ...
  Machine Branch Probability Analysis
    ModulePass Manager
      FunctionPass Manager
        ...
        CPU0 DAG->DAG Pattern Instruction Selection
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
        Greedy Register Allocator
        ...
        Prologue/Epilogue Insertion & Frame Finalization
        ...
        Post-RA pseudo instruction expansion pass
        ...
        Cpu0 Assembly Printer


.. table:: Functions for llvm backend stages

  ===================================================  ===========================================
  Stage                                                Function   
  ===================================================  ===========================================
  Before CPU0 DAG->DAG Pattern Instruction Selection   - Cpu0TargetLowering::LowerFormalArguments
                                                       - Cpu0TargetLowering::LowerReturn
  Instruction selection                                - Cpu0DAGToDAGISel::Select
  Prologue/Epilogue Insertion & Frame Finalization
    - Determine spill callee saved registers           - Cpu0SEFrameLowering::determineCalleeSaves
    - Spill callee saved registers                     - Cpu0SEFrameLowering::spillCalleeSavedRegisters
    - Prolog                                           - Cpu0SEFrameLowering::emitPrologue
    - Epilog                                           - Cpu0SEFrameLowering::emitEpilogue
    - Handle stack slot for local variables            - Cpu0RegisterInfo::eliminateFrameIndex
  Post-RA pseudo instruction expansion pass            - Cpu0SEInstrInfo::expandPostRAPseudo
  Cpu0 Assembly Printer                                - Cpu0AsmPrinter.cpp, Cpu0MCInstLower.cpp
                                                       - Cpu0InstPrinter.cpp
  ===================================================  ===========================================

We add a pass in Instruction Section stage in section "Add Cpu0DAGToDAGISel 
class". You can embed your code into other passes like that. Please check 
CodeGen/Passes.h for the information. Remember the pass is called according 
the function unit as the ``llc -debug-pass=Structure`` indicated.

We have finished a simple compiler for cpu0 which only supports **ld**, 
**st**, **addiu**, **ori**, **lui**, **addu**, **shl** and **ret** 8 
instructions.

We add a pass in the Instruction Selection stage in the section  
"Add Cpu0DAGToDAGISel class." You can embed your code into other passes  
similarly. Please check ``CodeGen/Passes.h`` for more information.  
Remember that passes are called according to the function unit, as indicated  
by the command ``llc -debug-pass=Structure``.  

We have completed a simple compiler for Cpu0, which only supports  
eight instructions: **ld**, **st**, **addiu**, **ori**, **lui**, **addu**,  
**shl**, and **ret**.  

We are satisfied with this result.  
However, you might think,  
"After writing so much code, we only implemented these eight instructions!"  
The key takeaway is that we have built a framework for the Cpu0 target machine.  
(Refer to the LLVM backend structure class inheritance tree earlier in this  
chapter.)  

So far, we have written over 3,000 lines of source code, including comments,  
across multiple files: ``*.cpp``, ``*.h``, ``*.td``, and ``CMakeLists.txt``.  
You can count them using the command:

``wc `find dir -name *.cpp```

for ``*.cpp``, ``*.h``, ``*.td``, and ``*.txt`` files.  
In contrast, the LLVM front-end tutorial contains only about 700 lines of  
source code (excluding comments).  

Don't be discouraged by these results.  
In reality, writing a backend starts slowly but speeds up over time.  

For comparison:  

- Clang has over **500,000** lines of source code (including comments) in  
  the ``clang/lib`` directory, supporting both C++ and Objective-C.  
- The MIPS backend in LLVM 3.1 contains **15,000** lines (with comments).  
- Even the complex x86 CPU backend, which is CISC externally but RISC internally  
  (using micro-instructions), has only **45,000** lines in LLVM 3.1 (with comments).  

In the next chapter, we will demonstrate how adding support for a new  
instruction is as easy as 1-2-3!

.. [#targetmachine] http://llvm.org/docs/WritingAnLLVMBackend.html#target-machine

.. [#datalayout] http://llvm.org/docs/LangRef.html#data-layout

.. [#registration] http://jonathan2251.github.io/lbd/llvmstructure.html#target-registration

.. [#nolargeframe] http://jonathan2251.github.io/lbd/funccall.html#dynamic-stack-allocation-support
