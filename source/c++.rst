.. _sec-c++:

C++ support
=============

.. contents::
   :local:
   :depth: 4

This chapter supports C++ compiler features. 


Exception handle
-------------------

The Chapter11_2 can be built and run with the C++ polymorphism example code of 
ch12_inherit.cpp as follows,

.. rubric:: lbdex/input/ch12_inherit.cpp
.. literalinclude:: ../lbdex/input/ch12_inherit.cpp
    :start-after: /// start

If using cout instead of printf in ch12_inherit.cpp on Linux won't generate exception 
handler IRs.
But on iMac, ch12_inherit.cpp will generate invoke, landing, resume and unreachable 
exception handler IRs.
Example code, ch12_eh.cpp, which supports **try** and **catch** exception handler 
as the following will generate these exception handler IRs both on iMac and Linux.

.. rubric:: lbdex/input/ch12_eh.cpp
.. literalinclude:: ../lbdex/input/ch12_eh.cpp
    :start-after: /// start

.. code-block:: console

  JonathantekiiMac:input Jonathan$ clang -c ch12_eh.cpp -emit-llvm 
  -o ch12_eh.bc
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llvm-dis ch12_eh.bc -o -
  
.. literalinclude:: ../lbdex/output/ch12_eh.ll

.. code:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch12_eh.bc -o -
	  .section .mdebug.abi32
	  .previous
	  .file	"ch12_eh.bc"
  llc: /Users/Jonathan/llvm/test/src/lib/CodeGen/LiveVariables.cpp:133: void llvm::
  LiveVariables::HandleVirtRegUse(unsigned int, llvm::MachineBasicBlock *, llvm
  ::MachineInstr *): Assertion `MRI->getVRegDef(reg) && "Register use before 
  def!"' failed.


About the IRs of LLVM exception handling, please reference here [#exception]_.
Chapter12_1 supports the llvm IRs of corresponding **try** and **catch** 
exception C++ keywords. It can compile ch12_eh.bc as follows,

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //5
    :end-before: #endif

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch12_eh.bc -o -
  
.. literalinclude:: ../lbdex/output/ch12_eh.cpu0.s


Thread variable
-------------------

C++ include the thread variable as the following file ch12_thread_var.cpp.

.. rubric:: lbdex/input/ch12_thread_var.cpp
.. literalinclude:: ../lbdex/input/ch12_thread_var.cpp
    :start-after: /// start

While global variable is a single instance shared by all threads in a process, 
thread variable has different instances for each different thread in a process. 
The same thread share the thread variable but different threads have their own 
thread variable with the same name [#thread-wiki]_.

To support thread variable, tlsgd, tlsldm, dtp_hi, dtp_lo, gottp, tp_hi and
tp_lo in evaluateRelocExpr() of Cpu0AsmParser.cpp, in printImpl() of
Cpu0MCExpr.cpp and the following code are required.
Most of them are for relocation record handle and display since the thread 
variable created by OS or language library which support multi-threads 
programming.

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0AsmBackend.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0AsmBackend.cpp
    :start-after: //@getFixupKindInfo {
    :end-before: { "fixup_Cpu0_32",             0,     32,   0 },
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0AsmBackend.cpp
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

      ...
    };
    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0BaseInfo.h
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0BaseInfo.h
    :start-after: //@Cpu0II
    :end-before: MO_NO_FLAG,
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0BaseInfo.h
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

      ...
    };
    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0ELFObjectWriter.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0ELFObjectWriter.cpp
    :start-after: //@GetRelocType {
    :end-before: default:
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0ELFObjectWriter.cpp
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0FixupKinds.h
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0FixupKinds.h
    :start-after: //@Fixups {
    :end-before: //@ Pure upper 32 bit fixup resulting in - R_CPU0_32.
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0FixupKinds.h
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

      ...
    };

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: //@getExprOpValue {
    :end-before: //@getExprOpValue body {
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 3.1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 4
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0SelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif
	
.. code-block:: c++

    ...
  }
  
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #if CH >= CH8_1 //6
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //3
    :end-before: #endif
	
.. code-block:: c++

      ...
    }
    ...
  }
  
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //4
    :end-before: #endif


.. rubric:: lbdex/chapters/Chapter12_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0MCInstLower.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: //@LowerSymbolOperand {
    :end-before: default:
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH12_1
    :end-before: #endif
    
.. code-block:: c++

    ...
    }
    ...
  }


.. code-block:: console

  JonathantekiiMac:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch12_thread_var.cpp -emit-llvm -std=c++11 -o ch12_thread_var.bc
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llvm-dis ch12_thread_var.bc -o -
  
.. literalinclude:: ../lbdex/output/ch12_thread_var.ll

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch12_thread_var.bc 
  -o -
  
.. literalinclude:: ../lbdex/output/ch12_thread_var.cpu0.pic.s


In pic mode, the __thread variable access by call function __tls_get_addr with 
the address of thread variable. 
The c++11 standard thread_local variable is accessed by call function _ZTW1b 
which call the function __tls_get_addr too to get the thread_local variable 
address. 
In static mode, the thread variable is accessed by machine instructions as follows,

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=static -filetype=asm 
  ch12_thread_var.bc -o -
  
.. literalinclude:: ../lbdex/output/ch12_thread_var.cpu0.static.s


While Mips uses rdhwr instruction to access thread varaible as below, 
Cpu0 access thread varaible without inventing any new instruction. 
The thread variables are keeped in thread varaible memory location which 
accessed through \%tp_hi and \%tp_lo. Furthermore, this section of memory is 
protected through kernel mode program. 
As a result, the user mode program cannot access this area of memory and 
no space to breathe for hack program.

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=mips -relocation-model=static -filetype=asm 
  ch12_thread_var.bc -o -
    ...
    lui $1, %tprel_hi(a)
    ori $1, $1, %tprel_lo(a)
    .set  push
    .set  mips32r2
    rdhwr $3, $29
    .set  pop
    addu  $1, $3, $1
    addiu $2, $zero, 2
    sw  $2, 0($1)
    addiu $2, $zero, 2
    ...

In static mode, the thread variable is similar to global variable. 
In general, they are same in IRs, DAGs and machine code translation. 
List them in the following tables. 
You can check them with debug option enabled.

.. table:: The DAGs of thread varaible of static mode

  ==========================  ===========================
  stage                       DAG
  ==========================  ===========================
  IR                          load i32* @a, align 4;
  Legalized selection DAG     (add Cpu0ISD::Hi Cpu0ISD::Lo);
  Instruction Selection       ori $2, $zero, %tp_lo(a);
  -                           lui $3, %tp_hi(a);
  -                           addu  $3, $3, $2;
  ==========================  ===========================

.. table:: The DAGs of local_thread varaible of static mode

  ==========================  ===========================
  stage                       DAG
  ==========================  ===========================
  IR                          ret i32* @b;
  Legalized selection DAG     %0=(add Cpu0ISD::Hi Cpu0ISD::Lo);...
  Instruction Selection       ori $2, $zero, %tp_lo(a);
  -                           lui $3, %tp_hi(a);
  -                           addu  $3, $3, $2;
  ==========================  ===========================


Atomic
--------

In tradition, C uses different API which provided by OS or library to support
multi-thread programming. For example, posix thread API on unix/linux, MS
windows API, ..., etc. In order to achieve synchronization to solve race
condition between threads, OS provide their own lock or semaphore functions to 
programmer. But this solution is OS dependent. 
After c++11, programmer can use atomic to program and run the code 
on every different platform since the thread and atomic are part of c++ standard.
Beside of portability, the other important benifit is the compiler can generate
high performance code by the target hardware instruction rather than couting on
lock() function only [#atomic-wiki]_ [#atomic-stackoverflow]_ 
[#atomic-herbsutter]_.

In order to support atomic in C++ and java, llvm provides the atomic IRs here 
[#atomics-llvm]_ [#llvmlang-ordering]_.

To support llvm atomic IRs, the following code added to Chapter12_1.

.. rubric:: lbdex/chapters/Chapter12_1/Disassembler/Cpu0Disassembler.cpp
.. literalinclude:: ../lbdex/Cpu0/Disassembler/Cpu0Disassembler.cpp
    :start-after: //@DecodeMem {
    :end-before: //@DecodeMem body {
.. literalinclude:: ../lbdex/Cpu0/Disassembler/Cpu0Disassembler.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 5
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 6
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 7
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 8
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 9
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 10
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 11
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 12
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 13
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //4
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0SelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@3_1 1 {
    :end-before: switch (Opcode) {
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //0.5
    :end-before: #endif //#if CH >= CH12_1 //0.5

.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //1.5
    :end-before: #endif

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #if CH >= CH8_1 //6
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //7
    :end-before: #endif //#if CH >= CH12_1 //7

.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //8
    :end-before: #endif

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //9
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0RegisterInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.h
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0RegisterInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0SEISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelLowering.cpp
    :start-after: //@Cpu0SETargetLowering {
    :end-before: //@Cpu0SETargetLowering body {
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelLowering.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. code-block:: c++

    ...
  }


.. rubric:: lbdex/chapters/Chapter12_1/Cpu0TargetMachine.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: //@Cpu0PassConfig {
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. code-block:: c++

    ...
  };

.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH12_1 //2
    :end-before: #endif

Since SC instruction uses RegisterOperand type in Cpu0InstrInfo.td and SC uses
FMem node which DecoderMethod is "DecodeMem", the DecodeMem() of 
Cpu0Disassembler.cpp need to be changed as above.

The atomic node defined in "let usesCustomInserter = 1 in" of Cpu0InstrInfo.td
tells llvm calling EmitInstrWithCustomInserter() of Cpu0ISelLowering.cpp. For
example, "def ATOMIC_LOAD_ADD_I8 : Atomic2Ops<atomic_load_add_8, CPURegs>;" will
calling EmitInstrWithCustomInserter() with Machine Instruction Opcode 
"ATOMIC_LOAD_ADD_I8" when it meets IR "load atomic i8*".

The "setInsertFencesForAtomic(true);" in Cpu0ISelLowering.cpp will trigger 
addIRPasses() of Cpu0TargetMachine.cpp, then createAtomicExpandPass() in 
addIRPasses() will create llvm IR ATOMIC_FENCE. Next, the lowerATOMIC_FENCE()
of Cpu0ISelLowering.cpp will create Cpu0ISD::Sync when it meets IR ATOMIC_FENCE
since "setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);" of 
Cpu0SEISelLowering.cpp. Finally the pattern defined in Cpu0InstrInfo.td translate
it into instruction "sync" by "def SYNC" and alias "SYNC 0".

This part of Cpu0 backend code is same with Mips except Cpu0 has no instruction 
"nor".

List the atomic IRs, corresponding DAGs and Opcode as the following table.

.. table:: The atomic related IRs, their corresponding DAGs and Opcode of Cpu0ISelLowering.cpp

  ==========================  ===========================  ===========================
  IR                          DAG                          Opcode
  ==========================  ===========================  ===========================
  load atomic                 AtomicLoad                   ATOMIC_CMP_SWAP_XXX
  store atomic                AtomicStore                  ATOMIC_SWAP_XXX
  atomicrmw add               AtomicLoadAdd                ATOMIC_LOAD_ADD_XXX
  atomicrmw sub               AtomicLoadSub                ATOMIC_LOAD_SUB_XXX
  atomicrmw xor               AtomicLoadXor                ATOMIC_LOAD_XOR_XXX
  atomicrmw and               AtomicLoadAnd                ATOMIC_LOAD_AND_XXX
  atomicrmw nand              AtomicLoadNand               ATOMIC_LOAD_NAND_XXX
  atomicrmw or                AtomicLoadOr                 ATOMIC_LOAD_OR_XXX
  cmpxchg                     AtomicCmpSwapWithSuccess     ATOMIC_CMP_SWAP_XXX
  atomicrmw xchg              AtomicLoadSwap               ATOMIC_SWAP_XXX
  ==========================  ===========================  ===========================

Input files atomics.ll and atomics-fences.ll include the llvm atomic IRs test.
Input files ch12_atomics.cpp and ch12_atomics-fences.cpp are the C++ source 
files for generating llvm atomic IRs. The C++ files need to run with clang 
options "clang++ -pthread -std=c++11".


.. [#exception] http://llvm.org/docs/ExceptionHandling.html

.. [#thread-wiki] http://en.wikipedia.org/wiki/Thread-local_storage

.. [#atomic-wiki] https://en.wikipedia.org/wiki/Memory_model_%28programming%29

.. [#atomic-stackoverflow] http://stackoverflow.com/questions/6319146/c11-introduced-a-standardized-memory-model-what-does-it-mean-and-how-is-it-g

.. [#atomic-herbsutter] http://herbsutter.com/2013/02/11/atomic-weapons-the-c-memory-model-and-modern-hardware/

.. [#atomics-llvm] http://llvm.org/docs/Atomics.html

.. [#llvmlang-ordering] http://llvm.org/docs/LangRef.html#ordering
