.. _sec-globalvars:

Global variables
==================

.. contents::
   :local:
   :depth: 4

In the last three chapters, we only access the local variables. 
This chapter deals global variable access translation. 

The global variable DAG translation is different from the previous DAG 
translations until now we have. 
It creates IR DAG nodes at run time in backend C++ code according the 
``llc -relocation-model`` option while the others of DAG just do IR DAG to 
Machine DAG translation directly according the input file of IR DAGs (except
the Pseudo instruction RetLR used in Chapter3_4).
Readers should focus on how to add code for creating DAG nodes at run time and 
how to define the pattern match in td for the run time created DAG nodes. 
In addition, the machine instruction printing function for global variable 
related assembly directive (macro) should be cared if your backend has it.

Chapter6_1/ supports the global variable, let's compile ch6_1.cpp with this 
version first, then explain the code changes after that.

.. rubric:: lbdex/input/ch6_1.cpp
.. literalinclude:: ../lbdex/input/ch6_1.cpp
    :start-after: /// start

.. code-block:: bash

  118-165-78-166:input Jonathan$ llvm-dis ch6_1.bc -o -
  ...
  @gStart = global i32 2, align 4
  @gI = global i32 100, align 4
  
  define i32 @_Z3funv() nounwind uwtable ssp {
    %1 = alloca i32, align 4
    %c = alloca i32, align 4
    store i32 0, i32* %1
    store i32 0, i32* %c, align 4
    %2 = load i32* @gI, align 4
    store i32 %2, i32* %c, align 4
    %3 = load i32* %c, align 4
    ret i32 %3
  }

Cpu0 global variable options
-----------------------------

Just like Mips, Cpu0 supports both static and pic mode. 
There are two different layout of global variables for static mode which 
controlled by option cpu0-use-small-section. 
Chapter6_1/ supports the global variable translation. 
Let's run Chapter6_1/ with ch6_1.cpp via four different options 
``llc  -relocation-model=static -cpu0-use-small-section=false``, 
``llc  -relocation-model=static -cpu0-use-small-section=true``, 
``llc  -relocation-model=pic -cpu0-use-small-section=false`` and
``llc  -relocation-model=pic -cpu0-use-small-section=true`` to tracing the 
DAGs and Cpu0 instructions.

.. code-block:: bash

  118-165-78-166:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch6_1.cpp -emit-llvm -o ch6_1.bc
  118-165-78-166:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=false 
  -filetype=asm -debug ch6_1.bc -o -
  
  ...
  Type-legalized selection DAG: BB#0 '_Z11test_globalv:'
  SelectionDAG has 12 nodes:
    ...
        0x7ffd5902cc10: <multiple use>
      0x7ffd5902cf10: ch = store 0x7ffd5902cd10, 0x7ffd5902ca10, 0x7ffd5902ce10, 
      0x7ffd5902cc10<ST4[%c]> [ORD=2] [ID=-3]
  
      0x7ffd5902d010: i32 = GlobalAddress<i32* @gI> 0 [ORD=3] [ID=-3]
  
      0x7ffd5902cc10: <multiple use>
    0x7ffd5902d110: i32,ch = load 0x7ffd5902cf10, 0x7ffd5902d010, 
    0x7ffd5902cc10<LD4[@gI]> [ORD=3] [ID=-3]
    ...
  
  Legalized selection DAG: BB#0 '_Z11test_globalv:'
  SelectionDAG has 16 nodes:
    ...
        0x7ffd5902cc10: <multiple use>
      0x7ffd5902cf10: ch = store 0x7ffd5902cd10, 0x7ffd5902ca10, 0x7ffd5902ce10, 
      0x7ffd5902cc10<ST4[%c]> [ORD=2] [ID=8]
  
          0x7ffd5902d310: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=5]
  
        0x7ffd5902d710: i32 = Cpu0ISD::Hi 0x7ffd5902d310
  
          0x7ffd5902d610: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=6]
  
        0x7ffd5902d810: i32 = Cpu0ISD::Lo 0x7ffd5902d610
  
      0x7ffd5902fe10: i32 = add 0x7ffd5902d710, 0x7ffd5902d810
  
      0x7ffd5902cc10: <multiple use>
    0x7ffd5902d110: i32,ch = load 0x7ffd5902cf10, 0x7ffd5902fe10, 
    0x7ffd5902cc10<LD4[@gI]> [ORD=3] [ID=9]
    ...  
    lui $2, %hi(gI)
    ori $2, $2, %lo(gI)
  	ld	$2, 0($2)
  	...
  	.type	gStart,@object          # @gStart
  	.data
  	.globl	gStart
  	.align	2
  gStart:
  	.4byte	2                       # 0x2
  	.size	gStart, 4
  
  	.type	gI,@object              # @gI
  	.globl	gI
  	.align	2
  gI:
  	.4byte	100                     # 0x64
  	.size	gI, 4

.. code-block:: bash

  118-165-78-166:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=true 
  -filetype=asm -debug ch6_1.bc -o -
  
  ...
  Type-legalized selection DAG: BB#0 '_Z11test_globalv:'
  SelectionDAG has 12 nodes:
    ...
        0x7fc5f382cc10: <multiple use>
      0x7fc5f382cf10: ch = store 0x7fc5f382cd10, 0x7fc5f382ca10, 0x7fc5f382ce10, 
      0x7fc5f382cc10<ST4[%c]> [ORD=2] [ID=-3]
  
      0x7fc5f382d010: i32 = GlobalAddress<i32* @gI> 0 [ORD=3] [ID=-3]
  
      0x7fc5f382cc10: <multiple use>
    0x7fc5f382d110: i32,ch = load 0x7fc5f382cf10, 0x7fc5f382d010, 
    0x7fc5f382cc10<LD4[@gI]> [ORD=3] [ID=-3]
    ...
  Legalized selection DAG: BB#0 '_Z11test_globalv:'
  SelectionDAG has 15 nodes:
    ...
        0x7fc5f382cc10: <multiple use>
      0x7fc5f382cf10: ch = store 0x7fc5f382cd10, 0x7fc5f382ca10, 0x7fc5f382ce10, 
      0x7fc5f382cc10<ST4[%c]> [ORD=2] [ID=8]
  
        0x7fc5f382d710: i32 = register %GP
  
          0x7fc5f382d310: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=4]
  
        0x7fc5f382d610: i32 = Cpu0ISD::GPRel 0x7fc5f382d310
  
      0x7fc5f382d810: i32 = add 0x7fc5f382d710, 0x7fc5f382d610
  
      0x7fc5f382cc10: <multiple use>
    0x7fc5f382d110: i32,ch = load 0x7fc5f382cf10, 0x7fc5f382d810, 
    0x7fc5f382cc10<LD4[@gI]> [ORD=3] [ID=9]
    ...
  
  	ori	$2, $gp, %gp_rel(gI)
  	ld	$2, 0($2)
  	...
  	.type	gStart,@object          # @gStart
  	.section	.sdata,"aw",@progbits
  	.globl	gStart
  	.align	2
  gStart:
  	.4byte	2                       # 0x2
  	.size	gStart, 4
  
  	.type	gI,@object              # @gI
  	.globl	gI
  	.align	2
  gI:
  	.4byte	100                     # 0x64
  	.size	gI, 4

.. code-block:: bash

  118-165-78-166:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=false 
  -filetype=asm -debug ch6_1.bc -o -
    
    ...
  Type-legalized selection DAG: BB#0 '_Z11test_globalv:'
  SelectionDAG has 11 nodes:
    ...
        0x7fe03c02e010: <multiple use>
      0x7fe03c02e118: ch = store 0x7fe03b50dee0, 0x7fe03c02de00, 0x7fe03c02df08, 
      0x7fe03c02e010<ST4[%c]> [ORD=3] [ID=-3]
  
      0x7fe03c02e220: i32 = GlobalAddress<i32* @gI> 0 [ORD=4] [ID=-3]
  
      0x7fe03c02e010: <multiple use>
    0x7fe03c02e328: i32,ch = load 0x7fe03c02e118, 0x7fe03c02e220, 
    0x7fe03c02e010<LD4[@gI]> [ORD=4] [ID=-3]
    ...
  Legalized selection DAG: BB#0 '_Z11test_globalv:'
  SelectionDAG has 15 nodes:
    ...
        0x7fe03c02e010: <multiple use>
      0x7fe03c02e118: ch = store 0x7fe03b50dee0, 0x7fe03c02de00, 0x7fe03c02df08, 
      0x7fe03c02e010<ST4[%c]> [ORD=3] [ID=6]
  
          0x7fe03c02e538: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=5] [ORD=4]
  
        0x7fe03c02ea60: i32 = Cpu0ISD::Hi 0x7fe03c02e538 [ORD=4]
  
          0x7fe03c02e958: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=6] [ORD=4]
  
        0x7fe03c02eb68: i32 = Cpu0ISD::Lo 0x7fe03c02e958 [ORD=4]
  
      0x7fe03c02ec70: i32 = add 0x7fe03c02ea60, 0x7fe03c02eb68 [ORD=4]
  
      0x7fe03c02e010: <multiple use>
    0x7fe03c02e328: i32,ch = load 0x7fe03c02e118, 0x7fe03c02ec70, 
    0x7fe03c02e010<LD4[@gI]> [ORD=4] [ID=7]
    ...
	  lui	$2, %got_hi(gI)
	  addu	$2, $2, $gp
	  ld	$2, %got_lo(gI)($2)
    ...
      .type gStart,@object          # @gStart
    .data
    .globl  gStart
    .align  2
  gStart:
    .4byte  3                       # 0x3
    .size gStart, 4
  
    .type gI,@object              # @gI
    .globl  gI
    .align  2
  gI:
    .4byte  100                     # 0x64
    .size gI, 4

.. code-block:: bash

  118-165-78-166:input Jonathan$ /Users/Jonathan/llvm/test/cmake_debug_build/
  Debug/bin/llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=true 
  -filetype=asm -debug ch6_1.bc -o -
  
  ...
  Type-legalized selection DAG: BB#0 '_Z11test_globalv:'
  SelectionDAG has 11 nodes:
    ...
        0x7fad7102cc10: <multiple use>
      0x7fad7102cf10: ch = store 0x7fad7102cd10, 0x7fad7102ca10, 0x7fad7102ce10, 
      0x7fad7102cc10<ST4[%c]> [ORD=2] [ID=-3]
  
      0x7fad7102d010: i32 = GlobalAddress<i32* @gI> 0 [ORD=3] [ID=-3]
  
      0x7fad7102cc10: <multiple use>
    0x7fad7102d110: i32,ch = load 0x7fad7102cf10, 0x7fad7102d010, 
    0x7fad7102cc10<LD4[@gI]> [ORD=3] [ID=-3]
    ...
  Legalized selection DAG: BB#0 '_Z11test_globalv:'
  SelectionDAG has 14 nodes:
    0x7ff3c9c10b98: ch = EntryToken [ORD=1] [ID=0]
    ...
        0x7fad7102cc10: <multiple use>
      0x7fad7102cf10: ch = store 0x7fad7102cd10, 0x7fad7102ca10, 0x7fad7102ce10, 
      0x7fad7102cc10<ST4[%c]> [ORD=2] [ID=8]
  
        0x7fad70c10b98: <multiple use>
          0x7fad7102d610: i32 = Register %GP
  
          0x7fad7102d310: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=1]
  
        0x7fad7102d710: i32 = Cpu0ISD::Wrapper 0x7fad7102d610, 0x7fad7102d310
  
        0x7fad7102cc10: <multiple use>
      0x7fad7102d810: i32,ch = load 0x7fad70c10b98, 0x7fad7102d710, 
      0x7fad7102cc10<LD4[<unknown>]>
      
      0x7ff3ca02cc10: <multiple use>
    0x7ff3ca02d110: i32,ch = load 0x7ff3ca02cf10, 0x7ff3ca02d810, 
    0x7ff3ca02cc10<LD4[@gI]> [ORD=3] [ID=9]
    ...
	  .set	noreorder
	  .cpload	$6
	  .set	nomacro
    ...
  	ld	$2, %got(gI)($gp)
  	ld	$2, 0($2)
    ...
  	.type	gStart,@object          # @gStart
  	.data
  	.globl	gStart
  	.align	2
  gStart:
  	.4byte	2                       # 0x2
  	.size	gStart, 4
  
  	.type	gI,@object              # @gI
  	.globl	gI
  	.align	2
  gI:
  	.4byte	100                     # 0x64
  	.size	gI, 4


Summary above information to Table: Cpu0 global variable options.

.. table:: Cpu0 global variable options

  ============================  =========  ===================  =================================================
  option name                   default    other option value   discription
  ============================  =========  ===================  =================================================
  -relocation-model             pic        static               - pic: Postion Independent Address
                                                                - static: Absolute Address
  -cpu0-use-small-section       false      true                 - false: .data or .bss, 32 bits addressable
                                                                - true: .sdata or .sbss, 16 bits addressable
  ============================  =========  ===================  =================================================
  

.. csv-table:: Cpu0 DAGs and instructions for -relocation-model=static
   :header: "option: cpu0-use-small-section", "false", "true"
   :widths: 20, 20, 20

   "addressing mode", "absolute", "$gp relative"
   "addressing", "absolute", "$gp+offset"
   "Legalized selection DAG", "(add Cpu0ISD::Hi<gI offset Hi16> Cpu0ISD::Lo<gI offset Lo16>)", "(add register %GP, Cpu0ISD::GPRel<gI offset>)"
   "Cpu0", "lui $2, %hi(gI); ori $2, $2, %lo(gI);", "ori	$2, $gp, %gp_rel(gI);"
   "relocation records solved", "link time", "link time"

- In static, cpu0-use-small-section=true, offset between gI and .data can be calculated since the $gp is assigned at fixed address of the start of global address table.
- In "static, cpu0-use-small-section=false", the gI high and low address (%hi(gI) and %lo(gI)) are translated into absolute address. 

.. csv-table:: Cpu0 DAGs and instructions for -relocation-model=pic
   :header: "option: cpu0-use-small-section", "false", "true"
   :widths: 20, 20, 20

   "addressing mode","$gp relative", "$gp relative"
   "addressing", "$gp+offset", "$gp+offset"
   "Legalized selection DAG", "(load (Cpu0ISD::Wrapper register %GP, <gI offset>))", "(load EntryToken, (Cpu0ISD::Wrapper (add Cpu0ISD::Hi<gI offset Hi16>, Register %GP), Cpu0ISD::Lo<gI offset Lo16>))"
   "Cpu0", "ld $2, %got(gI)($gp);", "lui	$2, %got_hi(gI); add $2, $2, $gp; ld $2, %got_lo(gI)($2);"
   "relocation records solved", "link/load time", "link/load time"

- In pic, offset between gI and .data cannot be calculated if the function is 
  loaded at run time (dynamic link); the offset can be calculated if use static 
  link.
- In C, all variable names binding staticly. In C++, the overload variable or 
  function are binding dynamicly.

According book of system program, there are Absolute Addressing Mode and 
Position Independent Addressing Mode. The dynamic function must be compiled with 
Position Independent Addressing Mode. In general, option -relocation-model is 
used to generate either Absolute Addressing or Position Independent Addressing.
The exception is -relocation-model=static and -cpu0-use-small-section=false.
In this case, the register $gp is reserved to set at the start address of global 
variable area. Cpu0 uses $gp relative addressing in this mode.

To support global variable, first add **UseSmallSectionOpt** command variable to 
Cpu0Subtarget.cpp. 
After that, user can run llc with option ``llc -cpu0-use-small-section=false`` 
to specify **UseSmallSectionOpt** to false. 
The default of **UseSmallSectionOpt** is false if without specify it further. 
About the **cl::opt** command line variable, you can refer to here [#]_ further.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0Subtarget.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.h
    :start-after: #if CH >= CH6_1 //1
    :end-before: #endif

.. code-block:: c++

  class Cpu0Subtarget : public Cpu0GenSubtargetInfo {
    ...

.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.h
    :start-after: #if CH >= CH6_1 //RM
    :end-before: #endif //TM
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.h
    :start-after: #if CH >= CH6_1 //hasSlt
    :end-before: #endif //abiUsesSoftFloat

.. code-block:: c++

    ...
  };

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0Subtarget.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.cpp
    :start-after: #if CH >= CH6_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.cpp
    :start-after: //@1 {
    :end-before: //@1 }
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.cpp
    :start-after: #if CH >= CH6_1 //2
    :end-before: #ifdef ENABLE_GPRESTORE

.. code-block:: c++

    ...
  }


The options ReserveGPOpt and NoCploadOpt will used in Cpu0 linker at later 
Chapter.
Next add the following code to files Cpu0BaseInfo.h, Cpu0TargetObjectFile.h, 
Cpu0TargetObjectFile.cpp, Cpu0RegisterInfo.cpp and Cpu0ISelLowering.cpp.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0BaseInfo.h
.. code-block:: c++

  enum TOF {
    ...
    /// MO_GOT16 - Represents the offset into the global offset table at which
    /// the address the relocation entry symbol resides during execution.
    MO_GOT16,
    MO_GOT,
  ...
  }; // enum TOF {

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0TargetObjectFile.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetObjectFile.h
    :start-after: #if CH >= CH6_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0TargetObjectFile.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetObjectFile.cpp
    :start-after: #if CH >= CH6_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0RegisterInfo.cpp
.. code-block:: c++

  BitVector Cpu0RegisterInfo::
  getReservedRegs(const MachineFunction &MF) const {
    ...
      Reserved.set(Cpu0::GP);
    ...
  }

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH6_1 //getGlobalReg
    :end-before: #endif


.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //2
    :end-before: #endif

.. code-block:: c++

  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #if CH >= CH8_1
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #endif //#if CH >= CH8_1
    :end-before: #if CH >= CH12_1
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #endif //#if CH >= CH12_1 //7
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //4
    :end-before: #endif


The setOperationAction(ISD::GlobalAddress, MVT::i32, Custom) tells ``llc`` that 
we implement global address operation in C++ function 
Cpu0TargetLowering::LowerOperation(). LLVM will call this function only when 
llvm want to translate IR DAG of loading global variable into machine code. 
Although all the Custom type of IR operations set by
setOperationAction(ISD::XXX, MVT::XXX, Custom) in construction function 
Cpu0TargetLowering() will invoke llvm to call 
Cpu0TargetLowering::LowerOperation() in stage "Legalized selection DAG", the 
global address access operation can be identified by checking whether the opcode
of DAG Node is ISD::GlobalAddress or not, furthmore. 

Finally, add the following code in Cpu0ISelDAGToDAG.cpp and Cpu0InstrInfo.td.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelDAGToDAG.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.h
    :start-after: #if CH >= CH6_1 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelDAGToDAG.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: #if CH >= CH6_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: //@SelectAddr {
    :end-before: //@SelectAddr }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: #if CH >= CH6_1 //2
    :end-before: #endif

.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: //@Select {
    :end-before: //@Select }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #endif

.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 1
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 2
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 3
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 4
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 5
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 6
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 7
    :end-before: //#endif


Static mode
-------------

From Table: Cpu0 global variable options, option cpu0-use-small-section=false 
puts the global varibale in data/bss while cpu0-use-small-section=true puts in 
sdata/sbss. The sdata stands for small data area.
Section data and sdata are areas for global variables with initial value (such 
as int gI = 100 in this example) while Section bss and sbss are areas for 
global variables without initial value (for instance, int gI;).

data or bss
~~~~~~~~~~~~

The data/bss are 32 bits addressable areas since Cpu0 is a 32 bits architecture. 
Option cpu0-use-small-section=false will generate the following instructions.

.. code-block:: bash

    ...
    lui $2, %hi(gI)
    ori $2, $2, %lo(gI)
    ld	$2, 0($2)
    ...
    .type	gStart,@object          # @gStart
    .data
    .globl	gStart
    .align	2
  gStart:
    .4byte	2                       # 0x2
    .size	gStart, 4
  
    .type	gI,@object              # @gI
    .globl	gI
    .align	2
  gI:
    .4byte	100                     # 0x64
    .size	gI, 4
  	
As above code, it loads the high address part of gI PC relative address 
(16 bits) to register $2 and shift 16 bits. 
Now, the register $2 got it's high part of gI absolute address. 
Next, it adds register $2 and low part of gI absolute address into $2. 
At this point, it gets the gI memory address. Finally, it gets the gI content by 
instruction "ld $2, 0($2)". 
The ``llc -relocation-model=static`` is for absolute address mode which must be 
used in static link mode. The dynamic link must be encoded with Position 
Independent Addressing. 
As you can see, the PC relative address can be solved in static link (
The offset between the address of gI and instruction "lui $2, %hi(gI)" can be 
caculated). Since Cpu0 uses PC relative address coding, this program can be 
loaded to any address and run correctly there.
If this program uses absolute address and can be loaded at a specific address 
known at link stage, the relocation record of gI variable access instruction 
such as "lui $2, %hi(gI)" and "ori	$2, $2, %lo(gI)" can be solved 
at link time. On the other hand, 
if this program use absolute address and the loading address is known at load 
time, then this relocation record will be solved by loader at load time. 

IsGlobalInSmallSection() returns true or false depends on UseSmallSectionOpt. 

The code fragment of lowerGlobalAddress() as the following corresponding option 
``llc -relocation-model=static -cpu0-use-small-section=false`` will translate 
DAG (GlobalAddress<i32* @gI> 0) into 
(add Cpu0ISD::Hi<gI offset Hi16> Cpu0ISD::Lo<gI offset Lo16>) in 
stage "Legalized selection DAG" as below.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@getAddrNonPIC
    :end-before: #endif // #if CH >= CH6_1

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@getTargetNode(GlobalAddressSDNode
    :end-before: //@getTargetNode(ExternalSymbolSDNode

.. code-block:: c++

  SDValue Cpu0TargetLowering::lowerGlobalAddress(SDValue Op,
                                                 SelectionDAG &DAG) const {
    ...
    EVT Ty = Op.getValueType();
    GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
    ..

    if (getTargetMachine().getRelocationModel() != Reloc::PIC_) {
      ...
      // %hi/%lo relocation
      return getAddrNonPIC(N, Ty, DAG);
    }
    ...
  }

.. code-block:: bash

  118-165-78-166:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch6_1.cpp -emit-llvm -o ch6_1.bc
  118-165-78-166:input Jonathan$ ~/llvm/test/cmake_debug_build/Debug/bin/llc 
  -march=cpu0 -relocation-model=static -cpu0-use-small-section=false 
  -filetype=asm -debug ch6_1.bc -o -
  
  ...
  Type-legalized selection DAG: BB#0 '_Z3funv:entry'
  SelectionDAG has 12 nodes:
    ...
        0x7ffd5902cc10: <multiple use>
      0x7ffd5902cf10: ch = store 0x7ffd5902cd10, 0x7ffd5902ca10, 0x7ffd5902ce10, 
      0x7ffd5902cc10<ST4[%c]> [ORD=2] [ID=-3]
  
      0x7ffd5902d010: i32 = GlobalAddress<i32* @gI> 0 [ORD=3] [ID=-3]
  
      0x7ffd5902cc10: <multiple use>
    0x7ffd5902d110: i32,ch = load 0x7ffd5902cf10, 0x7ffd5902d010, 
    0x7ffd5902cc10<LD4[@gI]> [ORD=3] [ID=-3]
    ...
  
  Legalized selection DAG: BB#0 '_Z3funv:entry'
  SelectionDAG has 16 nodes:
    ...
        0x7ffd5902cc10: <multiple use>
      0x7ffd5902cf10: ch = store 0x7ffd5902cd10, 0x7ffd5902ca10, 0x7ffd5902ce10, 
      0x7ffd5902cc10<ST4[%c]> [ORD=2] [ID=8]
  
          0x7ffd5902d310: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=5]
  
        0x7ffd5902d710: i32 = Cpu0ISD::Hi 0x7ffd5902d310
  
          0x7ffd5902d610: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=6]
  
        0x7ffd5902d810: i32 = Cpu0ISD::Lo 0x7ffd5902d610
  
      0x7ffd5902fe10: i32 = add 0x7ffd5902d710, 0x7ffd5902d810
  
      0x7ffd5902cc10: <multiple use>
    0x7ffd5902d110: i32,ch = load 0x7ffd5902cf10, 0x7ffd5902fe10, 
    0x7ffd5902cc10<LD4[@gI]> [ORD=3] [ID=9]


Finally, the pattern defined in Cpu0InstrInfo.td as the following will translate  
DAG (add Cpu0ISD::Hi<gI offset Hi16> Cpu0ISD::Lo<gI offset Lo16>) into Cpu0 
instructions as below.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 1
    :end-before: def Cpu0GPRel
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 4
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 5
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 6
    :end-before: //#endif

.. code-block:: bash

    ...
    lui	$2, %hi(gI)
    ori	$2, $2, %lo(gI)
    ...


As above, Pat<(...),(...)> include two lists of DAGs. 
The left is IR DAG and the right is machine instruction DAG. 
"Pat<(Cpu0Hi tglobaladdr:$in), (LUi, tglobaladdr:$in)>;" will 
translate DAG (Cpu0ISD::Hi tglobaladdr) into (lui (ori ZERO, tglobaladdr), 16).
"Pat<(add CPURegs:$hi, (Cpu0Lo tglobaladdr:$lo)), (ORi CPURegs:$hi, 
tglobaladdr:$lo)>;" will translate DAG (add Cpu0ISD::Hi, Cpu0ISD::Lo) into Cpu0 
instruction (ori Cpu0ISD::Hi, Cpu0ISD::Lo).


sdata or sbss
~~~~~~~~~~~~~~

The sdata/sbss are 16 bits addressable areas which placed in ELF for fast access. 
Option cpu0-use-small-section=true will generate the following instructions.

.. code-block:: bash

    ori	$2, $gp, %gp_rel(gI)
    ld	$2, 0($2)
    ...
    .type	gStart,@object          # @gStart
    .section	.sdata,"aw",@progbits
    .globl	gStart
    .align	2
  gStart:
    .4byte	2                       # 0x2
    .size	gStart, 4
  
    .type	gI,@object              # @gI
    .globl	gI
    .align	2
  gI:
    .4byte	100                     # 0x64
    .size	gI, 4


The code fragment of lowerGlobalAddress() as the following corresponding option 
``llc -relocation-model=static -cpu0-use-small-section=true`` will translate DAG 
(GlobalAddress<i32* @gI> 0) into 
(add register %GP Cpu0ISD::GPRel<gI offset>) in 
stage "Legalized selection DAG" as below.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //4
    :end-before: //@ %hi/%lo relocation

.. code-block:: c++

      ...
    }
    ...
  }

.. code-block:: bash

  ...
  Type-legalized selection DAG: BB#0 '_Z3funv:entry'
  SelectionDAG has 12 nodes:
    ...
        0x7fc5f382cc10: <multiple use>
      0x7fc5f382cf10: ch = store 0x7fc5f382cd10, 0x7fc5f382ca10, 0x7fc5f382ce10, 
      0x7fc5f382cc10<ST4[%c]> [ORD=2] [ID=-3]
  
      0x7fc5f382d010: i32 = GlobalAddress<i32* @gI> 0 [ORD=3] [ID=-3]
  
      0x7fc5f382cc10: <multiple use>
    0x7fc5f382d110: i32,ch = load 0x7fc5f382cf10, 0x7fc5f382d010, 
    0x7fc5f382cc10<LD4[@gI]> [ORD=3] [ID=-3]
  
  Legalized selection DAG: BB#0 '_Z3funv:entry'
  SelectionDAG has 15 nodes:
    ...
        0x7fc5f382cc10: <multiple use>
      0x7fc5f382cf10: ch = store 0x7fc5f382cd10, 0x7fc5f382ca10, 0x7fc5f382ce10, 
      0x7fc5f382cc10<ST4[%c]> [ORD=2] [ID=8]
  
        0x7fc5f382d710: i32 = register %GP
  
          0x7fc5f382d310: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=4]
  
        0x7fc5f382d610: i32 = Cpu0ISD::GPRel 0x7fc5f382d310
  
      0x7fc5f382d810: i32 = add 0x7fc5f382d710, 0x7fc5f382d610
  
      0x7fc5f382cc10: <multiple use>
    0x7fc5f382d110: i32,ch = load 0x7fc5f382cf10, 0x7fc5f382d810, 
    0x7fc5f382cc10<LD4[@gI]> [ORD=3] [ID=9]
    ...


Finally, the pattern defined in Cpu0InstrInfo.td as the following will translate  
DAG (add register %GP Cpu0ISD::GPRel<gI offset>) into Cpu0 
instruction as below. 

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: def Cpu0Lo
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 7
    :end-before: //@ wrapper_pic

.. code-block:: bash

    ori	$2, $gp, %gp_rel(gI)
    ...

"Pat<(add CPURegs:$gp, (Cpu0GPRel tglobaladdr:$in)), (ADD CPURegs:$gp, (ORi 
ZERO, tglobaladdr:$in))>;" will translate (add register %GP Cpu0ISD::GPRel 
tglobaladdr) into (add $gp, (ori ZERO, tglobaladdr)).

In this mode, the $gp content is assigned at compile/link time, changed only at 
program be loaded, and is fixed during the program running; on the contrary, 
when -relocation-model=pic the $gp can be changed during program running. 
For this example code, if $gp is assigned to the start address of .sdata by loader 
when program ch6_1.cpu0.s is loaded, then linker can caculate %gp_rel(gI) (= 
the relative address distance between gI and start of .sdata section). 
Which meaning this relocation record can be solved at link time, that's why it 
is static mode. 

In this mode, we reserve $gp to a specfic fixed address of the program is 
loaded. As a result, the $gp cannot be allocated as a general purpose for 
variables. The following code tells llvm never allocate $gp for variables.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0Subtarget.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.cpp
    :start-after: //@1 {
    :end-before: //@1 }
.. literalinclude:: ../lbdex/Cpu0/Cpu0Subtarget.cpp
    :start-after: #if CH >= CH6_1 //2
    :end-before: #endif //#if CH >= CH6_1

.. code-block:: c++

  }

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0RegisterInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: //@getReservedRegs {
    :end-before: static const uint16_t ReservedCPURegs[] = {
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: #if CH >= CH6_1
    :end-before: #endif //#if CH >= CH6_1

.. code-block:: c++

    ...
  }


pic mode
---------

sdata or sbss
~~~~~~~~~~~~~~

Option ``llc -relocation-model=pic -cpu0-use-small-section=true`` will 
generate the following instructions.

.. code-block:: bash

    ...
    .set	noreorder
    .cpload	$6
    .set	nomacro
    ...
    ld	$2, %got(gI)($gp)
    ld	$2, 0($2)
    ...
    .type	gStart,@object          # @gStart
    .data
    .globl	gStart
    .align	2
  gStart:
    .4byte	2                       # 0x2
    .size	gStart, 4
  
    .type	gI,@object              # @gI
    .globl	gI
    .align	2
  gI:
    .4byte	100                     # 0x64
    .size	gI, 4

The following code fragment of Cpu0AsmPrinter.cpp will emit **.cpload** asm 
pseudo instruction at function entry point as below.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0MachineFunction.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: //@1 {
    :end-before: #if CH >= CH3_4 //1
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH6_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH6_1 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH6_1 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.h
    :start-after: #if CH >= CH6_1 //4
    :end-before: #endif

.. code-block:: c++

    ...
  };

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0MachineFunction.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0MachineFunction.cpp
    :start-after: #if CH >= CH6_1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0AsmPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: //@-> .set  nomacro
    :end-before: MCInstLowering.Initialize(&MF->getContext());
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH6_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH6_1 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0AsmPrinter.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #endif

.. code-block:: c++

  }

.. code-block:: bash

    ...
    .set	noreorder
    .cpload	$6
    .set	nomacro
    ...

The **.cpload** is the assembly directive (macro) which 
will expand to several instructions. 
Issue **.cpload** before **.set nomacro** since the **.set nomacro** option 
causes the assembler to print a warning message whenever 
an assembler operation generates more than one machine language instruction, 
reference Mips ABI [#]_.

Following code will exspand .cpload into machine instructions as below. 
"0fa00000 09aa0000 13aa6000" is the **.cpload** machine instructions 
displayed in comments of Cpu0MCInstLower.cpp.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0MCInstLower.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.h
    :start-after: //@1 {
    :end-before: //@2
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.h
    :start-after: #if CH >= CH6_1 //1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.h
    :start-after: #if CH >= CH6_1 //2
    :end-before: #endif

.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0MCInstLower.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH6_1 //2
    :end-before: #endif

.. code-block:: bash

  118-165-76-131:input Jonathan$ /Users/Jonathan/llvm/test/
  cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=
  obj ch6_1.bc -o ch6_1.cpu0.o
  118-165-76-131:input Jonathan$ gobjdump -s ch6_1.cpu0.o 

  ch6_1.cpu0.o:     file format elf32-big

  Contents of section .text:
   0000 0fa00000 0daa0000 13aa6000  ...
  ...

  118-165-76-131:input Jonathan$ gobjdump -tr ch6_1.cpu0.o 
  ...
  RELOCATION RECORDS FOR [.text]:
  OFFSET   TYPE              VALUE 
  00000000 UNKNOWN           _gp_disp
  00000008 UNKNOWN           _gp_disp
  00000020 UNKNOWN           gI

.. note::

  // **Mips ABI: _gp_disp**
  After calculating the gp, a function allocates the local stack space and saves 
  the gp on the stack, so it can be restored after subsequent function calls. 
  In other words, the gp is a caller saved register. 
  
  ...
  
  _gp_disp represents the offset between the beginning of the function and the 
  global offset table. 
  Various optimizations are possible in this code example and the others that 
  follow. 
  For example, the calculation of gp need not be done for a position-independent 
  function that is strictly local to an object module. 

The _gp_disp as above is a relocation record, it means both the machine 
instructions 0da00000 (offset 0) and 0daa0000 (offset 8) which equal to assembly 
"ori $gp, $zero, %hi(_gp_disp)" and assembly "ori $gp, $gp, %lo(_gp_disp)",
respectively, are relocated records depend on _gp_disp. 
The loader or OS can caculate _gp_disp by (x - start address of .data) 
when load the dynamic function into memory x, and adjusts these two 
instructions offet correctly.
Since shared function is loaded when this function is called, the relocation 
record "ld $2, %got(gI)($gp)" cannot be resolved in link time. 
In spite of the reloation record is solved on load time, the name binding 
is static, since linker deliver the memory address to loader, and loader can solve 
this just by caculate the offset directly. The memory reference bind with 
the offset of _gp_disp at link time.
The ELF relocation records will be introduced in Chapter ELF Support. 
So, don't worry if you don't quite understand it at this point.

The code fragment of lowerGlobalAddress() as the following corresponding option 
``llc -relocation-model=pic`` will translate DAG (GlobalAddress<i32* @gI> 0) into  
(load EntryToken, (Cpu0ISD::Wrapper Register %GP, TargetGlobalAddress<i32* @gI> 0)) 
in stage "Legalized selection DAG" as below.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@getAddrGlobal {
    :end-before: //@getAddrGlobal }

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //4
    :end-before: //@lowerGlobalAddress }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@lga 1 {
    :end-before: //@lga 1 }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@ %gp_rel relocation
    :end-before: //@ %hi/%lo relocation

.. code-block:: c++

    ...
  }
    
.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelDAGToDAG.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: //@SelectAddr {
    :end-before: //@SelectAddr }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelDAGToDAG.cpp
    :start-after: #if CH >= CH6_1 //2
    :end-before: //@static

.. code-block:: c++

    ...
  }

.. code-block:: bash

  ...
  Type-legalized selection DAG: BB#0 '_Z3funv:entry'
  SelectionDAG has 12 nodes:
    ...
        0x7fad7102cc10: <multiple use>
      0x7fad7102cf10: ch = store 0x7fad7102cd10, 0x7fad7102ca10, 0x7fad7102ce10, 
      0x7fad7102cc10<ST4[%c]> [ORD=2] [ID=-3]
  
      0x7fad7102d010: i32 = GlobalAddress<i32* @gI> 0 [ORD=3] [ID=-3]
  
      0x7fad7102cc10: <multiple use>
    0x7fad7102d110: i32,ch = load 0x7fad7102cf10, 0x7fad7102d010, 
    0x7fad7102cc10<LD4[@gI]> [ORD=3] [ID=-3]
    ...
  Legalized selection DAG: BB#0 '_Z3funv:entry'
  SelectionDAG has 15 nodes:
    0x7ff3c9c10b98: ch = EntryToken [ORD=1] [ID=0]
    ...
        0x7fad7102cc10: <multiple use>
      0x7fad7102cf10: ch = store 0x7fad7102cd10, 0x7fad7102ca10, 0x7fad7102ce10, 
      0x7fad7102cc10<ST4[%c]> [ORD=2] [ID=8]
  
        0x7fad70c10b98: <multiple use>
          0x7fad7102d610: i32 = Register %GP
  
          0x7fad7102d310: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=1]
  
        0x7fad7102d710: i32 = Cpu0ISD::Wrapper 0x7fad7102d610, 0x7fad7102d310
  
        0x7fad7102cc10: <multiple use>
      0x7fad7102d810: i32,ch = load 0x7fad70c10b98, 0x7fad7102d710, 
      0x7fad7102cc10<LD4[<unknown>]>
      0x7ff3ca02cc10: <multiple use>
    0x7ff3ca02d110: i32,ch = load 0x7ff3ca02cf10, 0x7ff3ca02d810, 
    0x7ff3ca02cc10<LD4[@gI]> [ORD=3] [ID=9]
    ...


Finally, the pattern Cpu0 instruction **ld** defined before in Cpu0InstrInfo.td 
will translate DAG (load EntryToken, (Cpu0ISD::Wrapper Register %GP, 
TargetGlobalAddress<i32* @gI> 0)) into Cpu0 instruction as follows,

.. code-block:: bash

    ...
    ld	$2, %got(gI)($gp)
    ...

Remind in pic mode, Cpu0 uses ".cpload" and "ld $2, %got(gI)($gp)" to access 
global variable as Mips. It takes 4 instructions in both Cpu0 and Mips. 
The cost came from we didn't assume that register $gp is always assigned to 
address .sdata and fixed there. Even we reserve $gp in this function, the $gp
register can be changed at other functions. In last sub-section, the $gp is
assumed to preserved at any function. If $gp is fixed during the run time, then 
".cpload" can be removed here and have only one instruction cost in global 
variable access. The advantage of ".cpload" removing come from losing one 
general purpose register $gp which can be allocated for variables. 
In last sub-section, .sdata mode, we use ".cpload" removing since it is 
static link.
In pic mode, the dynamic loading takes too much time.
Romove ".cpload" with the cost of losing one general purpose register at all
functions is not deserved here. 
The relocation records of ".cpload" from ``llc -relocation-model=pic`` can also 
be solved in link stage if we want to link this function by static link.


data or bss
~~~~~~~~~~~~~

The code fragment of lowerGlobalAddress() as the following corresponding option 
``llc -relocation-model=pic`` will translate DAG (GlobalAddress<i32* @gI> 0) into  
(load EntryToken, (Cpu0ISD::Wrapper (add Cpu0ISD::Hi<gI offset Hi16>, Register %GP), 
TargetGlobalAddress<i32* @gI> 0)) 
in stage "Legalized selection DAG" as below.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: //@getAddrGlobalLargeGOT {
    :end-before: //@getAddrGlobalLargeGOT }

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0ISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //4
    :end-before: //@lowerGlobalAddress }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@lga 1 {
    :end-before: //@lga 1 }
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@large section
    :end-before: #endif

.. code-block:: bash

  ...
  Type-legalized selection DAG: BB#0 '_Z3funv:'
  SelectionDAG has 10 nodes:
    ...
      0x7fb77a02cd10: ch = store 0x7fb779c10a08, 0x7fb77a02ca10, 0x7fb77a02cb10, 
      0x7fb77a02cc10<ST4[%c]> [ORD=1] [ID=-3]
  
      0x7fb77a02ce10: i32 = GlobalAddress<i32* @gI> 0 [ORD=2] [ID=-3]
  
      0x7fb77a02cc10: <multiple use>
    0x7fb77a02cf10: i32,ch = load 0x7fb77a02cd10, 0x7fb77a02ce10, 
    0x7fb77a02cc10<LD4[@gI]> [ORD=2] [ID=-3]
    ...
  
  Legalized selection DAG: BB#0 '_Z3funv:'
  SelectionDAG has 16 nodes:
    ...
      0x7fb77a02cd10: ch = store 0x7fb779c10a08, 0x7fb77a02ca10, 0x7fb77a02cb10, 
      0x7fb77a02cc10<ST4[%c]> [ORD=1] [ID=6]
  
        0x7fb779c10a08: <multiple use>
              0x7fb77a02d110: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=19]
  
            0x7fb77a02d410: i32 = Cpu0ISD::Hi 0x7fb77a02d110
  
            0x7fb77a02d510: i32 = Register %GP
  
          0x7fb77a02d610: i32 = add 0x7fb77a02d410, 0x7fb77a02d510
  
          0x7fb77a02d710: i32 = TargetGlobalAddress<i32* @gI> 0 [TF=20]
  
        0x7fb77a02d810: i32 = Cpu0ISD::Wrapper 0x7fb77a02d610, 0x7fb77a02d710
  
        0x7fb77a02cc10: <multiple use>
      0x7fb77a02fe10: i32,ch = load 0x7fb779c10a08, 0x7fb77a02d810, 
      0x7fb77a02cc10<LD4[GOT]>
      
      0x7fb77a02cc10: <multiple use>
    0x7fb77a02cf10: i32,ch = load 0x7fb77a02cd10, 0x7fb77a02fe10, 
    0x7fb77a02cc10<LD4[@gI]> [ORD=2] [ID=7]
    ...


Finally, the pattern Cpu0 instruction **ld** defined before in Cpu0InstrInfo.td 
will translate DAG (load EntryToken, (Cpu0ISD::Wrapper (add Cpu0ISD::Hi<gI 
offset Hi16>, Register %GP), Cpu0ISD::Lo<gI offset Lo16>)) into Cpu0 
instructions as below.

.. code-block:: bash

    ...
    ori	$2, $zero, %got_hi(gI)
    shl	$2, $2, 16
    add	$2, $2, $gp
    ld	$2, %got_lo(gI)($2)
    ...

The following code in Cpu0InstrInfo.td is needed for example input ch8_5.cpp. 
Since ch8_5.cpp uses llvm IR **select**, it cannot be run at this point. It 
will be run in later Chapter Control flow statements.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //#if CH >= CH6_1 2
    :end-before: //#endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: //@ wrapper_pic
    :end-before: //#endif

.. rubric:: lbdex/input/ch8_5.cpp
.. literalinclude:: ../lbdex/input/ch8_5.cpp
    :start-after: /// start


Global variable print support
-------------------------------

Above code is for global address DAG translation. 
Next, add the following code to Cpu0MCInstLower.cpp, Cpu0InstPrinter.cpp and 
Cpu0ISelLowering.cpp for global variable printing operand function.

.. rubric:: lbdex/chapters/Chapter6_1/Cpu0MCInstLower.cpp
.. literalinclude:: ../lbdex/chapters/Chapter6_1/Cpu0MCInstLower.cpp
    :start-after: //@LowerSymbolOperand {
    :end-before: //@LowerSymbolOperand }
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: //@LowerOperand {
    :end-before: //@2
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #endif

.. code-block:: c++

    ...
    }
  ...
  }
    
.. rubric:: lbdex/chapters/Chapter6_1/InstPrinter/Cpu0InstPrinter.cpp
.. literalinclude:: ../lbdex/Cpu0/InstPrinter/Cpu0InstPrinter.cpp
    :start-after: //@printExpr {
    :end-before: //@printExpr body {
.. literalinclude:: ../lbdex/Cpu0/InstPrinter/Cpu0InstPrinter.cpp
    :start-after: #if CH >= CH6_1 //VK_Cpu0_GPREL
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/InstPrinter/Cpu0InstPrinter.cpp
    :start-after: #if CH >= CH6_1 //VK_Cpu0_GOT16
    :end-before: #endif

.. code-block:: c++

      ...
    }
    ...
  }

The following function is for ``llc -debug`` this chapter DAG node name printing.
It is added at Chapter3_1 already.
  
.. rubric:: lbdex/chapters/Chapter3_1/Cpu0ISelLowering.cpp
.. code-block:: c++

  const char *Cpu0TargetLowering::getTargetNodeName(unsigned Opcode) const {
    switch (Opcode) {
    ..
    case Cpu0ISD::GPRel:             return "Cpu0ISD::GPRel";
    ...
    case Cpu0ISD::Wrapper:           return "Cpu0ISD::Wrapper";
    ...
    }
  }



OS is the output stream which output to the assembly file.


Summary
--------

The global variable Instruction Selection for DAG translation is not like the 
ordinary IR node translation, it has static (absolute address) and pic mode. 
Backend deals this translation by create DAG nodes in function 
lowerGlobalAddress() which called by LowerOperation(). 
Function LowerOperation() takes care all Custom type of operation. 
Backend set global address as Custom operation by 
**”setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);”** in 
Cpu0TargetLowering() constructor. 
Different address mode create their own DAG list at run time. 
By setting the pattern Pat<> in Cpu0InstrInfo.td, the llvm can apply the 
compiler mechanism, pattern match, in the Instruction Selection stage.

There are three types for setXXXAction(), Promote, Expand and Custom. 
Except Custom, the other two maybe no need to coding. 
Here [#]_ is the references.

As shown in this chapter, the global variable can be laid in 
.sdata/.sbss by option -cpu0-use-small-section=true. 
It is possible that the variables of small data section (16 bits
addressable) are full out at link stage. When that happens, linker will 
highlights that error and forces the toolchain users to fix it. 
As the result, the toolchain user need to reconsider which global variables 
should be moved from .sdata/.sbss to .data/.bss by set option 
-cpu0-use-small-section=false in Makefile as follows,

.. rubric:: Makefile
.. code-block:: c++

  # Set the global variables declared in a.cpp to .data/.bss
  llc  -march=cpu0 -relocation-model=static -cpu0-use-small-section=false \
  -filetype=obj a.bc -o a.cpu0.o
  # Set the global variables declared in b.cpp to .sdata/.sbss
  llc  -march=cpu0 -relocation-model=static -cpu0-use-small-section=true \
  -filetype=obj b.bc -o b.cpu0.o

The rule for global variables allocation is "set the small and frequent
variables in small 16 addressable area".



.. _section Global variable:
    http://jonathan2251.github.io/lbd/globalvar.html#global-variable

.. _section Array and struct support:
    http://jonathan2251.github.io/lbd/globalvar.html#array-and-struct-support

.. [#] http://llvm.org/docs/CommandLine.html

.. [#] http://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf

.. [#] http://llvm.org/docs/WritingAnLLVMBackend.html#the-selectiondag-legalize-phase
