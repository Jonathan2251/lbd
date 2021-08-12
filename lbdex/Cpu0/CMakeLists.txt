add_llvm_component_group(Cpu0)

#if CH >= CH11_1 1
set(LLVM_TARGET_DEFINITIONS Cpu0Asm.td)
tablegen(LLVM Cpu0GenAsmMatcher.inc -gen-asm-matcher)
#endif

set(LLVM_TARGET_DEFINITIONS Cpu0Other.td)

# Generate Cpu0GenRegisterInfo.inc and Cpu0GenInstrInfo.inc which included by 
#  your hand code C++ files. 
# Cpu0GenRegisterInfo.inc came from Cpu0RegisterInfo.td, Cpu0GenInstrInfo.inc 
#  came from Cpu0InstrInfo.td.
tablegen(LLVM Cpu0GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM Cpu0GenInstrInfo.inc -gen-instr-info)
#if CH >= CH10_1 1
tablegen(LLVM Cpu0GenDisassemblerTables.inc -gen-disassembler)
#endif
#if CH >= CH3_2 1
tablegen(LLVM Cpu0GenCodeEmitter.inc -gen-emitter)
tablegen(LLVM Cpu0GenMCCodeEmitter.inc -gen-emitter)

tablegen(LLVM Cpu0GenAsmWriter.inc -gen-asm-writer)
#endif
#if CH >= CH3_1 1
tablegen(LLVM Cpu0GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM Cpu0GenCallingConv.inc -gen-callingconv)
#endif
tablegen(LLVM Cpu0GenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM Cpu0GenMCPseudoLowering.inc -gen-pseudo-lowering)

# Cpu0CommonTableGen must be defined
add_public_tablegen_target(Cpu0CommonTableGen)

# Cpu0CodeGen should match with LLVMBuild.txt Cpu0CodeGen
add_llvm_target(Cpu0CodeGen
#if CH >= CH3_5
  Cpu0AnalyzeImmediate.cpp
#endif
#if CH >= CH3_2 2
  Cpu0AsmPrinter.cpp
  Cpu0MCInstLower.cpp
#endif
#if CH >= CH8_2 //1
  Cpu0DelaySlotFiller.cpp
#endif
#if CH >= CH8_2 //2
  Cpu0DelUselessJMP.cpp
#endif
#if CH >= CH9_3
  Cpu0EmitGPRestore.cpp
#endif
#if CH >= CH3_1 2
  Cpu0FrameLowering.cpp
#endif
#if CH >= CH8_2 //3
  Cpu0LongBranch.cpp
#endif
#if CH >= CH3_1 2.5
  Cpu0InstrInfo.cpp
  Cpu0ISelLowering.cpp
  Cpu0MachineFunction.cpp
  Cpu0RegisterInfo.cpp
  Cpu0SEFrameLowering.cpp
  Cpu0SEInstrInfo.cpp
  Cpu0SEISelLowering.cpp
  Cpu0SERegisterInfo.cpp
  Cpu0Subtarget.cpp
  Cpu0TargetObjectFile.cpp
#endif
#if CH >= CH3_3
  Cpu0ISelDAGToDAG.cpp
  Cpu0SEISelDAGToDAG.cpp
#endif
  Cpu0TargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  MC
  Cpu0Desc
  Cpu0Info
#if CH >= CH11_1 2
  Cpu0AsmParser
#endif
#if CH >= CH3_2 3
  Cpu0AsmPrinter
#endif
#if CH >= CH10_1 2
  Cpu0Disassembler
#endif
  SelectionDAG
  Support
  Target
  GlobalISel

  ADD_TO_COMPONENT
  Cpu0
  )

# Should match with "subdirectories =  MCTargetDesc TargetInfo" in LLVMBuild.txt
#if CH >= CH3_2 4
add_subdirectory(InstPrinter)
#endif
#if CH >= CH10_1 3
add_subdirectory(Disassembler)
#endif
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
#if CH >= CH11_1 3
add_subdirectory(AsmParser)
#endif
