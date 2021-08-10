add_llvm_component_library(LLVMCpu0Disassembler
  Cpu0Disassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  Cpu0Info
  Support

  ADD_TO_COMPONENT
  Cpu0
  )
