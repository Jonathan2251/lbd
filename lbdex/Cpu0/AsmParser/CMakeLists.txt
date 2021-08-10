add_llvm_component_library(LLVMCpu0AsmParser
  Cpu0AsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  Cpu0Desc
  Cpu0Info
  Support

  ADD_TO_COMPONENT
  Cpu0
  )
