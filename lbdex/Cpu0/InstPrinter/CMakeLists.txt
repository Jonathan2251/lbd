add_llvm_component_library(LLVMCpu0AsmPrinter
  Cpu0InstPrinter.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  Cpu0
  )
