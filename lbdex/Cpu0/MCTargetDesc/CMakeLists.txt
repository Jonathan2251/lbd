# MCTargetDesc/CMakeLists.txt
add_llvm_component_library(LLVMCpu0Desc
#if CH >= CH3_1
  Cpu0ABIInfo.cpp
#endif
#if CH >= CH3_2
  Cpu0MCAsmInfo.cpp
#endif
#if CH >= CH5_1
  Cpu0AsmBackend.cpp
  Cpu0MCCodeEmitter.cpp
  Cpu0MCExpr.cpp
  Cpu0ELFObjectWriter.cpp
  Cpu0TargetStreamer.cpp
#endif
  Cpu0MCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  Cpu0Info
#if CH >= CH3_2 2
  Cpu0AsmPrinter
#endif
  Support

  ADD_TO_COMPONENT
  Cpu0
  )
