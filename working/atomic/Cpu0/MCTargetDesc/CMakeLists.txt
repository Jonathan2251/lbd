# MCTargetDesc/CMakeLists.txt
add_llvm_library(LLVMCpu0Desc
#if CH >= CH3_2
  Cpu0MCAsmInfo.cpp
#endif
#if CH >= CH5_1
  Cpu0AsmBackend.cpp
  Cpu0MCCodeEmitter.cpp
  Cpu0ELFObjectWriter.cpp
#endif
  Cpu0MCTargetDesc.cpp
  )
