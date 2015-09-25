# MCTargetDesc/CMakeLists.txt
add_llvm_library(LLVMCpu0Desc
#if CH >= CH3_1
  Cpu0ABIInfo.cpp
#endif
#if CH >= CH3_2
  Cpu0MCAsmInfo.cpp
#endif
#if CH >= CH5_1
  Cpu0AsmBackend.cpp
  Cpu0MCCodeEmitter.cpp
  Cpu0ELFObjectWriter.cpp
  Cpu0TargetStreamer.cpp
#endif
  Cpu0MCTargetDesc.cpp
  )
