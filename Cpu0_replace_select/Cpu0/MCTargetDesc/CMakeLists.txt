# MCTargetDesc/CMakeLists.txt
add_llvm_library(LLVMCpu0Desc
  Cpu0AsmBackend.cpp
  Cpu0MCAsmInfo.cpp
  Cpu0MCCodeEmitter.cpp
  Cpu0MCTargetDesc.cpp
  Cpu0ELFObjectWriter.cpp
  )

add_dependencies(LLVMCpu0Desc Cpu0CommonTableGen)
