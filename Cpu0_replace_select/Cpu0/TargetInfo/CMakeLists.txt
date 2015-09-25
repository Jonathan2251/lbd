include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMCpu0Info
  Cpu0TargetInfo.cpp
  )

add_dependencies(LLVMCpu0Info Cpu0CommonTableGen)
