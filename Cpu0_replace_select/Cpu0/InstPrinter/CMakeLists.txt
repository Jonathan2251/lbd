include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMCpu0AsmPrinter
  Cpu0InstPrinter.cpp
  )

add_dependencies(LLVMCpu0AsmPrinter Cpu0CommonTableGen)
