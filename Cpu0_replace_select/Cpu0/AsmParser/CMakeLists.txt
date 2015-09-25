include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )
add_llvm_library(LLVMCpu0AsmParser
  Cpu0AsmParser.cpp
  )

add_dependencies(LLVMCpu0AsmParser Cpu0CommonTableGen)
