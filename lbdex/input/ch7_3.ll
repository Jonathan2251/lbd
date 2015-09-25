; /Users/Jonathan/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch7_3.ll -o -

; /// start
define zeroext i1 @verify_load_bool() #0 {
entry:
  %retval = alloca i1, align 1
  store i1 1, i1* %retval, align 1
  %0 = load i1* %retval
  ret i1 %0
}
