; ~/llvm/test/cmake_debug_build/Debug/bin/llc -debug -print-after-all -march=cpu0 -mcpu=cpu032II -relocation-model=pic cpu0_gcd.ll -o -

define i32 @llvm_cpu0_gcd_test(i32 %a, i32 %b) nounwind {
entry:
  %res = tail call i32 @llvm.cpu0.gcd(i32 %a, i32 %b)
  ret i32 %res
}

declare i32 @llvm.cpu0.gcd(i32, i32) nounwind

