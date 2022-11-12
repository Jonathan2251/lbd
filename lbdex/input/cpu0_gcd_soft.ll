; ~/llvm/test/build/bin/llc -debug -print-after-all -march=cpu0 -mcpu=cpu032II -relocation-model=pic cpu0_gcd_soft.ll -o -

define i32 @llvm_cpu0_gcd_soft_test(i32 %a, i32 %b) nounwind {
entry:
  %res = tail call i32 @cpu0_gcd_soft(i32 %a, i32 %b)
  ret i32 %res
}

declare i32 @cpu0_gcd_soft(i32, i32) nounwind

