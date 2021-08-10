; ~/llvm/test/build/bin/llc -debug -march=cpu0 -mcpu=cpu0II -relocation-model=pic cpu0_sqrt.ll -o -

define i32 @llvm_cpu0_sqrt_test(i32 %a, i32 %b) nounwind {
entry:
  %res = tail call i32 @llvm.cpu0.sqrt(i32 %a, i32 %b)
  ret i32 %res
}

declare i32 @llvm.cpu0.sqrt(i32, i32) nounwind

