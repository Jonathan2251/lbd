; RUN: llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic < %s | FileCheck %s
; ~/llvm/test/build/bin/llc -debug -print-after-all -march=cpu0 -mcpu=cpu032II -relocation-model=pic cpu0_gcd.ll -o -

define i32 @llvm_cpu0_gcd_test(i32 %a, i32 %b) nounwind {
entry:
; CHECK:    gcd	$2, $4, $5
  %res = tail call i32 @llvm.cpu0.gcd(i32 %a, i32 %b)
  ret i32 %res
}

declare i32 @llvm.cpu0.gcd(i32, i32) nounwind

