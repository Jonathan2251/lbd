; RUN: llc  -march=cpu0 -relocation-model=pic < %s | FileCheck %s

; ModuleID = 'ch4_3.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: nounwind
define i32 @_Z8test_divv() #0 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %a1 = alloca i32, align 4
  %d1 = alloca i32, align 4
  store i32 -5, i32* %a, align 4
  store i32 2, i32* %b, align 4
  store i32 16777216, i32* %c, align 4
  store i32 0, i32* %d, align 4
  store i32 -5, i32* %a1, align 4
  store i32 0, i32* %d1, align 4
  %0 = load i32* %a, align 4
  %1 = load i32* %b, align 4
; CHECK:  div	${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; CHECK:  mflo	${{[0-9]+|t9}}
  %div = sdiv i32 %0, %1
  store i32 %div, i32* %d, align 4
  %2 = load i32* %a1, align 4
  %3 = load i32* %c, align 4
; CHECK:  divu	${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; CHECK:  mflo	${{[0-9]+|t9}}
  %div1 = udiv i32 %2, %3
  store i32 %div1, i32* %d1, align 4
  %4 = load i32* %d, align 4
  %5 = load i32* %d1, align 4
  %add = add i32 %4, %5
  ret i32 %add
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
