; RUN: llc -march=cpu0 -mcpu=cpu032I < %s | FileCheck %s

; ModuleID = 'ch4_2_2.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: nounwind
define i32 @_Z8test_modi(i32 %c) #0 {
entry:
  %c.addr = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
; CHECK:  addiu	$[[T0:[0-9]+|t9]], $zero, 11
  store i32 11, i32* %b, align 4
  %0 = load i32* %b, align 4
; CHECK:  addiu	$[[T0:[0-9]+|t9]], $zero, 12
  %add = add nsw i32 %0, 1
  %1 = load i32* %c.addr, align 4
; CHECK:  div	$[[T0]], ${{[0-9]+|t9}}
; CHECK:  mfhi	${{[0-9]+|t9}}
  %rem = srem i32 %add, %1
  store i32 %rem, i32* %b, align 4
  %2 = load i32* %b, align 4
  ret i32 %2
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
