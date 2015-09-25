; RUN: llc -march=cpu0 -relocation-model=pic -filetype=asm < %s | FileCheck %s

; ModuleID = 'ch7_1.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: nounwind
define i32 @_Z18test_local_pointerv() #0 {
entry:
  %b = alloca i32, align 4
  %p = alloca i32*, align 4
  store i32 3, i32* %b, align 4
  store i32* %b, i32** %p, align 4
  %0 = load i32** %p, align 4
  %1 = load i32* %0, align 4
  ret i32 %1
; CHECK:  addiu	$[[T0:[0-9]+|t9]], $zero, 3
; CHECK:  st	$[[T0]], 8($fp)
; CHECK:  addiu	$[[T1:[0-9]+|t9]], $fp, 8
; CHECK:  st	$[[T1]], 4($fp)
; CHECK:  ld	$2, 8($fp)
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
