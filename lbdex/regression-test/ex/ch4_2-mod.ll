; RUN: llc -march=cpu0 -mcpu=cpu032I < %s | FileCheck %s

; ModuleID = 'ch4_2.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: nounwind
define i32 @_Z8test_modv() #0 {
  %b = alloca i32, align 4
; CHECK:  addiu $[[T7:[0-9]+|t9]], $zero, 11
; CHECK:  lui $[[T0:[0-9]+|t9]], 10922
; CHECK:  ori $[[T1:[0-9]+|t9]], $[[T0]], 43691
; CHECK:  addiu $[[T2:[0-9]+|t9]], $zero, 12
; CHECK:  mult  $[[T2]], $[[T1]]
; CHECK:  mfhi  $[[T3:[0-9]+|t9]]
; CHECK:  shr $[[T4:[0-9]+|t9]], $[[T3]], 31
; CHECK:  sra $[[T5:[0-9]+|t9]], $[[T3]], 1
; CHECK:  addu  $[[T6:[0-9]+|t9]], $[[T5]], $[[T4]]
; CHECK:  mul
; CHECK:  subu 
  store i32 11, i32* %b, align 4
  %1 = load i32* %b, align 4
  %2 = add nsw i32 %1, 1
  %3 = srem i32 %2, 12
  store i32 %3, i32* %b, align 4
  %4 = load i32* %b, align 4
  ret i32 %4
}

attributes #0 = { nounwind }
