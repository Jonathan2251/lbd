; RUN: llc  -march=cpu0 -mcpu=cpu032I -relocation-model=pic %s -o - | FileCheck %s
; RUN: llc  -march=cpu0 -mcpu=cpu032II -relocation-model=pic %s -o - | FileCheck %s

; ModuleID = 'ch7_4-unsigned-longlong-mul.bc'

; Function Attrs: nounwind
define i64 @_Z13test_longlongv() #0 {
entry:
  %a = alloca i64, align 8
  %b = alloca i64, align 8
  %e = alloca i64, align 8
  store i64 12884901890, i64* %a, align 8
  store i64 17179869185, i64* %b, align 8
  %0 = load i64* %a, align 8
  %1 = load i64* %b, align 8
  %mul = mul i64 %0, %1
  store i64 %mul, i64* %e, align 8
; CHECK:  multu	${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; CHECK:  mflo	${{[0-9]+|t9}}
; CHECK:  mfhi	${{[0-9]+|t9}}
; CHECK:  shl	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, 2
; CHECK:  addu	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; CHECK:  addu	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}

  %2 = load i64* %e, align 8
  ret i64 %2
}

