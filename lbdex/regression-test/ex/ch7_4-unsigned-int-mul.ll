; RUN: llc  -march=cpu0 -mcpu=cpu032I -relocation-model=pic %s -o - | FileCheck %s
; RUN: llc  -march=cpu0 -mcpu=cpu032II -relocation-model=pic %s -o - | FileCheck %s

; ModuleID = 'ch7_4-unsigned-int-mul.bc'

; Function Attrs: nounwind
define i64 @_Z13test_longlongv() {
entry:
  %a1 = alloca i32, align 4
  %b1 = alloca i32, align 4
  %f = alloca i64, align 8
  store i32 50335744, i32* %a1, align 4
  store i32 33558528, i32* %b1, align 4
  %0 = load i32* %a1, align 4
  %conv = zext i32 %0 to i64
  %1 = load i32* %b1, align 4
  %conv1 = zext i32 %1 to i64
  %mul = mul nsw i64 %conv, %conv1
  store i64 %mul, i64* %f, align 8
; CHECK:  multu	${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; CHECK:  mflo	${{[0-9]+|t9}}
; CHECK:  mfhi	${{[0-9]+|t9}}
  %2 = load i64* %f, align 8
  ret i64 %2
}

