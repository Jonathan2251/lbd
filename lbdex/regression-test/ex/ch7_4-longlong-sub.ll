; RUN: llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic %s -o - | FileCheck %s -check-prefix=cpu032I
; RUN: llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic %s -o - | FileCheck %s -check-prefix=cpu032II

; ModuleID = 'ch7_4-longlong-sub.bc'

; Function Attrs: nounwind
define i64 @_Z13test_longlongv() #0 {
entry:
  %a = alloca i64, align 8
  %b = alloca i64, align 8
  %c = alloca i64, align 8
  store i64 12884901890, i64* %a, align 8
  store i64 4294967297, i64* %b, align 8
  %0 = load i64* %a, align 8
  %1 = load i64* %b, align 8
  %sub = sub nsw i64 %0, %1
  store i64 %sub, i64* %c, align 8
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 1
; cpu032I:  addu	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  addu	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}

; cpu032II: sltu ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II: addu ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II: addu ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}

  %2 = load i64* %c, align 8
  ret i64 %2
}

