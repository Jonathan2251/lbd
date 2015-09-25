; RUN: llc  -march=cpu0 -mcpu=cpu032I -relocation-model=pic %s -o - | FileCheck %s -check-prefix=cpu032I
; RUN: llc  -march=cpu0 -mcpu=cpu032II -relocation-model=pic %s -o - | FileCheck %s -check-prefix=cpu032II

; ModuleID = 'ch4_5.bc'
target triple = "mips-unknown-linux-gnu"

define i32 @_Z16test_andorxornotv() nounwind {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i32, align 4
  store i32 5, i32* %a, align 4
  store i32 3, i32* %b, align 4
  store i32 0, i32* %c, align 4
  store i32 0, i32* %d, align 4
  store i32 0, i32* %e, align 4
  %0 = load i32* %a, align 4
  %1 = load i32* %b, align 4
; cpu032I:  and	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  and	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %and = and i32 %0, %1
  store i32 %and, i32* %c, align 4
  %2 = load i32* %a, align 4
  %3 = load i32* %b, align 4
; cpu032I:  or	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  or	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %or = or i32 %2, %3
  store i32 %or, i32* %d, align 4
  %4 = load i32* %a, align 4
  %5 = load i32* %b, align 4
; cpu032I:  xor	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  xor	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %xor = xor i32 %4, %5
  store i32 %xor, i32* %e, align 4
  %6 = load i32* %a, align 4
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T0:[0-9]+|t9]], $sw, 2
; cpu032I:  shr	${{[0-9]+|t9}}, $[[T0]], 1
; cpu032II:  xor	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  sltiu	${{[0-9]+|t9}}, $[[T0]], 1
  %tobool = icmp ne i32 %6, 0
  %lnot = xor i1 %tobool, true
  %conv = zext i1 %lnot to i32
  store i32 %conv, i32* %b, align 4
  %7 = load i32* %c, align 4
  %8 = load i32* %d, align 4
  %add = add nsw i32 %7, %8
  %9 = load i32* %e, align 4
  %add1 = add nsw i32 %add, %9
  %10 = load i32* %b, align 4
  %add2 = add nsw i32 %add1, %10
  ret i32 %add2
}

