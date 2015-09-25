; RUN: llc -march=cpu0el < %s | FileCheck %s

; ModuleID = 'ch4_1.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: nounwind
define i32 @_Z9test_mathv() #0 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %a1 = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i32, align 4
  %f = alloca i32, align 4
  %g = alloca i32, align 4
  %h = alloca i32, align 4
  %i = alloca i32, align 4
  %f1 = alloca i32, align 4
  %g1 = alloca i32, align 4
  %h1 = alloca i32, align 4
  %i1 = alloca i32, align 4
  store i32 5, i32* %a, align 4
  store i32 2, i32* %b, align 4
  store i32 -5, i32* %a1, align 4
  %0 = load i32* %a, align 4
  %1 = load i32* %b, align 4
; CHECK:  addu  ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %add = add nsw i32 %0, %1
  store i32 %add, i32* %c, align 4
  %2 = load i32* %a, align 4
  %3 = load i32* %b, align 4
; CHECK:  subu  ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %sub = sub nsw i32 %2, %3
  store i32 %sub, i32* %d, align 4
  %4 = load i32* %a, align 4
  %5 = load i32* %b, align 4
; CHECK:  mul ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %mul = mul nsw i32 %4, %5
  store i32 %mul, i32* %e, align 4
  %6 = load i32* %a, align 4
; CHECK:  shl ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, 2
  %shl = shl i32 %6, 2
  store i32 %shl, i32* %f, align 4
  %7 = load i32* %a1, align 4
; CHECK:  shl ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, 1
  %shl1 = shl i32 %7, 1
  store i32 %shl1, i32* %f1, align 4
  %8 = load i32* %a, align 4
; CHECK:  sra ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, 2
  %shr = ashr i32 %8, 2
  store i32 %shr, i32* %g, align 4
  %9 = load i32* %a1, align 4
; CHECK:  shr ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, 30
  %shr2 = lshr i32 %9, 30
  store i32 %shr2, i32* %g1, align 4
  %10 = load i32* %a, align 4
; CHECK:  addiu $[[T0:[0-9]+|t9]], $zero, 1
; CHECK:  shlv  ${{[0-9]+|t9}}, $[[T0]], ${{[0-9]+|t9}}
  %shl3 = shl i32 1, %10
  store i32 %shl3, i32* %h, align 4
  %11 = load i32* %b, align 4
; CHECK:  shlv  ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %shl4 = shl i32 1, %11
  store i32 %shl4, i32* %h1, align 4
  %12 = load i32* %a, align 4
; CHECK:  shrv  ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %shr5 = ashr i32 128, %12
  store i32 %shr5, i32* %i, align 4
  %13 = load i32* %b, align 4
  %14 = load i32* %a, align 4
; CHECK:  srav  ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %shr6 = ashr i32 %13, %14
  store i32 %shr6, i32* %i1, align 4
  %15 = load i32* %c, align 4
  %16 = load i32* %d, align 4
  %add7 = add nsw i32 %15, %16
  %17 = load i32* %e, align 4
  %add8 = add nsw i32 %add7, %17
  %18 = load i32* %f, align 4
  %add9 = add nsw i32 %add8, %18
  %19 = load i32* %f1, align 4
  %add10 = add nsw i32 %add9, %19
  %20 = load i32* %g, align 4
  %add11 = add nsw i32 %add10, %20
  %21 = load i32* %g1, align 4
  %add12 = add nsw i32 %add11, %21
  %22 = load i32* %h, align 4
  %add13 = add nsw i32 %add12, %22
  %23 = load i32* %h1, align 4
  %add14 = add nsw i32 %add13, %23
  %24 = load i32* %i, align 4
  %add15 = add nsw i32 %add14, %24
  %25 = load i32* %i1, align 4
  %add16 = add nsw i32 %add15, %25
  ret i32 %add16
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
