; RUN: llc  -march=cpu0 -mcpu=cpu032I -relocation-model=pic %s -o - | FileCheck %s -check-prefix=cpu032I
; RUN: llc  -march=cpu0 -mcpu=cpu032II -relocation-model=pic %s -o - | FileCheck %s -check-prefix=cpu032II

; ModuleID = 'ch4_5.bc'
target triple = "mips-unknown-linux-gnu"

define i32 @_Z10test_setxxv() nounwind {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i32, align 4
  %f = alloca i32, align 4
  %g = alloca i32, align 4
  %h = alloca i32, align 4
  store i32 5, i32* %a, align 4
  store i32 3, i32* %b, align 4
  %0 = load i32* %a, align 4
  %1 = load i32* %b, align 4
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 2
; cpu032I:  shr	${{[0-9]+|t9}}, $[[T1]], 1
; cpu032II:  xor	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  sltiu	${{[0-9]+|t9}}, $[[T0]], 1
  %cmp = icmp eq i32 %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* %c, align 4
  %2 = load i32* %a, align 4
  %3 = load i32* %b, align 4
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 2
; cpu032I:  shr	$[[T2:[0-9]+|t9]], $[[T1]], 1
; cpu032I:  xori	${{[0-9]+|t9}}, $[[T2]], 1
; cpu032II:  xor	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  sltu	${{[0-9]+|t9}}, $zero, $[[T0]]
  %cmp1 = icmp ne i32 %2, %3
  %conv2 = zext i1 %cmp1 to i32
  store i32 %conv2, i32* %d, align 4
  %4 = load i32* %a, align 4
  %5 = load i32* %b, align 4
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 1
; cpu032II:  slt	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %cmp3 = icmp slt i32 %4, %5
  %conv4 = zext i1 %cmp3 to i32
  store i32 %conv4, i32* %e, align 4
  %6 = load i32* %a, align 4
  %7 = load i32* %b, align 4
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 1
; cpu032I:  xori	${{[0-9]+|t9}}, $[[T1]], 1
; cpu032II:  slt	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  xori	${{[0-9]+|t9}}, $[[T0]], 1
  %cmp5 = icmp sle i32 %6, %7
  %conv6 = zext i1 %cmp5 to i32
  store i32 %conv6, i32* %f, align 4
  %8 = load i32* %a, align 4
  %9 = load i32* %b, align 4
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 1
; cpu032II:  slt	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %cmp7 = icmp sgt i32 %8, %9
  %conv8 = zext i1 %cmp7 to i32
  store i32 %conv8, i32* %g, align 4
  %10 = load i32* %a, align 4
  %11 = load i32* %b, align 4
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 1
; cpu032I:  xori	${{[0-9]+|t9}}, $[[T1]], 1
; cpu032II:  slt	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  xori	${{[0-9]+|t9}}, $[[T0]], 1
  %cmp9 = icmp sge i32 %10, %11
  %conv10 = zext i1 %cmp9 to i32
  store i32 %conv10, i32* %h, align 4
  %12 = load i32* %c, align 4
  %13 = load i32* %d, align 4
  %add = add nsw i32 %12, %13
  %14 = load i32* %e, align 4
  %add11 = add nsw i32 %add, %14
  %15 = load i32* %f, align 4
  %add12 = add nsw i32 %add11, %15
  %16 = load i32* %g, align 4
  %add13 = add nsw i32 %add12, %16
  %17 = load i32* %h, align 4
  %add14 = add nsw i32 %add13, %17
  ret i32 %add14
}

