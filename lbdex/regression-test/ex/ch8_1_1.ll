; RUN: llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic < %s | FileCheck %s -check-prefix=cpu032I
; RUN: llc -march=cpu0 -mcpu=cpu032II -relocation-model=pic < %s | FileCheck %s -check-prefix=cpu032II

; ModuleID = 'ch8_1_1.bc'

; Function Attrs: nounwind
define i32 @_Z13test_control1v() {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i32, align 4
  %f = alloca i32, align 4
  %g = alloca i32, align 4
  %h = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 0, i32* %a, align 4
  store i32 1, i32* %b, align 4
  store i32 2, i32* %c, align 4
  store i32 3, i32* %d, align 4
  store i32 4, i32* %e, align 4
  store i32 5, i32* %f, align 4
  store i32 6, i32* %g, align 4
  store i32 7, i32* %h, align 4
  store i32 8, i32* %i, align 4
  store i32 9, i32* %j, align 4
  %0 = load i32* %a, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jne $sw
; cpu032II:  bne ${{[0-9]+|t9}}, $zero

if.then:                                          ; preds = %entry
  %1 = load i32* %a, align 4
  %inc = add i32 %1, 1
  store i32 %inc, i32* %a, align 4
  br label %if.end

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jeq $sw
; cpu032II:  beq ${{[0-9]+|t9}}, $zero

if.end:                                           ; preds = %if.then, %entry
  %2 = load i32* %b, align 4
  %cmp1 = icmp ne i32 %2, 0
  br i1 %cmp1, label %if.then2, label %if.end4

if.then2:                                         ; preds = %if.end
  %3 = load i32* %b, align 4
  %inc3 = add nsw i32 %3, 1
  store i32 %inc3, i32* %b, align 4
  br label %if.end4

if.end4:                                          ; preds = %if.then2, %if.end
  %4 = load i32* %c, align 4
  %cmp5 = icmp sgt i32 %4, 0
  br i1 %cmp5, label %if.then6, label %if.end8

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jlt $sw
; cpu032II:  slt $[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  bne $[[T0]], $zero

if.then6:                                         ; preds = %if.end4
  %5 = load i32* %c, align 4
  %inc7 = add nsw i32 %5, 1
  store i32 %inc7, i32* %c, align 4
  br label %if.end8

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jlt $sw
; cpu032II:  slt $[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  bne $[[T0]], $zero

if.end8:                                          ; preds = %if.then6, %if.end4
  %6 = load i32* %d, align 4
  %cmp9 = icmp sge i32 %6, 0
  br i1 %cmp9, label %if.then10, label %if.end12

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jgt $sw
; cpu032II:  slt $[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  bne $[[T0]], $zero

if.then10:                                        ; preds = %if.end8
  %7 = load i32* %d, align 4
  %inc11 = add nsw i32 %7, 1
  store i32 %inc11, i32* %d, align 4
  br label %if.end12

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jgt $sw
; cpu032II:  slt	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  bne $[[T0]], $zero

if.end12:                                         ; preds = %if.then10, %if.end8
  %8 = load i32* %e, align 4
  %cmp13 = icmp slt i32 %8, 0
  br i1 %cmp13, label %if.then14, label %if.end16

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jgt $sw
; cpu032II:  slt	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  bne $[[T0]], $zero

if.then14:                                        ; preds = %if.end12
  %9 = load i32* %e, align 4
  %inc15 = add nsw i32 %9, 1
  store i32 %inc15, i32* %e, align 4
  br label %if.end16

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jlt $sw
; cpu032II:  slt $[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  bne $[[T0]], $zero

if.end16:                                         ; preds = %if.then14, %if.end12
  %10 = load i32* %f, align 4
  %cmp17 = icmp sle i32 %10, 0
  br i1 %cmp17, label %if.then18, label %if.end20

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jge $sw
; cpu032II:  slt	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  beq $[[T0]], $zero

if.then18:                                        ; preds = %if.end16
  %11 = load i32* %f, align 4
  %inc19 = add nsw i32 %11, 1
  store i32 %inc19, i32* %f, align 4
  br label %if.end20

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  jeq $sw
; cpu032II:  beq ${{[0-9]+|t9}}, ${{[0-9]+|t9}}

if.end20:                                         ; preds = %if.then18, %if.end16
  %12 = load i32* %g, align 4
  %cmp21 = icmp sle i32 %12, 1
  br i1 %cmp21, label %if.then22, label %if.end24

if.then22:                                        ; preds = %if.end20
  %13 = load i32* %g, align 4
  %inc23 = add nsw i32 %13, 1
  store i32 %inc23, i32* %g, align 4
  br label %if.end24

if.end24:                                         ; preds = %if.then22, %if.end20
  %14 = load i32* %h, align 4
  %cmp25 = icmp sge i32 %14, 1
  br i1 %cmp25, label %if.then26, label %if.end28

if.then26:                                        ; preds = %if.end24
  %15 = load i32* %h, align 4
  %inc27 = add nsw i32 %15, 1
  store i32 %inc27, i32* %h, align 4
  br label %if.end28

if.end28:                                         ; preds = %if.then26, %if.end24
  %16 = load i32* %i, align 4
  %17 = load i32* %h, align 4
  %cmp29 = icmp slt i32 %16, %17
  br i1 %cmp29, label %if.then30, label %if.end32

if.then30:                                        ; preds = %if.end28
  %18 = load i32* %i, align 4
  %inc31 = add nsw i32 %18, 1
  store i32 %inc31, i32* %i, align 4
  br label %if.end32

if.end32:                                         ; preds = %if.then30, %if.end28
  %19 = load i32* %a, align 4
  %20 = load i32* %b, align 4
  %cmp33 = icmp ne i32 %19, %20
  br i1 %cmp33, label %if.then34, label %if.end36

if.then34:                                        ; preds = %if.end32
  %21 = load i32* %j, align 4
  %inc35 = add nsw i32 %21, 1
  store i32 %inc35, i32* %j, align 4
  br label %if.end36

if.end36:                                         ; preds = %if.then34, %if.end32
  %22 = load i32* %a, align 4
  %23 = load i32* %b, align 4
  %add = add i32 %22, %23
  %24 = load i32* %c, align 4
  %add37 = add i32 %add, %24
  %25 = load i32* %d, align 4
  %add38 = add i32 %add37, %25
  %26 = load i32* %e, align 4
  %add39 = add i32 %add38, %26
  %27 = load i32* %f, align 4
  %add40 = add i32 %add39, %27
  %28 = load i32* %g, align 4
  %add41 = add i32 %add40, %28
  %29 = load i32* %h, align 4
  %add42 = add i32 %add41, %29
  %30 = load i32* %i, align 4
  %add43 = add i32 %add42, %30
  %31 = load i32* %j, align 4
  %add44 = add i32 %add43, %31
  ret i32 %add44
}

