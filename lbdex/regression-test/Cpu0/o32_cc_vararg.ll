; RUN: llc -march=cpu0 -mcpu=cpu032II -cpu0-s32-calls=false < %s | FileCheck %s

; All test functions do the same thing - they return the first variable
; argument.

; All CHECK's do the same thing - they check whether variable arguments from
; registers are placed on correct stack locations, and whether the first
; variable argument is returned from the correct stack location.


declare void @llvm.va_start(i8*) nounwind
declare void @llvm.va_end(i8*) nounwind

; return int
define i32 @va1(i32 %a, ...) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %ap = alloca i8*, align 4
  %b = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, i32
  store i32 %0, i32* %b, align 4
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load i32, i32* %b, align 4
  ret i32 %tmp

; CHECK: va1:
; CHECK: addiu   $sp, $sp, -16
; CHECK: addiu	 $[[R0:[0-9]+|t9]], $sp, 20
; CHECK: addiu	 $[[R1:[0-9]+|t9]], $[[R0:[0-9]+|t9]], 4
; CHECK: st      $[[R1]], 8($sp)
; CHECK: st      $5, 20($sp)
; CHECK: st      $5, 4($sp)
}

; check whether the variable double argument will be accessed from the 8-byte
; aligned location (i.e. whether the address is computed by adding 7 and
; clearing lower 3 bits)
define double @va2(i32 %a, ...) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %ap = alloca i8*, align 4
  %b = alloca double, align 8
  store i32 %a, i32* %a.addr, align 4
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, double
  store double %0, double* %b, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load double, double* %b, align 8
  ret double %tmp

; CHECK: va2:
; CHECK: addiu   $sp, $sp, -16
; CHECK: st      $4, 12($sp)
; CHECK: st      $5, 20($sp)
; CHECK: addiu   $[[R0:[0-9]+|t9]], $sp, 20
; CHECK: addiu   $[[R1:[0-9]+|t9]], $[[R0]], 7
; CHECK: addiu   $[[R2:[0-9]+|t9]], $zero, -8
; CHECK: and     $[[R3:[0-9]+|t9]], $[[R1]], $[[R2]]
}

; int
define i32 @va3(double %a, ...) nounwind {
entry:
  %a.addr = alloca double, align 8
  %ap = alloca i8*, align 4
  %b = alloca i32, align 4
  store double %a, double* %a.addr, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, i32
  store i32 %0, i32* %b, align 4
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load i32, i32* %b, align 4
  ret i32 %tmp

; CHECK: va3:
; CHECK: addiu   $sp, $sp, -16
; CHECK: st      $5, 12($sp)
; CHECK: st      $4, 8($sp)
; CHECK: ld      $2, 24($sp)
}

; double
define double @va4(double %a, ...) nounwind {
entry:
  %a.addr = alloca double, align 8
  %ap = alloca i8*, align 4
  %b = alloca double, align 8
  store double %a, double* %a.addr, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, double
  store double %0, double* %b, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load double, double* %b, align 8
  ret double %tmp

; CHECK: va4:
; CHECK: addiu   $sp, $sp, -24
; CHECK: st      $5, 20($sp)
; CHECK: st      $4, 16($sp)
; CHECK: addiu   ${{[0-9]+|t9}}, $sp, 32
}

