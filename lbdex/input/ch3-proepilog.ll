; RUN: llc -march=cpu0el -mcpu=cpu032I < %s | FileCheck %s

; stack align 8, so 32759 is 32760
; 32760 = 0x7ff8
; 2147483640 = 0x7ffffff8
; 2415951872 = 0x90008000

define i32 @e() #0 {
entry:
; Prolog
; CHECK:	addiu	$sp, $sp, -32760
; CHECK:	.cfi_def_cfa_offset 32760

; Epilog
; CHECK:	addiu	$sp, $sp, 32760

  %retval = alloca i8, i32 32759, align 4
  ret i32 0
}

define i32 @f() #0 {
entry:
; Prolog
; CHECK:	addiu	$sp, $sp, -32768
; CHECK:	.cfi_def_cfa_offset 32768

; Epilog
; CHECK:	ori	$[[T0:[0-9]+|t9]], $zero, 32768
; CHECK:	addu	$sp, $sp, $[[T0]]

  %retval = alloca i8, i32 32768, align 4
  ret i32 0
}

define i32 @g() #0 {
entry:
; Prolog
; CHECK:	lui	$[[T0:[0-9]+|t9]], 32768
; CHECK:	addiu	$[[T1:[0-9]+|t9]], $[[T0]], 8
; CHECK:	addu	$sp, $sp, $[[T1]]
; CHECK:	.cfi_def_cfa_offset 2147483640

; Epilog
; CHECK:	lui	$[[T2:[0-9]+|t9]], 32767
; CHECK:	ori	$[[T3:[0-9]+|t9]], $[[T2]], 65528
; CHECK:	addu	$sp, $sp, $[[T3]]

  %retval = alloca i8, i32 2147483635, align 4
  ret i32 0
}

define i32 @h() #0 {
entry:
; Prolog
; CHECK:	lui	$[[T0:[0-9]+|t9]], 28671
; CHECK:	ori	$[[T1:[0-9]+|t9]], $[[T0]], 32768
; CHECK:	addu	$sp, $sp, $[[T1]]
; CHECK:	.cfi_def_cfa_offset -1879015424

; Epilog
; CHECK:	lui	$[[T2:[0-9]+|t9]], 36865
; CHECK:	addiu	$[[T3:[0-9]+|t9]], $[[T2]], -32768
; CHECK:	addu	$sp, $sp, $[[T3]]

  %retval = alloca i8, i32 2415951872, align 4
  ret i32 0
}

