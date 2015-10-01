; RUN: llc -march=cpu0el -mcpu=cpu032II < %s | FileCheck %s

declare void @llvm.eh.return.i32(i32, i8*)
declare void @foo(...)

define i8* @f1(i32 %offset, i8* %handler) {
entry:
  call void (...) @foo()
  call void @llvm.eh.return.i32(i32 %offset, i8* %handler)
  unreachable

; CHECK:        f1
; CHECK:        addiu   $sp, $sp, -[[spoffset:[0-9]+|t9]]

; check that $a0-$a1 are saved on stack.
; CHECK:        st      $4, [[offset0:[0-9]+|t9]]($sp)
; CHECK:        st      $5, [[offset1:[0-9]+|t9]]($sp)

; check that .cfi_offset directives are emitted for $a0-$a1.
; CHECK:        .cfi_offset 4,
; CHECK:        .cfi_offset 5,

; check that stack adjustment and handler are put in $v1 and $v0.
; CHECK:        addu      $[[R0:[a-z0-9]+]], $zero, $5
; CHECK:        addu      $[[R1:[a-z0-9]+]], $zero, $4
; CHECK:        addu      $3, $zero, $[[R1]]
; CHECK:        addu      $2, $zero, $[[R0]]

; check that $a0-$a1 are restored from stack.
; CHECK:        ld      $4, [[offset0]]($sp)
; CHECK:        ld      $5, [[offset1]]($sp)

; check that stack is adjusted by $v1 and that code returns to address in $v0
; CHECK:        addiu   $sp, $sp, [[spoffset]]
; CHECK:        move    $lr, $2
; CHECK:        addu    $sp, $sp, $3
; CHECK:        ret     $lr
}

define i8* @f2(i32 %offset, i8* %handler) {
entry:
  call void @llvm.eh.return.i32(i32 %offset, i8* %handler)
  unreachable

; CHECK:        f2
; CHECK:        addiu   $sp, $sp, -[[spoffset:[0-9]+|t9]]

; check that $a0-$a1 are saved on stack.
; CHECK:        st      $4, [[offset0:[0-9]+|t9]]($sp)
; CHECK:        st      $5, [[offset1:[0-9]+|t9]]($sp)

; check that .cfi_offset directives are emitted for $a0-$a1.
; CHECK:        .cfi_offset 4,
; CHECK:        .cfi_offset 5,

; check that stack adjustment and handler are put in $v1 and $v0.
; CHECK:        addu    $3, $zero, $4
; CHECK:        addu    $2, $zero, $5

; check that $a0-$a1 are restored from stack.
; CHECK:        ld      $4, [[offset0]]($sp)
; CHECK:        ld      $5, [[offset1]]($sp)

; check that stack is adjusted by $v1 and that code returns to address in $v0
; CHECK:        addiu   $sp, $sp, [[spoffset]]
; CHECK:        move    $lr, $2
; CHECK:        addu    $sp, $sp, $3
; CHECK:        ret     $lr
}
