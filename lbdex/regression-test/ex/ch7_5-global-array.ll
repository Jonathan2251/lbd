; RUN: llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=false -filetype=asm < %s | FileCheck %s -check-prefix=STATIC
; RUN: llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=true -filetype=asm < %s | FileCheck %s -check-prefix=STATIC
; RUN: llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=false -filetype=asm < %s | FileCheck %s -check-prefix=PIC
; RUN: llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=true -filetype=asm < %s | FileCheck %s -check-prefix=PIC

; ModuleID = 'ch7_5-global-array.bc'

@a = global [3 x i32] [i32 2012, i32 10, i32 12], align 4

; Function Attrs: nounwind
define i32 @_Z11test_structv() {
entry:
  %i = alloca i32, align 4
  %0 = load i32* getelementptr inbounds ([3 x i32]* @a, i32 0, i32 1), align 4
  store i32 %0, i32* %i, align 4
  %1 = load i32* %i, align 4
; STATIC:  lui	$[[T0:[0-9]+|t9]], %hi(a)
; STATIC:  ori	$[[T1:[0-9]+|t9]], $[[T0]], %lo(a)
; STATIC:  ld	${{[0-9]+|t9}}, 4($[[T1]])

; PIC:  .cpload	$t9
; PIC:  lui	$[[T0:[0-9]+|t9]], %got_hi(a)
; PIC:  addu	$[[T1:[0-9]+|t9]], $[[T0]], $gp
; PIC:  ld	$[[T2:[0-9]+|t9]], %got_lo(a)($[[T1]])
; PIC:  ld	${{[0-9]+|t9}}, 4($[[T2]])

  ret i32 %1
}

