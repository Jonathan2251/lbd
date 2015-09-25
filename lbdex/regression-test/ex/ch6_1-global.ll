; RUN: llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=false -filetype=asm < %s | FileCheck %s -check-prefix=STATIC_LARGE
; RUN: llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=true -filetype=asm < %s | FileCheck %s -check-prefix=STATIC_SMALL
; RUN: llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=false -filetype=asm < %s | FileCheck %s -check-prefix=PIC_LARGE
; RUN: llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=true -filetype=asm < %s | FileCheck %s -check-prefix=PIC_SMALL

; ModuleID = 'ch6_1.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

@gStart = global i32 3, align 4
@gI = global i32 100, align 4

; Function Attrs: nounwind
define i32 @_Z3funv() #0 {
entry:
  %c = alloca i32, align 4
  store i32 0, i32* %c, align 4
  %0 = load i32* @gI, align 4
; STATIC_LARGE:  lui	$[[T0:[0-9]+|t9]], %hi(gI)
; STATIC_LARGE:  ori	$[[T1:[0-9]+|t9]], $[[T0]], %lo(gI)
; STATIC_LARGE:  ld	${{[0-9]+|t9}}, 0($[[T1]])
; STATIC_LARGE:  .data
; STATIC_SMALL:  ori	$[[T0:[0-9]+|t9]], $gp, %gp_rel(gI)
; STATIC_SMALL:  ld	${{[0-9]+|t9}}, 0($[[T0]])
; STATIC_SMALL:  .sdata
; PIC_LARGE:  .cpload	$t9
; PIC_LARGE:  lui	$[[T0:[0-9]+|t9]], %got_hi(gI)
; PIC_LARGE:  addu	$[[T1:[0-9]+|t9]], $[[T0]], $gp
; PIC_LARGE:  ld	${{[0-9]+|t9}}, %got_lo(gI)($[[T1]])
; PIC_LARGE:  .data
; PIC_SMALL:  .cpload	$t9
; PIC_SMALL:  ld	$[[T0:[0-9]+|t9]], %got(gI)($gp)
; PIC_SMALL:  ld	${{[0-9]+|t9}}, 0($[[T0]])
; PIC_SMALL:  .sdata
  store i32 %0, i32* %c, align 4
  %1 = load i32* %c, align 4
  ret i32 %1
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
