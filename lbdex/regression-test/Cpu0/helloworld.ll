; RUN: llc  -march=cpu0el -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC
; RUN: llc  -march=cpu0el -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC
; check got .str as well as save/restore $lr

@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0))
  ret i32 0

; PIC: .ent main
; PIC: .set noreorder
; PIC: .cpload $t9
; PIC: .set nomacro
; PIC: st $lr, [[FS:[0-9]+|t9]]($sp)
; PIC: .cprestore 8
; PIC: ld	$[[R0:[0-9]+|t9]], %got($.str)($gp)
; PIC: ori	${{[0-9]+|t9}}, $[[R0:[0-9]+|t9]], %lo($.str)
; PIC: ld	$t9, %call16(printf)($gp)
; PIC: jalr $t9
; PIC: ld $lr, [[FS]]($sp)
; STATIC: .ent main
; STATIC: .set noreorder
; STATIC: .set nomacro
; STATIC: st $lr, [[FS:[0-9]+|t9]]($sp)
; STATIC: lui	$[[R0:[0-9]+|t9]], %hi($.str)
; STATIC: ori	${{[0-9]+|t9}}, $[[R0:[0-9]+|t9]], %lo($.str)
; STATIC: jsub printf
; STATIC: ld $lr, [[FS]]($sp)
}

; PIC: .set macro
; PIC: .set reorder
; PIC: .end main
; STATIC: .set macro
; STATIC: .set reorder
; STATIC: .end main
declare i32 @printf(i8*, ...)
