; RUN: llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic -cpu0-use-small-section=true -filetype=asm < %s | FileCheck %s

; CHECK: ld	${{[0-9]+|t9}}, %got(s0)($gp)
; CHECK: ld	${{[0-9]+|t9}}, %got(foo)($gp)
; CHECK: .sdata

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-unknown-psp-elf"

  %struct.anon = type { i32, i32 }
@s0 = global [8 x i8] c"AAAAAAA\00", align 4
@foo = global %struct.anon { i32 2, i32 3 }
@bar = global %struct.anon zeroinitializer 

define i8* @A0() nounwind {
entry:
	ret i8* getelementptr ([8 x i8], [8 x i8]* @s0, i32 0, i32 0)
}

define i32 @A1() nounwind {
entry:
  load i32, i32* getelementptr (%struct.anon, %struct.anon* @foo, i32 0, i32 0), align 8 
  load i32, i32* getelementptr (%struct.anon, %struct.anon* @foo, i32 0, i32 1), align 4 
  add i32 %1, %0
  ret i32 %2
}

