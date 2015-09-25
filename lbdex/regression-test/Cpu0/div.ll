; RUN: llc  -march=cpu0 -relocation-model=pic < %s | FileCheck %s

@iiii = global i32 100, align 4
@jjjj = global i32 -4, align 4
@kkkk = common global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, i32* @iiii, align 4
  %1 = load i32, i32* @jjjj, align 4
  %div = sdiv i32 %0, %1
; CHECK:	div	${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; CHECK: 	mflo	${{[0-9]+|t9}}
  store i32 %div, i32* @kkkk, align 4
  ret void
}


