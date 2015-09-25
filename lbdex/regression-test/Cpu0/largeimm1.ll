; RUN: llc -march=cpu0 -relocation-model=static -filetype=asm < %s | FileCheck %s
; RUN: llc -march=cpu0 -relocation-model=pic -filetype=asm < %s | FileCheck %s

; CHECK: lui $[[T0:[0-9]+|t9]], 49152
; CHECK:  addiu	${{[0-9]+|t9}}, $[[T0]], -{{[0-9]+|t9}}
define void @f() nounwind {
entry:
  %a1 = alloca [1073741824 x i8], align 1
  %arrayidx = getelementptr inbounds [1073741824 x i8], [1073741824 x i8]* %a1, i32 0, i32 1048676
  call void @f2(i8* %arrayidx) nounwind
  ret void
}

declare void @f2(i8*)
