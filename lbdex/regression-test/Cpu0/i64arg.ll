; RUN: llc -march=cpu0 -relocation-model=pic < %s | FileCheck %s

define void @f1(i64 %ll1, float %f, i64 %ll, i32 %i, float %f2) nounwind {
entry:
; CHECK: lui $[[R0:[0-9]+|t9]], 4660
; CHECK: ori ${{[0-9]+|t9}}, $[[R0]], 22136
; CHECK: lui $[[R0:[0-9]+|t9]], 3855
; CHECK: ori ${{[0-9]+|t9}}, $[[R0]], 3855
; CHECK: ld  $t9, %call16(ff1)($gp)
; CHECK: jalr $t9
  tail call void @ff1(i32 %i, i64 1085102592623924856) nounwind
; CHECK: ld $t9, %call16(ff2)($gp)
; CHECK: jalr $t9
  tail call void @ff2(i64 %ll, double 3.000000e+00) nounwind
  %sub = add nsw i32 %i, -1
; CHECK: ld $t9, %call16(ff3)($gp)
; CHECK: jalr $t9
  tail call void @ff3(i32 %i, i64 %ll, i32 %sub, i64 %ll1) nounwind
  ret void
}

declare void @ff1(i32, i64)

declare void @ff2(i64, double)

declare void @ff3(i32, i64, i32, i64)
