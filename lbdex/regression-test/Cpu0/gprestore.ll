; RUN: llc -march=cpu0  -relocation-model=pic < %s | FileCheck %s

; check gprestore and PIC function call

@p = external global i32
@q = external global i32
@r = external global i32

define void @f0() nounwind {
entry:
; CHECK: .cprestore [[FS:[0-9]+|t9]]
; CHECK:	ld	$t9, %call16(f1)($gp)
; CHECK: jalr $t9
; CHECK: ld $gp, [[FS]]($sp)
; CHECK-NOT: got({{.*}})($gp)
; CHECK:	ld	$t9, %call16(f2)($gp)
; CHECK: jalr $t9
; CHECK: ld $gp, [[FS]]($sp)
; CHECK-NOT: got({{.*}})($gp)
; CHECK:	ld	$t9, %call16(f3)($gp)
; CHECK: jalr $t9
; CHECK: ld $gp, [[FS]]($sp)
; CHECK-NOT: got({{.*}})($gp)
  tail call void (...) @f1() nounwind
  %tmp = load i32, i32* @p, align 4
  tail call void @f2(i32 %tmp) nounwind
  %tmp1 = load i32, i32* @q, align 4
  %tmp2 = load i32, i32* @r, align 4
  tail call void @f3(i32 %tmp1, i32 %tmp2) nounwind
  ret void
}

declare void @f1(...)

declare void @f2(i32)

declare void @f3(i32, i32)

