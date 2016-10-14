; RUN: llc < %s -march=cpu0el -relocation-model=pic | FileCheck %s 

@g0 = external global i32
@g1 = external global i32
@g2 = external global i32

define void @foo1() nounwind {
entry:
; CHECK:    .cpload
; CHECK:    .cprestore

  tail call void @foo2(i32* @g0) nounwind
  tail call void @foo2(i32* @g1) nounwind
  tail call void @foo2(i32* @g2) nounwind
  ret void
}

declare void @foo2(i32*)
