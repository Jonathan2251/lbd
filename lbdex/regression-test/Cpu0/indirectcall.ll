; RUN: llc  < %s -march=cpu0el -relocation-model=static | FileCheck %s 

define void @foo0(void (i32)* nocapture %f1) nounwind {
entry:
; CHECK: jalr $t9
  tail call void %f1(i32 13) nounwind
  ret void
}
