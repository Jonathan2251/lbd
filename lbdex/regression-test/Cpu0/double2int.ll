; RUN: llc -march=cpu0 -relocation-model=pic < %s | FileCheck %s

define i32 @f1(double %d) nounwind readnone {
entry:
; CHECK: %call16(__fixdfsi)
  %conv = fptosi double %d to i32
  ret i32 %conv
}
