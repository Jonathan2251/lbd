; RUN: llc  < %s -march=cpu0el | FileCheck %s

@d2 = external global double
@d3 = external global double

define i32 @sel1(i32 %s, i32 %f0, i32 %f1) nounwind readnone {
entry:
; CHECK: movn
  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, i32 %f1, i32 %f0
  ret i32 %cond
}

define float @sel2(i32 %s, float %f0, float %f1) nounwind readnone {
entry:
; CHECK: movn
  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, float %f0, float %f1
  ret float %cond
}

define double @sel2_1(i32 %s, double %f0, double %f1) nounwind readnone {
entry:
; CHECK: movn
  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, double %f0, double %f1
  ret double %cond
}

define float @sel3(float %f0, float %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK: movz
  %cmp = fcmp oeq float %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define float @sel4(float %f0, float %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK: movn
  %cmp = fcmp olt float %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define float @sel5(float %f0, float %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK: movn
  %cmp = fcmp ogt float %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define double @sel5_1(double %f0, double %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK: movn
  %cmp = fcmp ogt float %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define double @sel6(double %f0, double %f1, double %f2, double %f3) nounwind readnone {
entry:
; CHECK: movz
  %cmp = fcmp oeq double %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define double @sel7(double %f0, double %f1, double %f2, double %f3) nounwind readnone {
entry:
; CHECK: movn
  %cmp = fcmp olt double %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define double @sel8(double %f0, double %f1, double %f2, double %f3) nounwind readnone {
entry:
; CHECK: movn
  %cmp = fcmp ogt double %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define float @sel8_1(float %f0, float %f1, double %f2, double %f3) nounwind readnone {
entry:
; CHECK: movn
  %cmp = fcmp ogt double %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define i32 @sel9(i32 %f0, i32 %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK: movz
  %cmp = fcmp oeq float %f2, %f3
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel10(i32 %f0, i32 %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK: movn
  %cmp = fcmp olt float %f2, %f3
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel11(i32 %f0, i32 %f1, float %f2, float %f3) nounwind readnone {
entry:
; CHECK: movn
  %cmp = fcmp ogt float %f2, %f3
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel12(i32 %f0, i32 %f1) nounwind readonly {
entry:
; CHECK: movz
  %tmp = load double, double* @d2, align 8
  %tmp1 = load double, double* @d3, align 8
  %cmp = fcmp oeq double %tmp, %tmp1
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel13(i32 %f0, i32 %f1) nounwind readonly {
entry:
; CHECK: movn
  %tmp = load double, double* @d2, align 8
  %tmp1 = load double, double* @d3, align 8
  %cmp = fcmp olt double %tmp, %tmp1
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @sel14(i32 %f0, i32 %f1) nounwind readonly {
entry:
; CHECK: movn
  %tmp = load double, double* @d2, align 8
  %tmp1 = load double, double* @d3, align 8
  %cmp = fcmp ogt double %tmp, %tmp1
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

