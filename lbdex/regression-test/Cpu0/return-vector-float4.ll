; RUN: llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic < %s | FileCheck %s

define <4 x float> @retvec4() nounwind readnone {
entry:
; CHECK: lui $2, 16256
; CHECK: lui $3, 16384
; CHECK: lui $4, 16448
; CHECK: $5, 16512

  ret <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>
}

