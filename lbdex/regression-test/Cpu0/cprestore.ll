; RUN: llc -march=cpu0el -relocation-model=pic < %s | FileCheck %s

; CHECK: .set nomacro
; CHECK: .cprestore
; CHECK: .set macro

;%struct.S = type { [16384 x i32] }
%struct.S = type { i32 }

declare void @foo1(%struct.S* byval align 8 %s)

define void @foo2() nounwind {
entry:
;  %s = alloca %struct.S, align 4
;  call void @foo1(%struct.S* byval %s)
  %s = alloca %struct.S, align 8
  call void @foo1(%struct.S* byval align 8 %s)
  ret void
}

;declare void @foo1(%struct.S* byval)
