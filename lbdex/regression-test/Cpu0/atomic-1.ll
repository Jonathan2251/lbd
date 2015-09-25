; RUN: llc < %s -march=cpu0 -mcpu=cpu032II |  FileCheck %s

define i32 @exchange_and_add(i32* %mem, i32 %val) nounwind {
; CHECK-LABEL: exchange_and_add:
; CHECK: ll
; CHECK: sc
  %tmp = atomicrmw add i32* %mem, i32 %val monotonic
  ret i32 %tmp
}

define i32 @exchange_and_cmp(i32* %mem) nounwind {
; CHECK-LABEL: exchange_and_cmp:
; CHECK: ll
; CHECK: sc
  %tmppair = cmpxchg i32* %mem, i32 0, i32 1 monotonic monotonic
  %tmp = extractvalue { i32, i1 } %tmppair, 0
  ret i32 %tmp
}

define i32 @exchange(i32* %mem, i32 %val) nounwind {
; CHECK-LABEL: exchange:
; CHECK: ll
; CHECK: sc
  %tmp = atomicrmw xchg i32* %mem, i32 1 monotonic
  ret i32 %tmp
}
