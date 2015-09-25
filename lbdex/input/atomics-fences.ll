; RUN: llc < %s -march=cpu0 -mcpu=cpu032II | FileCheck %s --check-prefix=CHECK

; Fences
define void @fence_acquire() {
; CHECK-LABEL: fence_acquire
; CHECK: sync
  fence acquire
  ret void
}
define void @fence_release() {
; CHECK-LABEL: fence_release
; CHECK: sync
  fence release
  ret void
}
define void @fence_seq_cst() {
; CHECK-LABEL: fence_seq_cst
; CHECK: sync
  fence seq_cst
  ret void
}
