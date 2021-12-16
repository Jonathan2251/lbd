; RUN: llc -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=true < %s | FileCheck %s
; RUN: llc -march=cpu0 -mcpu=cpu032II -cpu0-s32-calls=false < %s | FileCheck %s -check-prefix=cpu032II

define i64 @add64(i64 %u, i64 %v) nounwind  {
entry:
; CHECK: addu
; CHECK: cmp
; CHECK: andi
; CHECK: addu
; cpu032II: sltu 
; cpu032II: addu
  %tmp2 = add i64 %u, %v  
  ret i64 %tmp2
}

define i64 @sub64(i64 %u, i64 %v) nounwind  {
entry:
; CHECK: cmp
; CHECK: andi
; CHECK: subu
; CHECK: subu
; cpu032II: sltu 
; cpu032II: subu
  %tmp2 = sub i64 %u, %v
  ret i64 %tmp2
}
