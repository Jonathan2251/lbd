; RUN: llc  < %s -march=cpu0el -mcpu=cpu032I -cpu0-s32-calls=true | FileCheck %s 
; RUN: llc  < %s -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=true   | FileCheck %s

define void @f(i64 %l, i64* nocapture %p) nounwind {
entry:
; CHECK: lui ${{[0-9]+|t9}}, 37035
; CHECK: ori ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, 52719
; CHECK: cmpu $sw,  ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; CHECK: andi ${{[0-9]+|t9}}, $sw, 1
; CHECK: lui ${{[0-9]+|t9}}, 4660
; CHECK: ori ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, 22136
; CHECK: addu
  %add = add i64 %l, 1311768467294899695
  store i64 %add, i64* %p, align 4 
  ret void
}

