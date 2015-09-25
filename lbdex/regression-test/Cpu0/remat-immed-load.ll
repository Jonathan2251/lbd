; RUN: llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic < %s | FileCheck %s -check-prefix=32

define void @f0() nounwind {
entry:
; 32:  addiu ${{[0-9]+|t9}}, $zero, 1
; 32:  addu $4, $zero, ${{9|10}}
; 32:  addu $4, $zero, ${{9|10}}

  tail call void @foo1(i32 1) nounwind
  tail call void @foo1(i32 1) nounwind
  ret void
}

declare void @foo1(i32)

define void @f3() nounwind {
entry:
; 32:  addiu ${{[0-9]+|t9}}, $zero, 1
; 32:  addu $4, $zero, ${{9|10}}
; 32:  addu $4, $zero, ${{9|10}}

  tail call void @foo2(i64 1) nounwind
  tail call void @foo2(i64 1) nounwind
  ret void
}

declare void @foo2(i64)

define void @f5() nounwind {
entry:
; 32:  lui ${{[0-9]+|t9}}, 1
; 32:  addu $4, $zero, ${{9|10}}
; 32:  addu $4, $zero, ${{9|10}}

  tail call void @f6(i32 65536) nounwind
  tail call void @f6(i32 65536) nounwind
  ret void
}

declare void @f6(i32)

define void @f7() nounwind {
entry:
; 32:  lui ${{[0-9]+|t9}}, 1
; 32:  addu $4, $zero, ${{9|10}}
; 32:  addu $4, $zero, ${{9|10}}

  tail call void @f8(i64 65536) nounwind
  tail call void @f8(i64 65536) nounwind
  ret void
}

declare void @f8(i64)

