; RUN: llc < %s -march=cpu0 -mcpu=cpu032I | FileCheck %s

@g1 = external global i32

define i32 @foo0(i32 %s) nounwind readonly {
entry:
; CHECK:     addiu	${{[0-9]+|t9}}, $zero, 0
; CHECK:     movn ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %tobool = icmp ne i32 %s, 0
  %0 = load i32, i32* @g1, align 4
  %cond = select i1 %tobool, i32 0, i32 %0
  ret i32 %cond
}

define i32 @foo1(i32 %s) nounwind readonly {
entry:
; CHECK:     addiu	${{[0-9]+|t9}}, $zero, 0
; CHECK:     movn ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  %tobool = icmp ne i32 %s, 0
  %0 = load i32, i32* @g1, align 4
  %cond = select i1 %tobool, i32 %0, i32 0
  ret i32 %cond
}

