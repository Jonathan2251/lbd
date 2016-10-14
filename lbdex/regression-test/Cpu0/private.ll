; Test to make sure that the 'private' is used correctly.
;
; RUN: llc -march=cpu0el -mcpu=cpu032II -relocation-model=pic < %s | FileCheck %s

define private void @foo() {
; CHECK: foo:
  ret void
}

@baz = private global i32 4

define i32 @bar() {
; CHECK: bar:
; CHECK: call16($foo)
; CHECK: ld $[[R0:[0-9]+|t9]], %got($baz)($gp)
; CHECK: ori ${{[0-9]+|t9}}, $[[R0]], %lo($baz)
  call void @foo()
  %1 = load i32, i32* @baz, align 4
  ret i32 %1
}
