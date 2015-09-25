; RUN: llc -march=cpu0el < %s | FileCheck %s

define i32 @main() #0 {
entry:
; CHECK: addiu	$2, $zero, 0
; CHECK: ret $lr
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0
}

define void @_Z1fd() {
entry:
; CHECK: ret $lr
  ret void
}

