; RUN: llc -march=cpu0 -relocation-model=pic -filetype=asm < %s | FileCheck %s

; /// start
define zeroext i1 @verify_load_bool() #0 {
entry:
  %retval = alloca i1, align 1
  store i1 1, i1* %retval, align 1
  %0 = load i1* %retval
  ret i1 %0
; CHECK:  addiu	$[[T0:[0-9]+|t9]], $zero, 1
; CHECK:  sb	$[[T0]], 7($sp)
; CHECK:  lbu	$2, 7($sp)
}
