; RUN: llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic < %s | FileCheck %s

define i8* @f1() nounwind {
entry:
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0

; CHECK:    addu	$2, $zero, $lr
}

define i8* @f2() nounwind {
entry:
  call void @g()
  %0 = call i8* @llvm.returnaddress(i32 0)
  ret i8* %0

; CHECK:    jalr
; CHECK:    addu	$2, $zero, $lr
}

declare i8* @llvm.returnaddress(i32) nounwind readnone
declare void @g()
