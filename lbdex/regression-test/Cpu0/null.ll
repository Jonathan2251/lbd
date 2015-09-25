; RUN: llc  -march=cpu0el -mcpu=cpu032II < %s | FileCheck %s -check-prefix=16


define i32 @main() nounwind {
entry:
  ret i32 0

; 16:	ret	$lr

}
