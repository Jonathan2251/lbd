; RUN: llc -march=cpu0el -relocation-model=pic < %s | FileCheck %s

define void @count(i32 %x, i32 %y, i32 %z) noreturn nounwind readnone {
entry:
  br label %bosco

bosco:                                            ; preds = %bosco, %entry
  br label %bosco
}

; CHECK: jmp
