; RUN: llc -march=cpu0el -relocation-model=static -cpu0-use-small-section=true \
; RUN: < %s | FileCheck %s -check-prefix=STATIC

@i = internal unnamed_addr global i32 0, align 4

define i32 @geti() nounwind readonly {
entry:
; STATIC: ori  $[[R0:[0-9]+|t9]], $gp, %gp_rel(i)
; STATIC: ld  ${{[0-9]+|t9}}, 0($[[R0]])
  %0 = load i32, i32* @i, align 4
  ret i32 %0
}

