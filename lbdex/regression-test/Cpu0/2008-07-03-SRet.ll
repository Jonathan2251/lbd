; RUN: llc -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=true < %s | FileCheck %s

%struct.sret0 = type { i32, i32, i32 }

define void @test0(%struct.sret0* noalias sret(%struct.sret0) %agg.result, i32 %dummy) nounwind {
entry:
; Support by section "Structure type support" of chapter "Function call'
; CHECK: ld ${{[0-9]+|t9}}, 0($sp)
; CHECK: ld ${{[0-9]+|t9}}, 4($sp)
; CHECK: st ${{[0-9]+|t9}}, 8($2)
; CHECK: st ${{[0-9]+|t9}}, 4($2)
; CHECK: st ${{[0-9]+|t9}}, 0($2)
  getelementptr %struct.sret0, %struct.sret0* %agg.result, i32 0, i32 0    ; <i32*>:0 [#uses=1]
  store i32 %dummy, i32* %0, align 4
  getelementptr %struct.sret0, %struct.sret0* %agg.result, i32 0, i32 1    ; <i32*>:1 [#uses=1]
  store i32 %dummy, i32* %1, align 4
  getelementptr %struct.sret0, %struct.sret0* %agg.result, i32 0, i32 2    ; <i32*>:2 [#uses=1]
  store i32 %dummy, i32* %2, align 4
  ret void
}

