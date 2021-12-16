; RUN: llc  < %s -march=cpu0 -mcpu=cpu032I -relocation-model=pic -cpu0-s32-calls=true | FileCheck %s -check-prefix=CHECK
%struct.S2 = type { %struct.S1, %struct.S1 }
%struct.S1 = type { i8, i8 }
%struct.S4 = type { [7 x i8] }

@s2 = common global %struct.S2 zeroinitializer, align 1
@s4 = common global %struct.S4 zeroinitializer, align 1

define void @foo1() nounwind {
entry:
; CHECK: lbu ${{[0-9]+|t9}}, 3($[[R0:[0-9]+|t9]])
; CHECK: sb  ${{[0-9]+|t9}}, 1($sp)
; CHECK: lbu ${{[0-9]+|t9}}, 2($[[R0]])
; CHECK: sb  ${{[0-9]+|t9}}, 0($sp)
; CHECK: jalr
; CHECK: lbu ${{[0-9]+|t9}}, 6($[[R0:[0-9]+|t9]])
; CHECK: sb  ${{[0-9]+|t9}}, 6($sp)
; CHECK: lbu ${{[0-9]+|t9}}, 5($[[R0]])
; CHECK: sb  ${{[0-9]+|t9}}, 5($sp)
; CHECK: lbu ${{[0-9]+|t9}}, 4($[[R0:[0-9]+|t9]])
; CHECK: sb  ${{[0-9]+|t9}}, 4($sp)
; CHECK: lbu ${{[0-9]+|t9}}, 3($[[R0]])
; CHECK: sb  ${{[0-9]+|t9}}, 3($sp)
; CHECK: lbu ${{[0-9]+|t9}}, 2($[[R0:[0-9]+|t9]])
; CHECK: sb  ${{[0-9]+|t9}}, 2($sp)
; CHECK: lbu ${{[0-9]+|t9}}, 1($[[R0]])
; CHECK: sb  ${{[0-9]+|t9}}, 1($sp)
; CHECK: lbu ${{[0-9]+|t9}}, 0($[[R0]])
; CHECK: sb  ${{[0-9]+|t9}}, 0($sp)
; CHECK: jalr

  tail call void @foo2(%struct.S1* byval(%struct.S1) getelementptr inbounds (%struct.S2, %struct.S2* @s2, i32 0, i32 1)) nounwind
  tail call void @foo4(%struct.S4* byval(%struct.S4) @s4) nounwind
  ret void
}

declare void @foo2(%struct.S1* byval(%struct.S1))

declare void @foo4(%struct.S4* byval(%struct.S4))
