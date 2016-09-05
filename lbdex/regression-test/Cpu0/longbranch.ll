; RUN: llc -march=cpu0el -mcpu=cpu032II -relocation-model=pic -filetype=asm -force-cpu0-long-branch < %s | FileCheck %s -check-prefix=O32
; RUN: llc -march=cpu0el -mcpu=cpu032II -relocation-model=pic -filetype=asm < %s | FileCheck %s


@x = external global i32

define void @test1(i32 signext %s) {
entry:
  %cmp = icmp eq i32 %s, 0
  br i1 %cmp, label %end, label %then

then:
  store i32 1, i32* @x, align 4
  br label %end

end:
  ret void

; Check the MIPS32 version.  Check that branch logic is inverted, so that the
; target of the new branch (bnez) is the fallthrough block of the original
; branch.  Check that fallthrough block of the new branch contains long branch
; expansion which at the end indirectly jumps to the target of the original
; branch.

; O32:        lui     $[[R0:[0-9]+]], %hi(_gp_disp)
; O32:        addiu   $[[R0]], $[[R0]], %lo(_gp_disp)
; O32:        bne     $4, $zero, $[[BB0:BB[0-9_]+]]
; O32:        nop

; Check for long branch expansion:
; O32:             addiu   $sp, $sp, -8
; O32-NEXT:        st      $lr, 0($sp)
; O32-NEXT:        lui     $1, %hi(($[[BB2:BB[0-9_]+]])-($[[BB1:BB[0-9_]+]]))
; O32-NEXT:        addiu   $1, $1, %lo(($[[BB2]])-($[[BB1]]))
; O32-NEXT:        bal     $[[BB1]]
; O32-NEXT:   $[[BB1]]:
; O32-NEXT:        addu    $1, $lr, $1
; O32-NEXT:        ld      $lr, 0($sp)
; O32-NEXT:        addiu   $sp, $sp, 8
; O32-NEXT:        jr      $1
; O32-NEXT:        nop

; O32:   $[[BB0]]:
; O32:        ld      $[[R1:[0-9]+]], %got_lo(x)
; O32:        addiu   $[[R2:[0-9]+]], $zero, 1
; O32:        st      $[[R2]], 0($[[R1]])
; O32:   $[[BB2]]:
; O32:        ret      $lr
; O32:        nop

; First check the normal version (without long branch).  beqz jumps to return,
; and fallthrough block stores 1 to global variable.

; CHECK:        lui     $[[R0:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[R0]], $[[R0]], %lo(_gp_disp)
; CHECK:        beq     $4, $zero, $[[BB0:BB[0-9_]+]]
; CHECK:        nop
; CHECK:        ld      $[[R2:[0-9]+]], %got_lo(x)(${{[0-9]+|t9}})
; CHECK:        addiu   $[[R3:[0-9]+]], $zero, 1
; CHECK:        st      $[[R3]], 0($[[R2]])
; CHECK:   $[[BB0]]:
; CHECK:        ret      $lr
; CHECK:        nop
}
