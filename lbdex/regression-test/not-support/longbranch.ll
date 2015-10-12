; RUN: llc -march=cpu0el -mcpu=cpu032II -relocation-model=pic -filetype=asm -force-cpu0-long-branch < %s | FileCheck %s


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


; First check the normal version (without long branch).  beqz jumps to return,
; and fallthrough block stores 1 to global variable.

; CHECK:        lui     $[[R0:[0-9]+]], %got_hi(x)
; CHECK:        addu    $[[R1:[0-9]+]], $[[R0]], $gp;
; CHECK:        beqz    $4, $[[BB0:BB[0-9_]+]]
; CHECK:        ld      $[[R2:[0-9]+]], %got_lo(x)($[[R1]])
; CHECK:        addiu   $[[R3:[0-9]+]], $zero, 1
; CHECK:        st      $[[R3]], 0($[[R2]])
; CHECK:   $[[BB0]]:
; CHECK:        ret      $lr
; CHECK:        nop
}
