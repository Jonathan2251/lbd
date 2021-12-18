; RUN: llc -march=cpu0el -mcpu=cpu032II -relocation-model=pic < %s | \
; RUN:     FileCheck %s -check-prefix=PIC
; RUN: llc -march=cpu0el -mcpu=cpu032II -relocation-model=static < \
; RUN:     %s | FileCheck %s -check-prefix=STATIC

@t1 = thread_local global i32 0, align 4

define i32 @f1() nounwind {
entry:
  %tmp = load i32, i32* @t1, align 4
  ret i32 %tmp

; CHECK: f1:

; PIC:   ld      $t9, %call16(__tls_get_addr)($gp)
; PIC:   ori   $4, $gp, %tlsgd(t1)
; PIC:   jalr    $t9
; PIC:   ld      $2, 0($2)

; STATIC:   ld     $[[R0:[0-9]+|t9]], %gottprel(t1)($gp)
; STATIC:   ld      $2, 0($[[R0]])
}


@t2 = external thread_local global i32

define i32 @f2() nounwind {
entry:
  %tmp = load i32, i32* @t2, align 4
  ret i32 %tmp

; CHECK: f2:

; PIC:   ld      $t9, %call16(__tls_get_addr)($gp)
; PIC:   ori   $4, $gp, %tlsgd(t2)
; PIC:   jalr    $t9
; PIC:   ld      $2, 0($2)

; STATIC:   ld      $[[R0:[0-9]+|t9]], %gottprel(t2)($gp)
; STATIC:   ld      $2, 0($[[R0]])
}

@f3.i = internal thread_local unnamed_addr global i32 1, align 4

define i32 @f3() nounwind {
entry:
; CHECK: f3:

; PIC:   ori   $4, ${{[a-z0-9]+}}, %tlsldm(f3.i)
; PIC:   jalr    $t9
; PIC:   lui     $[[R0:[0-9]+|t9]], %dtp_hi(f3.i)
; PIC:   addu    $[[R1:[0-9]+|t9]], $[[R0]], $2
; PIC:   ori   ${{[0-9]+|t9}}, $[[R1]], %dtp_lo(f3.i)

  %0 = load i32, i32* @f3.i, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @f3.i, align 4
  ret i32 %inc
}

