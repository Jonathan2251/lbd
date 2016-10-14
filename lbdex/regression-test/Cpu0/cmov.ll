; RUN: llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=true < %s | FileCheck %s -check-prefix=O32

@i1 = global [3 x i32] [i32 1, i32 2, i32 3], align 4
@i3 = common global i32* null, align 4

; O32:  lui $[[R0:[0-9]+|t9]], %got_hi
; O32:  ori $[[R1:[0-9]+|t9]], ${{[0-9]+|t9}}, %got_lo
; O32:  movn ${{[0-9]+|t9}}, $[[R1]], ${{[0-9]+|t9}} 
define i32* @cmov1(i32 %s) nounwind readonly {
entry:
  %tobool = icmp ne i32 %s, 0
  %tmp1 = load i32*, i32** @i3, align 4
  %cond = select i1 %tobool, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @i1, i32 0, i32 0), i32* %tmp1
  ret i32* %cond
}

@c = global i32 1, align 4
@d = global i32 0, align 4

; O32: cmov2:
; O32: ori $[[R1:[0-9]+|t9]], ${{[a-z0-9]+}}, %got(d)
; O32: ori $[[R0:[0-9]+|t9]], ${{[a-z0-9]+}}, %got(c)
; O32: movn  ${{[0-9]+|t9}}, $[[R0]], ${{[0-9]+|t9}}
define i32 @cmov2(i32 %s) nounwind readonly {
entry:
  %tobool = icmp ne i32 %s, 0
  %tmp1 = load i32, i32* @c, align 4
  %tmp2 = load i32, i32* @d, align 4
  %cond = select i1 %tobool, i32 %tmp1, i32 %tmp2
  ret i32 %cond
}

; O32: cmov3:
; O32: addiu $[[R0:[0-9]+|t9]], $zero, 234
; O32: xor $[[R1:[0-9]+|t9]], ${{[a-z0-9]+}}, $[[R0:[0-9]+|t9]]
; O32: movz ${{[0-9]+|t9}}, ${{[0-9]+|t9}}, $[[R1]]
define i32 @cmov3(i32 %a, i32 %b, i32 %c) nounwind readnone {
entry:
  %cmp = icmp eq i32 %a, 234
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

