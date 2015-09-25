; RUN: llc  -march=cpu0 -mcpu=cpu032I  -relocation-model=pic %s -o - | FileCheck %s -check-prefix=cpu032I
; RUN: llc  -march=cpu0 -mcpu=cpu032II  -relocation-model=pic %s -o - | FileCheck %s -check-prefix=cpu032II
; terminal command (not work): llc  -march=cpu0 -mcpu=cpu032II  -relocation-model=pic %s < %s | FileCheck %s

@i = global i32 1, align 4
@j = global i32 10, align 4
@k = global i32 1, align 4
@r1 = common global i32 0, align 4
@r2 = common global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, i32* @i, align 4
  %1 = load i32, i32* @k, align 4
  %cmp = icmp eq i32 %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @r1, align 4

; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 2
; cpu032I:  shr	${{[0-9]+|t9}}, $[[T1]], 1
; cpu032II:  xor	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  sltiu	${{[0-9]+|t9}}, $[[T0]], 1

  ret void
}

