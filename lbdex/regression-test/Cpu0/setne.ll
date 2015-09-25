; RUN: llc  -march=cpu0 -mcpu=cpu032I  -relocation-model=pic -O3 %s -o - | FileCheck %s -check-prefix=cpu032I
; RUN: llc  -march=cpu0 -mcpu=cpu032II  -relocation-model=pic -O3 %s -o - | FileCheck %s -check-prefix=cpu032II

@i = global i32 1, align 4
@j = global i32 10, align 4
@k = global i32 1, align 4
@r1 = common global i32 0, align 4
@r2 = common global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, i32* @i, align 4
  %1 = load i32, i32* @k, align 4
  %cmp = icmp ne i32 %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @r1, align 4
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 2
; cpu032I:  shr	$[[T2:[0-9]+|t9]], $[[T1]], 1
; cpu032I:  xori	${{[0-9]+|t9}}, $[[T2]], 1
; cpu032II:  xor	$[[T0:[0-9]+|t9]], ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032II:  sltu	${{[0-9]+|t9}}, $zero, $[[T0]]
  ret void
}
