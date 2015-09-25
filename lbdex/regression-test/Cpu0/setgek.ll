; RUN: llc  -march=cpu0 -mcpu=cpu032I  -relocation-model=pic -O3 %s -o - | FileCheck %s -check-prefix=cpu032I
; RUN: llc  -march=cpu0 -mcpu=cpu032II  -relocation-model=pic -O3 %s -o - | FileCheck %s -check-prefix=cpu032II

@k = global i32 10, align 4
@r1 = common global i32 0, align 4
@r2 = common global i32 0, align 4
@r3 = common global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, i32* @k, align 4
  %cmp = icmp sgt i32 %0, -32769
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @r1, align 4
; cpu032I:  cmp	$sw, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
; cpu032I:  andi	$[[T1:[0-9]+|t9]], $sw, 1
; cpu032II:  slt	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, ${{[0-9]+|t9}}
  ret void
}
