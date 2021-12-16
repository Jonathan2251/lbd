; RUN: llc  -march=cpu0 -relocation-model=pic < %s | FileCheck %s

define i32 @test_sext_inreg_from_32(i32 %in) {
; CHECK: test_sext_inreg_from_32:

  %small = trunc i32 %in to i1
  %ext = sext i1 %small to i32
; CHECK:  andi	$[[T1:[0-9]+|t9]], $4, 1
; CHECK:  addiu	$[[T2:[0-9]+|t9]], $zero, 0
; CHECK:  subu	${{[0-9]+|t9}}, $[[T2]], $[[T1]]
  ret i32 %ext
}

