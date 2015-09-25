; RUN: llc  -march=cpu0 -relocation-model=pic < %s | FileCheck %s

define i32 @test_sext_inreg_from_32(i32 %in) {
; CHECK: test_sext_inreg_from_32:

  %small = trunc i32 %in to i1
  %ext = sext i1 %small to i32
; CHECK:  shl	$[[T1:[0-9]+|t9]], ${{[0-9]+|t9}}, 31
; CHECK:  sra	${{[0-9]+|t9}}, $[[T1]], 31
  ret i32 %ext
}

