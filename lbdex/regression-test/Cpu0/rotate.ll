; RUN: llc -march=cpu0 -mcpu=cpu032I -cpu0-s32-calls=false < %s | FileCheck %s

; CHECK:  rolv $2, $4, $5
define i32 @rot0(i32 %a, i32 %b) nounwind readnone {
entry:
  %shl = shl i32 %a, %b
  %sub = sub i32 32, %b
  %shr = lshr i32 %a, %sub
  %or = or i32 %shr, %shl
  ret i32 %or
}

; CHECK:  rol  $2, $4, 30
define i32 @rot1(i32 %a) nounwind readnone {
entry:
  %shl = shl i32 %a, 30
  %shr = lshr i32 %a, 2
  %or = or i32 %shl, %shr
  ret i32 %or
}

; CHECK:  rorv $2, $4, $5
define i32 @rot2(i32 %a, i32 %b) nounwind readnone {
entry:
  %shr = lshr i32 %a, %b
  %sub = sub i32 32, %b
  %shl = shl i32 %a, %sub
  %or = or i32 %shl, %shr
  ret i32 %or
}

