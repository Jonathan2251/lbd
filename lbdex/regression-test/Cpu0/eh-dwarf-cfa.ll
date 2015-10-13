; RUN: llc -march=cpu0el -mcpu=cpu032II < %s | FileCheck %s

declare i8* @llvm.eh.dwarf.cfa(i32) nounwind
declare i8* @llvm.frameaddress(i32) nounwind readnone

define i8* @f1() nounwind {
entry:
  %x = alloca [32 x i8], align 1
  %0 = call i8* @llvm.eh.dwarf.cfa(i32 0)
  ret i8* %0

; CHECK:        addiu   $sp, $sp, -40
; CHECK:        addu    $2,  $zero, $fp
}


define i8* @f2() nounwind {
entry:
  %x = alloca [65536 x i8], align 1
  %0 = call i8* @llvm.eh.dwarf.cfa(i32 0)
  ret i8* %0

; check stack size (65536 + 8)
; CHECK:        lui     $[[R0:[a-z0-9]+]], 65535
; CHECK:        addiu   $[[R0]], $[[R0]], -8
; CHECK:        addu    $sp, $sp, $[[R0]]

; check return value ($sp + stack size)
; CHECK:        addu    $2,  $zero, $fp
}


define i32 @f3() nounwind {
entry:
  %x = alloca [32 x i8], align 1
  %0 = call i8* @llvm.eh.dwarf.cfa(i32 0)
  %1 = ptrtoint i8* %0 to i32
  %2 = call i8* @llvm.frameaddress(i32 0)
  %3 = ptrtoint i8* %2 to i32
  %add = add i32 %1, %3
  ret i32 %add

; CHECK:        addiu   $sp, $sp, -40

; check return value ($fp + stack size + $fp)
; CHECK:        move     $fp, $sp
; CHECK:        addu    $2, $fp, $fp
}

