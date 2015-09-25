; RUN: llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=false -filetype=asm < %s | FileCheck %s -check-prefix=STATIC
; RUN: llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=true -filetype=asm < %s | FileCheck %s -check-prefix=STATIC
; RUN: llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=false -filetype=asm < %s | FileCheck %s -check-prefix=PIC
; RUN: llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=true -filetype=asm < %s | FileCheck %s -check-prefix=PIC

; ModuleID = 'ch7_5_2.bc'

@_ZZ4mainE1a = private unnamed_addr constant [3 x i32] [i32 0, i32 1, i32 2], align 4

; Function Attrs: nounwind
define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %a = alloca [3 x i32], align 4
  store i32 0, i32* %retval
  %0 = bitcast [3 x i32]* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* bitcast ([3 x i32]* @_ZZ4mainE1a to i8*), i32 12, i32 4, i1 false)

; STATIC:  lui	$[[T0:[0-9]+|t9]], %hi($_ZZ4mainE1a)
; STATIC:  ori	$[[T1:[0-9]+|t9]], $[[T0]], %lo($_ZZ4mainE1a)
; STATIC:  ld	$[[T2:[0-9]+|t9]], 8($[[T1]])
; STATIC:  st	$[[T2]], {{[0-9]+|t9}}(${{[fs]p}})
; STATIC:  ld	$[[T2:[0-9]+|t9]], 4($[[T1]])
; STATIC:  st	$[[T2]], {{[0-9]+|t9}}(${{[fs]p}})
; STATIC:  ld	$[[T2:[0-9]+|t9]], 0($[[T1]])
; STATIC:  st	$[[T2]], {{[0-9]+|t9}}(${{[fs]p}})

; PIC:  ld	$[[T0:[0-9]+|t9]], %got($_ZZ4mainE1a)($gp)
; PIC:  ori	$[[T1:[0-9]+|t9]], $[[T0]], %lo($_ZZ4mainE1a)
; PIC:  ld	$[[T2:[0-9]+|t9]], 8($[[T1]])
; PIC:  st	$[[T2]], {{[0-9]+|t9}}(${{[fs]p}})
; PIC:  ld	$[[T2:[0-9]+|t9]], 4($[[T1]])
; PIC:  st	$[[T2]], {{[0-9]+|t9}}(${{[fs]p}})
; PIC:  ld	$[[T2:[0-9]+|t9]], 0($[[T1]])
; PIC:  st	$[[T2]], {{[0-9]+|t9}}(${{[fs]p}})

  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) #1

