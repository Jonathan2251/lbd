; RUN: llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=false -filetype=asm < %s | FileCheck %s -check-prefix=STATIC_LARGE
; RUN: llc -march=cpu0 -relocation-model=static -cpu0-use-small-section=true -filetype=asm < %s | FileCheck %s -check-prefix=STATIC_SMALL
; RUN: llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=false -filetype=asm < %s | FileCheck %s -check-prefix=PIC_LARGE
; RUN: llc -march=cpu0 -relocation-model=pic -cpu0-use-small-section=true -filetype=asm < %s | FileCheck %s -check-prefix=PIC_SMALL

; ModuleID = 'ch7_2.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

%struct.Date = type { i16, i8, i8, i8, i8, i8 }

@b = global [4 x i8] c"abc\00", align 1
@_ZZ9test_charvE5date1 = private unnamed_addr constant { i16, i8, i8, i8, i8, i8, i8 } { i16 2012, i8 11, i8 25, i8 9, i8 40, i8 15, i8 undef }, align 2

; Function Attrs: nounwind
define i32 @_Z9test_charv() #0 {
entry:
  %a = alloca i8, align 1
  %c = alloca i8, align 1
  %date1 = alloca %struct.Date, align 2
  %m = alloca i8, align 1
  %s = alloca i8, align 1
  %0 = load i8* getelementptr inbounds ([4 x i8]* @b, i32 0, i32 1), align 1
  store i8 %0, i8* %a, align 1
  %1 = load i8* getelementptr inbounds ([4 x i8]* @b, i32 0, i32 1), align 1
  store i8 %1, i8* %c, align 1
; STATIC_LARGE:  lui	$[[T0:[0-9]+|t9]], %hi(b)
; STATIC_LARGE:  ori	$[[T1:[0-9]+|t9]], $[[T0]], %lo(b)
; STATIC_LARGE:  lbu	$[[T2:[0-9]+|t9]], 1($[[T1]])
; STATIC_LARGE:  sb	$[[T2]], {{[0-9]+|t9}}(${{[fs]p}})
; STATIC_LARGE:  lbu	$[[T1:[0-9]+|t9]], 1($[[T0:[0-9]+|t9]])
; STATIC_LARGE:  sb	$[[T1]], {{[0-9]+|t9}}(${{[fs]p}})
; STATIC_SMALL:  ori	$[[T0:[0-9]+|t9]], $gp, %gp_rel(b)
; STATIC_SMALL:  lbu	$[[T1:[0-9]+|t9]], 1($[[T0]])
; STATIC_SMALL:  sb	$[[T1]], {{[0-9]+|t9}}(${{[fs]p}})
; STATIC_SMALL:  lbu	$[[T1:[0-9]+|t9]], 1($[[T0:[0-9]+|t9]])
; STATIC_SMALL:  sb	$[[T1]]
; PIC_LARGE:  .cpload	$t9
; PIC_LARGE:  lui	$[[T0:[0-9]+|t9]], %got_hi(b)
; PIC_LARGE:  addu	$[[T1:[0-9]+|t9]], $[[T0]], $gp
; PIC_LARGE:  ld	$[[T2:[0-9]+|t9]], %got_lo(b)($[[T1]])
; PIC_LARGE:  lbu	$[[T3:[0-9]+|t9]], 1($[[T2]])
; PIC_LARGE:  sb	$[[T3]], {{[0-9]+|t9}}(${{[fs]p}})
; PIC_LARGE:  lbu	$[[T1:[0-9]+|t9]], 1($[[T0:[0-9]+|t9]])
; PIC_LARGE:  sb	$[[T1]], {{[0-9]+|t9}}(${{[fs]p}})
; PIC_SMALL:  .cpload	$t9
; PIC_SMALL:  ld	$[[T0:[0-9]+|t9]], %got(b)($gp)
; PIC_SMALL:  lbu	$[[T1:[0-9]+|t9]], 1($[[T0]])
; PIC_SMALL:  sb	$[[T1]], {{[0-9]+|t9}}(${{[fs]p}})
; PIC_SMALL:  lbu	$[[T1:[0-9]+|t9]], 1($[[T0:[0-9]+|t9]])
; PIC_SMALL:  sb	$[[T1]], {{[0-9]+|t9}}(${{[fs]p}})
  %2 = bitcast %struct.Date* %date1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %2, i8* bitcast ({ i16, i8, i8, i8, i8, i8, i8 }* @_ZZ9test_charvE5date1 to i8*), i32 8, i32 2, i1 false)
; STATIC_LARGE:  lui	$[[T0:[0-9]+|t9]], %hi($_ZZ9test_charvE5date1)
; STATIC_LARGE:  ori	$[[T1:[0-9]+|t9]], $[[T0]], %lo($_ZZ9test_charvE5date1)
; STATIC_LARGE:  lhu	$[[T2:[0-9]+|t9]], 4($[[T1]])
; STATIC_LARGE:  shl	$[[T3:[0-9]+|t9]], $[[T2]], 16
; STATIC_LARGE:  lhu	$[[T4:[0-9]+|t9]], 6($[[T0]])
; STATIC_LARGE:  or	$[[T5:[0-9]+|t9]], $[[T3]], $[[T4]]
; STATIC_LARGE:  st	$[[T3]], {{[0-9]+|t9}}(${{[fs]p}})
; STATIC_LARGE:  lhu	$[[T6:[0-9]+|t9]], 2($[[T0]])
; STATIC_LARGE:  lhu	$[[T7:[0-9]+|t9]], 0($[[T0]])
; STATIC_LARGE:  shl	$[[T8:[0-9]+|t9]], $[[T7]], 16
; STATIC_LARGE:  or	$[[T9:[0-9]+|t9]], $[[T8]], $[[T6]]
; STATIC_LARGE:  st	$[[T9]], {{[0-9]+|t9}}(${{[fs]p}})
; STATIC_SMALL:  lui	$[[T0:[0-9]+|t9]], %hi($_ZZ9test_charvE5date1)
; STATIC_SMALL:  ori	$[[T1:[0-9]+|t9]], $[[T0]], %lo($_ZZ9test_charvE5date1)
; STATIC_SMALL:  lhu	$[[T2:[0-9]+|t9]], 4($[[T1]])
; STATIC_SMALL:  shl	$[[T3:[0-9]+|t9]], $[[T2]], 16
; STATIC_SMALL:  lhu	$[[T4:[0-9]+|t9]], 6($[[T0]])
; STATIC_SMALL:  or	$[[T5:[0-9]+|t9]], $[[T3]], $[[T4]]
; STATIC_SMALL:  st	$[[T3]], {{[0-9]+|t9}}(${{[fs]p}})
; STATIC_SMALL:  lhu	$[[T6:[0-9]+|t9]], 2($[[T0]])
; STATIC_SMALL:  lhu	$[[T7:[0-9]+|t9]], 0($[[T0]])
; STATIC_SMALL:  shl	$[[T8:[0-9]+|t9]], $[[T7]], 16
; STATIC_SMALL:  or	$[[T9:[0-9]+|t9]], $[[T8]], $[[T6]]
; STATIC_SMALL:  st	$[[T9]], {{[0-9]+|t9}}(${{[fs]p}})

; PIC_LARGE:  ld	$[[T0:[0-9]+|t9]], %got($_ZZ9test_charvE5date1)($gp)
; PIC_LARGE:  ori	$[[T1:[0-9]+|t9]], $[[T0]], %lo($_ZZ9test_charvE5date1)
; PIC_LARGE:  lhu	$[[T2:[0-9]+|t9]], 4($[[T1]])
; PIC_LARGE:  shl	$[[T3:[0-9]+|t9]], $[[T2]], 16
; PIC_LARGE:  lhu	$[[T4:[0-9]+|t9]], 6($[[T0]])
; PIC_LARGE:  or	$[[T5:[0-9]+|t9]], $[[T3]], $[[T4]]
; PIC_LARGE:  st	$[[T3]], {{[0-9]+|t9}}(${{[fs]p}})
; PIC_LARGE:  lhu	$[[T6:[0-9]+|t9]], 2($[[T0]])
; PIC_LARGE:  lhu	$[[T7:[0-9]+|t9]], 0($[[T0]])
; PIC_LARGE:  shl	$[[T8:[0-9]+|t9]], $[[T7]], 16
; PIC_LARGE:  or	$[[T9:[0-9]+|t9]], $[[T8]], $[[T6]]
; PIC_LARGE:  st	$[[T9]], {{[0-9]+|t9}}(${{[fs]p}})
; PIC_SMALL:  ld	$[[T0:[0-9]+|t9]], %got($_ZZ9test_charvE5date1)
; PIC_SMALL:  ori	$[[T1:[0-9]+|t9]], $[[T0]], %lo($_ZZ9test_charvE5date1)
; PIC_SMALL:  lhu	$[[T2:[0-9]+|t9]], 4($[[T1]])
; PIC_SMALL:  shl	$[[T3:[0-9]+|t9]], $[[T2]], 16
; PIC_SMALL:  lhu	$[[T4:[0-9]+|t9]], 6($[[T0]])
; PIC_SMALL:  or	$[[T5:[0-9]+|t9]], $[[T3]], $[[T4]]
; PIC_SMALL:  st	$[[T3]], {{[0-9]+|t9}}(${{[fs]p}})
; PIC_SMALL:  lhu	$[[T6:[0-9]+|t9]], 2($[[T0]])
; PIC_SMALL:  lhu	$[[T7:[0-9]+|t9]], 0($[[T0]])
; PIC_SMALL:  shl	$[[T8:[0-9]+|t9]], $[[T7]], 16
; PIC_SMALL:  or	$[[T9:[0-9]+|t9]], $[[T8]], $[[T6]]
; PIC_SMALL:  st	$[[T9]], {{[0-9]+|t9}}(${{[fs]p}})
  %month = getelementptr inbounds %struct.Date* %date1, i32 0, i32 1
  %3 = load i8* %month, align 1
  store i8 %3, i8* %m, align 1
  %second = getelementptr inbounds %struct.Date* %date1, i32 0, i32 5
  %4 = load i8* %second, align 1
  store i8 %4, i8* %s, align 1
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
