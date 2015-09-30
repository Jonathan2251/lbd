; RUN: llc -march=cpu0el -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC-O32
; RUN: llc -march=cpu0el -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC-O32

@reg = common global i8* null, align 4

define i8* @dummy(i8* %x) nounwind readnone noinline {
entry:
  ret i8* %x
}

; PIC-O32: ld  $[[R0:[0-9]+|t9]], %got($tmp0)($gp)
; PIC-O32: ori ${{[0-9]+|t9}}, $[[R0]], %lo($tmp0)
; PIC-O32: ld  $[[R1:[0-9]+|t9]], %got($tmp1)($gp)
; PIC-O32: ori ${{[0-9]+|t9}}, $[[R1]], %lo($tmp1)
; STATIC-O32: lui  $[[R2:[0-9]+|t9]], %hi($tmp[[T2:[0-9]+|t9]])
; STATIC-O32: ori ${{[0-9]+}}, $[[R2]], %lo($tmp[[T2]])
; STATIC-O32: lui   $[[R3:[0-9]+|t9]], %hi($tmp[[T3:[0-9]+|t9]])
; STATIC-O32: ori ${{[0-9]+}}, $[[R3]], %lo($tmp[[T3]])
define void @f() nounwind {
entry:
  %call = tail call i8* @dummy(i8* blockaddress(@f, %baz))
  indirectbr i8* %call, [label %baz, label %foo]

foo:                                              ; preds = %foo, %entry
  store i8* blockaddress(@f, %foo), i8** @reg, align 4
  br label %foo

baz:                                              ; preds = %entry
  store i8* null, i8** @reg, align 4
  ret void
}
