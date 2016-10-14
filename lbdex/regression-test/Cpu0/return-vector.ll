; RUN: llc -march=cpu0 -mcpu=cpu032I -relocation-model=pic < %s | FileCheck %s


; Check that function accesses vector return value from stack in cases when
; vector can't be returned in registers. Also check that caller passes in
; register $4 stack address where the vector should be placed.


declare <8 x i32>    @i8(...)
declare <4 x float>  @f4(...)
declare <4 x double> @d4(...)

define i32 @call_i8() {
entry:
  %call = call <8 x i32> (...) @i8()
  %v0 = extractelement <8 x i32> %call, i32 0
  %v1 = extractelement <8 x i32> %call, i32 1
  %v2 = extractelement <8 x i32> %call, i32 2
  %v3 = extractelement <8 x i32> %call, i32 3
  %v4 = extractelement <8 x i32> %call, i32 4
  %v5 = extractelement <8 x i32> %call, i32 5
  %v6 = extractelement <8 x i32> %call, i32 6
  %v7 = extractelement <8 x i32> %call, i32 7
  %add1 = add i32 %v0, %v1
  %add2 = add i32 %v2, %v3
  %add3 = add i32 %v4, %v5
  %add4 = add i32 %v6, %v7
  %add5 = add i32 %add1, %add2
  %add6 = add i32 %add3, %add4
  %add7 = add i32 %add5, %add6
  ret i32 %add7

; CHECK:        call_i8:
; CHECK:        call16(i8)
; CHECK:        addiu   $4, $fp, 32
; CHECK:        ld      $[[R0:[a-z0-9]+]], 60($fp)
; CHECK:        ld      $[[R1:[a-z0-9]+]], 56($fp)
; CHECK:        ld      $[[R2:[a-z0-9]+]], 52($fp)
; CHECK:        ld      $[[R3:[a-z0-9]+]], 48($fp)
; CHECK:        ld      $[[R4:[a-z0-9]+]], 44($fp)
; CHECK:        ld      $[[R5:[a-z0-9]+]], 40($fp)
; CHECK:        ld      $[[R6:[a-z0-9]+]], 36($fp)
; CHECK:        ld      $[[R7:[a-z0-9]+]], 32($fp)
}


; Check that function accesses vector return value from registers in cases when
; vector can be returned in registers


declare <4 x i32>    @i4(...)
declare <2 x float>  @f2(...)
declare <2 x double> @d2(...)

define i32 @call_i4() {
entry:
  %call = call <4 x i32> (...) @i4()
  %v0 = extractelement <4 x i32> %call, i32 0
  %v1 = extractelement <4 x i32> %call, i32 1
  %v2 = extractelement <4 x i32> %call, i32 2
  %v3 = extractelement <4 x i32> %call, i32 3
  %add1 = add i32 %v0, %v1
  %add2 = add i32 %v2, %v3
  %add3 = add i32 %add1, %add2
  ret i32 %add3

; CHECK:        call_i4:
; CHECK:        call16(i4)
; CHECK:        addu    $[[R2:[a-z0-9]+]], $[[R0:[a-z0-9]+]], $[[R1:[a-z0-9]+]]
; CHECK:        addu    $[[R5:[a-z0-9]+]], $[[R3:[a-z0-9]+]], $[[R4:[a-z0-9]+]]
; CHECK:        addu    $[[R6:[a-z0-9]+]], $[[R5]], $[[R2]]
}


; Check that function returns vector on stack in cases when vector can't be
; returned in registers. Also check that vector is placed on stack starting
; from the address in register $4.


define <8 x i32> @return_i8() {
entry:
  ret <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>

; CHECK:        return_i8:
; CHECK:        st      $[[R0:[a-z0-9]+]], 28($4)
; CHECK:        st      $[[R1:[a-z0-9]+]], 24($4)
; CHECK:        st      $[[R2:[a-z0-9]+]], 20($4)
; CHECK:        st      $[[R3:[a-z0-9]+]], 16($4)
; CHECK:        st      $[[R4:[a-z0-9]+]], 12($4)
; CHECK:        st      $[[R5:[a-z0-9]+]], 8($4)
; CHECK:        st      $[[R6:[a-z0-9]+]], 4($4)
; CHECK:        st      $[[R7:[a-z0-9]+]], 0($4)
}


define <4 x float> @return_f4(float %a, float %b, float %c, float %d) {
entry:
  %vecins1 = insertelement <4 x float> undef,    float %a, i32 0
  %vecins2 = insertelement <4 x float> %vecins1, float %b, i32 1
  %vecins3 = insertelement <4 x float> %vecins2, float %c, i32 2
  %vecins4 = insertelement <4 x float> %vecins3, float %d, i32 3
  ret <4 x float> %vecins4

; CHECK:        return_f4:
; CHECK:        addu	$2, $zero, $4
; CHECK:        addu	$3, $zero, $5
; CHECK:        addu	$4, $zero, ${{[0-9]+|t9}}
; CHECK:        addu	$5, $zero, ${{[0-9]+|t9}}
}


define <4 x double> @return_d4(double %a, double %b, double %c, double %d) {
entry:
  %vecins1 = insertelement <4 x double> undef,    double %a, i32 0
  %vecins2 = insertelement <4 x double> %vecins1, double %b, i32 1
  %vecins3 = insertelement <4 x double> %vecins2, double %c, i32 2
  %vecins4 = insertelement <4 x double> %vecins3, double %d, i32 3
  ret <4 x double> %vecins4

; CHECK:        return_d4:
; CHECK:        st      $[[R0:[a-z0-9]+]], 28($4)
; CHECK:        st      $[[R1:[a-z0-9]+]], 24($4)
; CHECK:        st      $[[R2:[a-z0-9]+]], 20($4)
; CHECK:        st      $[[R3:[a-z0-9]+]], 16($4)
; CHECK:        st      $[[R4:[a-z0-9]+]], 12($4)
; CHECK:        st      $[[R5:[a-z0-9]+]], 8($4)
; CHECK:        st      $[[R6:[a-z0-9]+]], 4($4)
; CHECK:        st      $[[R7:[a-z0-9]+]], 0($4)
}



; Check that function returns vector in registers in cases when vector can be
; returned in registers.


define <4 x i32> @return_i4() {
entry:
  ret <4 x i32> <i32 0, i32 1, i32 2, i32 3>

; CHECK:        return_i4:
; CHECK:        addiu   $2, $zero, 0
; CHECK:        addiu   $3, $zero, 1
; CHECK:        addiu   $4, $zero, 2
; CHECK:        addiu   $5, $zero, 3
}


define <2 x float> @return_f2(float %a, float %b) {
entry:
  %vecins1 = insertelement <2 x float> undef,    float %a, i32 0
  %vecins2 = insertelement <2 x float> %vecins1, float %b, i32 1
  ret <2 x float> %vecins2

; CHECK:        return_f2:
; CHECK:        addu	$2, $zero, $4
; CHECK:        addu	$3, $zero, $5
}


define <2 x double> @return_d2(double %a, double %b) {
entry:
  %vecins1 = insertelement <2 x double> undef,    double %a, i32 0
  %vecins2 = insertelement <2 x double> %vecins1, double %b, i32 1
  ret <2 x double> %vecins2

; CHECK:        return_d2:
; CHECK:        addu	$2, $zero, $4
; CHECK:        addu	$3, $zero, $5
; CHECK:        addu	$4, $zero, ${{[0-9]+|t9}}
; CHECK:        addu	$5, $zero, ${{[0-9]+|t9}}
}
