; RUN: llc  -march=cpu0el -relocation-model=pic < %s | FileCheck %s -check-prefix=1


@x = common global float 0.000000e+00, align 4
@xd = common global double 0.000000e+00, align 8
@y = common global float 0.000000e+00, align 4
@yd = common global double 0.000000e+00, align 8
@xy = common global { float, float } zeroinitializer, align 4
@xyd = common global { double, double } zeroinitializer, align 8

define void @foo() nounwind {
entry:
  %0 = load float, float* @x, align 4
  call void @v_sf(float %0)
  %1 = load double, double* @xd, align 8
  call void @v_df(double %1)
  %2 = load float, float* @x, align 4
  %3 = load float, float* @y, align 4
  call void @v_sf_sf(float %2, float %3)
  %4 = load double, double* @xd, align 8
  %5 = load float, float* @x, align 4
  call void @v_df_sf(double %4, float %5)
  %6 = load double, double* @xd, align 8
  %7 = load double, double* @yd, align 8
  call void @v_df_df(double %6, double %7)
  %call = call float @sf_v()
  %8 = load float, float* @x, align 4
  %call1 = call float @sf_sf(float %8)
  %9 = load double, double* @xd, align 8
  %call2 = call float @sf_df(double %9)
  %10 = load float, float* @x, align 4
  %11 = load float, float* @y, align 4
  %call3 = call float @sf_sf_sf(float %10, float %11)
  %12 = load double, double* @xd, align 8
  %13 = load float, float* @x, align 4
  %call4 = call float @sf_df_sf(double %12, float %13)
  %14 = load double, double* @xd, align 8
  %15 = load double, double* @yd, align 8
  %call5 = call float @sf_df_df(double %14, double %15)
  %call6 = call double @df_v()
  %16 = load float, float* @x, align 4
  %call7 = call double @df_sf(float %16)
  %17 = load double, double* @xd, align 8
  %call8 = call double @df_df(double %17)
  %18 = load float, float* @x, align 4
  %19 = load float, float* @y, align 4
  %call9 = call double @df_sf_sf(float %18, float %19)
  %20 = load double, double* @xd, align 8
  %21 = load float, float* @x, align 4
  %call10 = call double @df_df_sf(double %20, float %21)
  %22 = load double, double* @xd, align 8
  %23 = load double, double* @yd, align 8
  %call11 = call double @df_df_df(double %22, double %23)
  %call12 = call { float, float } @sc_v()
  %24 = extractvalue { float, float } %call12, 0
  %25 = extractvalue { float, float } %call12, 1
  %26 = load float, float* @x, align 4
  %call13 = call { float, float } @sc_sf(float %26)
  %27 = extractvalue { float, float } %call13, 0
  %28 = extractvalue { float, float } %call13, 1
  %29 = load double, double* @xd, align 8
  %call14 = call { float, float } @sc_df(double %29)
  %30 = extractvalue { float, float } %call14, 0
  %31 = extractvalue { float, float } %call14, 1
  %32 = load float, float* @x, align 4
  %33 = load float, float* @y, align 4
  %call15 = call { float, float } @sc_sf_sf(float %32, float %33)
  %34 = extractvalue { float, float } %call15, 0
  %35 = extractvalue { float, float } %call15, 1
  %36 = load double, double* @xd, align 8
  %37 = load float, float* @x, align 4
  %call16 = call { float, float } @sc_df_sf(double %36, float %37)
  %38 = extractvalue { float, float } %call16, 0
  %39 = extractvalue { float, float } %call16, 1
  %40 = load double, double* @xd, align 8
  %41 = load double, double* @yd, align 8
  %call17 = call { float, float } @sc_df_df(double %40, double %41)
  %42 = extractvalue { float, float } %call17, 0
  %43 = extractvalue { float, float } %call17, 1
  %call18 = call { double, double } @dc_v()
  %44 = extractvalue { double, double } %call18, 0
  %45 = extractvalue { double, double } %call18, 1
  %46 = load float, float* @x, align 4
  %call19 = call { double, double } @dc_sf(float %46)
  %47 = extractvalue { double, double } %call19, 0
  %48 = extractvalue { double, double } %call19, 1
  %49 = load double, double* @xd, align 8
  %call20 = call { double, double } @dc_df(double %49)
  %50 = extractvalue { double, double } %call20, 0
  %51 = extractvalue { double, double } %call20, 1
  %52 = load float, float* @x, align 4
  %53 = load float, float* @y, align 4
  %call21 = call { double, double } @dc_sf_sf(float %52, float %53)
  %54 = extractvalue { double, double } %call21, 0
  %55 = extractvalue { double, double } %call21, 1
  %56 = load double, double* @xd, align 8
  %57 = load float, float* @x, align 4
  %call22 = call { double, double } @dc_df_sf(double %56, float %57)
  %58 = extractvalue { double, double } %call22, 0
  %59 = extractvalue { double, double } %call22, 1
  %60 = load double, double* @xd, align 8
  %61 = load double, double* @yd, align 8
  %call23 = call { double, double } @dc_df_df(double %60, double %61)
  %62 = extractvalue { double, double } %call23, 0
  %63 = extractvalue { double, double } %call23, 1
  ret void
}

declare void @v_sf(float)

declare void @v_df(double)

declare void @v_sf_sf(float, float)

declare void @v_df_sf(double, float)

declare void @v_df_df(double, double)

declare float @sf_v()

declare float @sf_sf(float)

declare float @sf_df(double)

declare float @sf_sf_sf(float, float)

declare float @sf_df_sf(double, float)

declare float @sf_df_df(double, double)

declare double @df_v()

declare double @df_sf(float)

declare double @df_df(double)

declare double @df_sf_sf(float, float)

declare double @df_df_sf(double, float)

declare double @df_df_df(double, double)

declare { float, float } @sc_v()

declare { float, float } @sc_sf(float)

declare { float, float } @sc_df(double)

declare { float, float } @sc_sf_sf(float, float)

declare { float, float } @sc_df_sf(double, float)

declare { float, float } @sc_df_df(double, double)

declare { double, double } @dc_v()

declare { double, double } @dc_sf(float)

declare { double, double } @dc_df(double)

declare { double, double } @dc_sf_sf(float, float)

declare { double, double } @dc_df_sf(double, float)

declare { double, double } @dc_df_df(double, double)

; 1:	ld	$t9, %call16(v_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(v_df)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(v_sf_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(v_df_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(v_df_df)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sf_v)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sf_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sf_df)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sf_sf_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sf_df_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(df_v)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(df_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(df_df)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(df_sf_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(df_df_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sc_v)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sc_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sc_df)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sc_sf_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sc_df_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(sc_df_df)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(dc_v)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(dc_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(dc_df)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(dc_sf_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(dc_df_sf)($gp)
; 1:	jalr	$t9

; 1:	ld	$t9, %call16(dc_df_df)($gp)
; 1:	jalr	$t9

