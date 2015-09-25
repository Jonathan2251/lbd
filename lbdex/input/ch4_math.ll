
; Function Attrs: nounwind
define i32 @_Z9test_mathv() #0 {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %1 = load i32* %a, align 4
  %2 = load i32* %b, align 4

  %3 = add nsw i32 %1, %2
  %4 = sub nsw i32 %1, %2
  %5 = mul nsw i32 %1, %2
  %6 = shl i32 %1, 2
  %7 = ashr i32 %1, 2
  %8 = lshr i32 %1, 30
  %9 = shl i32 1, %2
  %10 = ashr i32 128, %2
  %11 = ashr i32 %1, %2

  %12 = add nsw i32 %3, %4
  %13 = add nsw i32 %12, %5
  %14 = add nsw i32 %13, %6
  %15 = add nsw i32 %14, %7
  %16 = add nsw i32 %15, %8
  %17 = add nsw i32 %16, %9
  %18 = add nsw i32 %17, %10
  %19 = add nsw i32 %18, %11

  ret i32 %19
}

