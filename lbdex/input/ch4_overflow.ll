
; Function Attrs: nounwind
define i32 @_Z17test_add_overflowv() #0 {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %1 = load i32* %a, align 4
  %2 = load i32* %b, align 4

  %3 = add nsw i32 %1, %2
  %4 = sub nsw i32 %1, %2

  %5 = add nsw i32 %3, %4
  ret i32 %5
}

