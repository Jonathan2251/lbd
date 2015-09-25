; ModuleID = 'ch9_3-vararg.cpp'
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: nounwind
define i32 @_Z5sum_iiz(i32 %amount, ...) #0 {
  %vl = alloca i8*, align 4
  %1 = bitcast i8** %vl to i8*
  call void @llvm.va_start(i8* %1)
  %2 = icmp sgt i32 %amount, 0
  br i1 %2, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %.pre = load i8** %vl, align 4
  br label %3

; <label>:3                                       ; preds = %3, %.lr.ph
  %4 = phi i8* [ %.pre, %.lr.ph ], [ %6, %3 ]
  %sum.02 = phi i32 [ 0, %.lr.ph ], [ %8, %3 ]
  %i.01 = phi i32 [ 0, %.lr.ph ], [ %9, %3 ]
  %5 = bitcast i8* %4 to i32*
  %6 = getelementptr i8* %4, i32 4
  store i8* %6, i8** %vl, align 4
  %7 = load i32* %5, align 4
  %8 = add nsw i32 %7, %sum.02
  %9 = add nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %9, %amount
  br i1 %exitcond, label %._crit_edge.loopexit, label %3

._crit_edge.loopexit:                             ; preds = %3
  %.lcssa = phi i32 [ %8, %3 ]
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %.lcssa, %._crit_edge.loopexit ]
  call void @llvm.va_end(i8* %1)
  ret i32 %sum.0.lcssa
}

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) #1

; Function Attrs: nounwind
declare void @llvm.va_end(i8*) #1

; Function Attrs: nounwind
define i32 @_Z11test_varargv() #0 {
  %1 = tail call i32 (i32, ...)* @_Z5sum_iiz(i32 6, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5)
  ret i32 %1
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.5.0 (tags/RELEASE_350/final)"}
