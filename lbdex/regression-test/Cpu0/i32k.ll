; RUN: llc  -march=cpu0 -relocation-model=pic < %s | FileCheck %s

@.str = private unnamed_addr constant [4 x i8] c"%i\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 1075344593) nounwind
; CHECK:	lui	${{[0-9]+|t9}}, 16408
; CHECK:	ori	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, 29905
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 -1075344593) nounwind
; CHECK:	lui	${{[0-9]+|t9}}, 49127
; CHECK:	ori	${{[0-9]+|t9}}, ${{[0-9]+|t9}}, 35631
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind
