; RUN: llc -march=cpu0 -relocation-model=static < %s | FileCheck %s
; RUN: llc -march=cpu0 -relocation-model=pic < %s | FileCheck %s

@i = common global i32 0, align 4
@j = common global i32 0, align 4
@.str = private unnamed_addr constant [8 x i8] c"%i %i \0A\00", align 1

define void @foo(i32* %p, i32 %i, i32 %j) nounwind {
entry:
  %p.addr = alloca i32*, align 4
  %i.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  store i32* %p, i32** %p.addr, align 4
  store i32 %i, i32* %i.addr, align 4
  store i32 %j, i32* %j.addr, align 4
  %0 = load i32, i32* %j.addr, align 4
  %1 = load i32*, i32** %p.addr, align 4
  %2 = load i32, i32* %i.addr, align 4
  %add.ptr = getelementptr inbounds i32, i32* %1, i32 %2
  store i32 %0, i32* %add.ptr, align 4
  ret void
}

define i32 @main() nounwind {
entry:
; CHECK: main: 
; CHECK:            lui $[[T0:[0-9]+|t9]], 65524
; CHECK:            addiu $[[T0]], $[[T0]], -{{[0-9]+|t9}}
; CHECK:            addu $sp, $sp, $[[T0]]
; CHECK:            lui	$[[T1:[0-9]+|t9]], {{[0-9]+|t9}}
; CHECK:            ori	$[[T1]], $[[T1]], {{[0-9]+|t9}}
  %retval = alloca i32, align 4
  %one = alloca [100000 x i32], align 4
  %two = alloca [100000 x i32], align 4
  store i32 0, i32* %retval
  %arrayidx = getelementptr inbounds [100000 x i32], [100000 x i32]* %one, i32 0, i32 0
  call void @foo(i32* %arrayidx, i32 50, i32 9999)
  %arrayidx1 = getelementptr inbounds [100000 x i32], [100000 x i32]* %two, i32 0, i32 0
  call void @foo(i32* %arrayidx1, i32 99999, i32 5555)
  %arrayidx2 = getelementptr inbounds [100000 x i32], [100000 x i32]* %one, i32 0, i32 50
  %0 = load i32, i32* %arrayidx2, align 4
  store i32 %0, i32* @i, align 4
  %arrayidx3 = getelementptr inbounds [100000 x i32], [100000 x i32]* %two, i32 0, i32 99999
  %1 = load i32, i32* %arrayidx3, align 4
  store i32 %1, i32* @j, align 4
  %2 = load i32, i32* @i, align 4
  %3 = load i32, i32* @j, align 4
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i32 0, i32 0), i32 %2, i32 %3)
  ret i32 0
}

declare i32 @printf(i8*, ...)
