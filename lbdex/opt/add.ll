; ~/llvm/3.9.0/release/cmake_debug_build/bin/opt -O3 -S add.ll -o -

define i32 @add(i32 %a, i32 %b) nounwind {
entry:
  %add = add i32 %a, %b
  ret i32 %add
}

define i32 @test() nounwind {
entry:
  %call1 = call i32 @add(i32 1, i32 2)
  ret i32 %call1
}
