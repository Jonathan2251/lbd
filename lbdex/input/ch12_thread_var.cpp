// clang -target mips-unknown-linux-gnu -c ch12_thread_var.cpp -emit-llvm -std=c++11 -o ch12_thread_var.bc
// ~/llvm/test/cmake_debug_build/bin/llvm-dis ch12_thread_var.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch12_thread_var.bc -o -
// ~/llvm/test/cmake_debug_build/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch12_thread_var.bc -o -

// ~/llvm/test/cmake_debug_build/Debug/bin/llvm-dis ch12_thread_var.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch12_thread_var.bc -o -
// ~/llvm/test/cmake_debug_build/Debug/bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch12_thread_var.bc -o -

/// start
__thread int a = 0;
thread_local int b = 0; // need option -std=c++11
int test_thread_var()
{
    a = 2;
    return a;
}

int test_thread_var_2()
{
    b = 3;
    return b;
}

