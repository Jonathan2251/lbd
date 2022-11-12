/* 
~/llvm/debug/build/bin/clang -target mips-unknown-linux-gnu -emit-llvm -S ch7_1_fmul.c
        ...
        %mul = fmul float %0, %1

~/llvm/debug/build/bin/llc -march=mips ch7_1_fmul.ll -relocation-model=static -o -
        ...
	v_log_f32_e32 v1, v0
	v_mul_legacy_f32_e32 v0, v0, v1
	v_exp_f32_e32 v0, v0

~/llvm/test/build/bin/llc -march=cpu0 ch7_1_fmul.ll -relocation-model=static -o -
         ...
        jsub __mulsf3
*/

float ch7_1_fmul(float a, float b) {
  float c = a * b;
  return c;
}
