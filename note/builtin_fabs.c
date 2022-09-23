// ~/llvm/debug/build/bin/clang -emit-llvm -S builtin_fabs.c -o -

// CHECK-LABEL: define{{.*}} void @test_float_builtin_ops
void test_float_builtin_ops(float F, double D, long double LD) {
  volatile float resf;
  volatile double resd;
  volatile long double resld;

  resf = __builtin_fabsf(F);
  resd = __builtin_fabs(D);
  resld = __builtin_fabsl(LD);
  // CHECK: call float @llvm.fabs.f32(float
  // CHECK: call double @llvm.fabs.f64(double
  // CHECK: call x86_fp80 @llvm.fabs.f80(x86_fp80
}
