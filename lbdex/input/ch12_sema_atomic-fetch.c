// clang -S ch12_sema_atomic-fetch.c -emit-llvm -o -
// Uses /opt/homebrew/opt/llvm/bin/clang in macOS.

#include <stdatomic.h>

//#define WANT_COMPILE_FAIL

// From __c11_atomic_fetch_xxx of memory_checks() of Sema/atomic-ops.c
void memory_checks(_Atomic(int) *Ap, int *p, int val) {
  (void)__c11_atomic_fetch_add(Ap, 1, memory_order_relaxed);
  (void)__c11_atomic_fetch_add(Ap, 1, memory_order_acquire);
  (void)__c11_atomic_fetch_add(Ap, 1, memory_order_consume);
  (void)__c11_atomic_fetch_add(Ap, 1, memory_order_release);
  (void)__c11_atomic_fetch_add(Ap, 1, memory_order_acq_rel);
  (void)__c11_atomic_fetch_add(Ap, 1, memory_order_seq_cst);

#ifdef WANT_COMPILE_FAIL // fail to compile:
  (void)__c11_atomic_fetch_add(
      (struct Incomplete * _Atomic *)0, // expected-error {{incomplete type 'struct Incomplete'}}
      1, memory_order_seq_cst);
#endif

  (void)__c11_atomic_init(Ap, val);
  (void)__c11_atomic_init(Ap, val);
  (void)__c11_atomic_init(Ap, val);
  (void)__c11_atomic_init(Ap, val);
  (void)__c11_atomic_init(Ap, val);
  (void)__c11_atomic_init(Ap, val);

  (void)__c11_atomic_fetch_sub(Ap, val, memory_order_relaxed);
  (void)__c11_atomic_fetch_sub(Ap, val, memory_order_acquire);
  (void)__c11_atomic_fetch_sub(Ap, val, memory_order_consume);
  (void)__c11_atomic_fetch_sub(Ap, val, memory_order_release);
  (void)__c11_atomic_fetch_sub(Ap, val, memory_order_acq_rel);
  (void)__c11_atomic_fetch_sub(Ap, val, memory_order_seq_cst);

  (void)__c11_atomic_fetch_and(Ap, val, memory_order_relaxed);
  (void)__c11_atomic_fetch_and(Ap, val, memory_order_acquire);
  (void)__c11_atomic_fetch_and(Ap, val, memory_order_consume);
  (void)__c11_atomic_fetch_and(Ap, val, memory_order_release);
  (void)__c11_atomic_fetch_and(Ap, val, memory_order_acq_rel);
  (void)__c11_atomic_fetch_and(Ap, val, memory_order_seq_cst);

  (void)__c11_atomic_fetch_or(Ap, val, memory_order_relaxed);
  (void)__c11_atomic_fetch_or(Ap, val, memory_order_acquire);
  (void)__c11_atomic_fetch_or(Ap, val, memory_order_consume);
  (void)__c11_atomic_fetch_or(Ap, val, memory_order_release);
  (void)__c11_atomic_fetch_or(Ap, val, memory_order_acq_rel);
  (void)__c11_atomic_fetch_or(Ap, val, memory_order_seq_cst);

  (void)__c11_atomic_fetch_xor(Ap, val, memory_order_relaxed);
  (void)__c11_atomic_fetch_xor(Ap, val, memory_order_acquire);
  (void)__c11_atomic_fetch_xor(Ap, val, memory_order_consume);
  (void)__c11_atomic_fetch_xor(Ap, val, memory_order_release);
  (void)__c11_atomic_fetch_xor(Ap, val, memory_order_acq_rel);
  (void)__c11_atomic_fetch_xor(Ap, val, memory_order_seq_cst);

  (void)__c11_atomic_fetch_nand(Ap, val, memory_order_relaxed);
  (void)__c11_atomic_fetch_nand(Ap, val, memory_order_acquire);
  (void)__c11_atomic_fetch_nand(Ap, val, memory_order_consume);
  (void)__c11_atomic_fetch_nand(Ap, val, memory_order_release);
  (void)__c11_atomic_fetch_nand(Ap, val, memory_order_acq_rel);
  (void)__c11_atomic_fetch_nand(Ap, val, memory_order_seq_cst);

  (void)__c11_atomic_fetch_min(Ap, val, memory_order_relaxed);
  (void)__c11_atomic_fetch_min(Ap, val, memory_order_acquire);
  (void)__c11_atomic_fetch_min(Ap, val, memory_order_consume);
  (void)__c11_atomic_fetch_min(Ap, val, memory_order_release);
  (void)__c11_atomic_fetch_min(Ap, val, memory_order_acq_rel);
  (void)__c11_atomic_fetch_min(Ap, val, memory_order_seq_cst);

  (void)__c11_atomic_fetch_max(Ap, val, memory_order_relaxed);
  (void)__c11_atomic_fetch_max(Ap, val, memory_order_acquire);
  (void)__c11_atomic_fetch_max(Ap, val, memory_order_consume);
  (void)__c11_atomic_fetch_max(Ap, val, memory_order_release);
  (void)__c11_atomic_fetch_max(Ap, val, memory_order_acq_rel);
  (void)__c11_atomic_fetch_max(Ap, val, memory_order_seq_cst);
}
