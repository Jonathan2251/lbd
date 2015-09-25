; RUN: llc < %s -march=cpu0 -mcpu=cpu032II | FileCheck %s

define i64 @exchange_and_add(i64* %mem, i64 %val) nounwind {
; CHECK-LABEL: exchange_and_add:
; CHECK: __sync_fetch_and_add_8
  %tmp = atomicrmw add i64* %mem, i64 %val monotonic
  ret i64 %tmp
}

define i64 @exchange_and_cmp(i64* %mem) nounwind {
; CHECK-LABEL: exchange_and_cmp:
; CHECK: __sync_val_compare_and_swap_8
  %tmppair = cmpxchg i64* %mem, i64 0, i64 1 monotonic monotonic
  %tmp = extractvalue { i64, i1 } %tmppair, 0
  ret i64 %tmp
}

define i64 @exchange(i64* %mem, i64 %val) nounwind {
; CHECK-LABEL: exchange:
; CHECK: __sync_lock_test_and_set_8
  %tmp = atomicrmw xchg i64* %mem, i64 1 monotonic
  ret i64 %tmp
}

define void @atomic_store(i64* %mem, i64 %val) nounwind {
entry:
; CHECK-LABEL: @atomic_store
; CHECK: sync
; CHECK: __sync_lock_test_and_set_8
  store atomic i64 %val, i64* %mem release, align 64
  ret void
}

define i64 @atomic_load(i64* %mem) nounwind {
entry:
; CHECK-LABEL: @atomic_load
; CHECK: __sync_val_compare_and_swap_8
; CHECK: sync
  %tmp = load atomic i64, i64* %mem acquire, align 64
  ret i64 %tmp
}

