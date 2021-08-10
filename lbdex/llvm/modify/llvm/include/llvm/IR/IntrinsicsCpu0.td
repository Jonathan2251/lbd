//===- IntrinsicsCpu0.td - Defines Mips intrinsics ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the CPU0-specific intrinsics.
//
//===----------------------------------------------------------------------===//

def int_cpu0_gcd : GCCBuiltin<"__builtin_cpu0_gcd">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
  [Commutative, IntrNoMem]>;

