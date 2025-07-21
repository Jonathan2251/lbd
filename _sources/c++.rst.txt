.. _sec-c++:

C++ support
=============

.. contents::
   :local:
   :depth: 4

This chapter supports some C++ compiler features.

Exception Handling
------------------

Chapter11_2 can be built and run using the C++ polymorphism example code in
``ch12_inherit.cpp`` as follows:

.. rubric:: lbdex/input/ch12_inherit.cpp
.. code-block:: c++

  ...
  class CPolygon { // _ZTVN10__cxxabiv117__class_type_infoE for parent class
    ...
  #ifdef COUT_TEST
   // generate IR nvoke, landing, resume and unreachable on iMac
      { cout << this->area() << endl; }
  #else
      { printf("%d\n", this->area()); }
  #endif
  };
  ...

If you use ``cout`` instead of ``printf`` in ``ch12_inherit.cpp``, it will not
generate exception handling IR on Linux. However, it will generate exception
handling IRs such as ``invoke``, ``landingpad``, ``resume``, and
``unreachable`` on iMac.

The example code ``ch12_eh.cpp``, which includes **try** and **catch**
exception handling, will generate these exception-related IRs on both iMac
and Linux.

.. rubric:: lbdex/input/ch12_eh.cpp
.. literalinclude:: ../lbdex/input/ch12_eh.cpp
    :start-after: /// start

.. code-block:: console

  JonathantekiiMac:input Jonathan$ clang -c ch12_eh.cpp -emit-llvm 
  -o ch12_eh.bc
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llvm-dis ch12_eh.bc -o -
  
.. rubric:: ../lbdex/output/ch12_eh.ll
.. code-block:: llvm

  ...
  define dso_local i32 @_Z14test_try_catchv() #0 personality i8* bitcast (i32 (...
  )* @__gxx_personality_v0 to i8*) {
  entry:
    ...
    invoke void @_Z15throw_exceptionii(i32 signext 2, i32 signext 1)
          to label %invoke.cont unwind label %lpad

  invoke.cont:                                      ; preds = %entry
    br label %try.cont

  lpad:                                             ; preds = %entry
    %0 = landingpad { i8*, i32 }
            catch i8* null
    ...
  }
  ...

.. code:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch12_eh.bc -o -
	  .section .mdebug.abi32
	  .previous
	  .file	"ch12_eh.bc"
  llc: /Users/Jonathan/llvm/test/llvm/lib/CodeGen/LiveVariables.cpp:133: void llvm::
  LiveVariables::HandleVirtRegUse(unsigned int, llvm::MachineBasicBlock *, llvm
  ::MachineInstr *): Assertion `MRI->getVRegDef(reg) && "Register use before 
  def!"' failed.

A description of the C++ exception table formats can be found here
[#itanium-exception]_.

For details about the LLVM IR used in exception handling, please refer to
[#exception]_.

Chapter12_1 supports the LLVM IRs that correspond to the C++ **try** and
**catch** keywords. It can compile ``ch12_eh.bc`` as follows:

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //5
    :end-before: #endif

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=static -filetype=asm ch12_eh.bc -o -
  
.. rubric:: ../lbdex/output/ch12_eh.cpu0.s
.. code:: text

    .type  _Z14test_try_catchv,@function
    .ent  _Z14test_try_catchv             # @_Z14test_try_catchv
  _Z14test_try_catchv:
    ...
  $tmp0:
    addiu  $4, $zero, 2
    addiu  $5, $zero, 1
    jsub  _Z15throw_exceptionii
    nop
  $tmp1:
  # %bb.1:                                # %invoke.cont
    jmp  $BB1_4
  $BB1_2:                                 # %lpad
  $tmp2:
    st  $4, 16($fp)
    st  $5, 12($fp)
  # %bb.3:                                # %catch
    ld  $4, 16($fp)
    jsub  __cxa_begin_catch
    nop
    addiu  $2, $zero, 1
    st  $2, 20($fp)
    jsub  __cxa_end_catch
    nop
    jmp  $BB1_5
  $BB1_4:                                 # %try.cont
    addiu  $2, $zero, 0
    st  $2, 20($fp)
  $BB1_5:                                 # %return
    ld  $2, 20($fp)
    ...


Thread variable
-------------------

C++ support thread variable as the following file ch12_thread_var.cpp.

.. rubric:: lbdex/input/ch12_thread_var.cpp
.. literalinclude:: ../lbdex/input/ch12_thread_var.cpp
    :start-after: /// start

While a global variable is a single instance shared by all threads in a process,
a thread-local variable has a separate instance for each thread in the process.
The same thread accesses the same instance of the thread-local variable, while
different threads have their own instances with the same variable name
[#thread-wiki]_.

To support thread-local variables, symbols such as **tlsgd**, **tlsldm**,
**dtp_hi**, **dtp_lo**, **gottp**, **tp_hi**, and **tp_lo** must be handled in
both `evaluateRelocExpr()` of `Cpu0AsmParser.cpp` and `printImpl()` of
`Cpu0MCExpr.cpp`.

Most of these symbols are used for relocation record handling,
because the actual thread-local storage is created by the OS or language
runtime that supports multi-threaded programming.

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0AsmBackend.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0AsmBackend.cpp
    :start-after: //@getFixupKindInfo {
    :end-before: { "fixup_Cpu0_32",             0,     32,   0 },
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0AsmBackend.cpp
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

      ...
    };
    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0BaseInfo.h
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0BaseInfo.h
    :start-after: //@Cpu0II
    :end-before: MO_NO_FLAG,
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0BaseInfo.h
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

      ...
    };
    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0ELFObjectWriter.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0ELFObjectWriter.cpp
    :start-after: //@GetRelocType {
    :end-before: default:
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0ELFObjectWriter.cpp
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0FixupKinds.h
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0FixupKinds.h
    :start-after: //@Fixups {
    :end-before: //@ Pure upper 32 bit fixup resulting in - R_CPU0_32.
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0FixupKinds.h
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

      ...
    };

.. rubric:: lbdex/chapters/Chapter12_1/MCTargetDesc/Cpu0MCCodeEmitter.cpp
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: //@getExprOpValue {
    :end-before: //@getExprOpValue body {
.. literalinclude:: ../lbdex/Cpu0/MCTargetDesc/Cpu0MCCodeEmitter.cpp
    :start-after: #if CH >= CH12_1
    :end-before: #endif
	
.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 3.1
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 4
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0SelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif
	
.. code-block:: c++

    ...
  }
  
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #if CH >= CH8_1 //6
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //3
    :end-before: #endif
	
.. code-block:: c++

      ...
    }
    ...
  }
  
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //4
    :end-before: #endif


.. rubric:: lbdex/chapters/Chapter12_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0MCInstLower.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: //@LowerSymbolOperand {
    :end-before: default:
.. literalinclude:: ../lbdex/Cpu0/Cpu0MCInstLower.cpp
    :start-after: #if CH >= CH12_1
    :end-before: #endif
    
.. code-block:: c++

    ...
    }
    ...
  }


.. code-block:: console

  JonathantekiiMac:input Jonathan$ clang -target mips-unknown-linux-gnu -c 
  ch12_thread_var.cpp -emit-llvm -std=c++11 -o ch12_thread_var.bc
  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llvm-dis ch12_thread_var.bc -o -
  
.. rubric:: ../lbdex/output/ch12_thread_var.ll
.. code-block:: llvm

  ...
  @a = dso_local thread_local global i32 0, align 4
  @b = dso_local thread_local global i32 0, align 4

  ; Function Attrs: noinline nounwind optnone mustprogress
  define dso_local i32 @_Z15test_thread_varv() #0 {
  entry:
    store i32 2, i32* @a, align 4
    %0 = load i32, i32* @a, align 4
    ret i32 %0
  }

  ; Function Attrs: noinline nounwind optnone mustprogress
  define dso_local i32 @_Z17test_thread_var_2v() #0 {
  entry:
    store i32 3, i32* @b, align 4
    %0 = load i32, i32* @b, align 4
    ret i32 %0
  }
  ...

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=pic -filetype=asm ch12_thread_var.bc 
  -o ch12_thread_var.cpu0.pic.s
  JonathantekiiMac:input Jonathan$ cat ch12_thread_var.cpu0.pic.s
  
.. rubric:: ../lbdex/output/ch12_thread_var.cpu0.pic.s
.. code-block:: text

  ...
    .ent  _Z15test_thread_varv            # @_Z15test_thread_varv
  _Z15test_thread_varv:
    ...
    ori  $4, $gp, %tlsldm(a)
    ld  $t9, %call16(__tls_get_addr)($gp)
    jalr  $t9
    nop
    ld  $gp, 8($fp)
    lui  $3, %dtp_hi(a)
    addu  $2, $3, $2
    ori  $2, $2, %dtp_lo(a)
    ...

In PIC (Position-Independent Code) mode, the `__thread` variable is accessed by
calling the function `__tls_get_addr` with the address of the thread-local
variable as an argument.

For C++11 `thread_local` variables, the compiler generates a call to the function
`_ZTW1b`, which internally calls `__tls_get_addr` to retrieve the address of the
`thread_local` variable.

In static mode, thread-local variables are accessed directly by loading their
addresses using machine instructions. For example, variables `a` and `b` are
accessed through direct address calculation instructions.

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=cpu0 -relocation-model=static -filetype=asm 
  ch12_thread_var.bc -o ch12_thread_var.cpu0.static.s
  JonathantekiiMac:input Jonathan$ cat ch12_thread_var.cpu0.static.s
  
.. rubric:: ../lbdex/output/ch12_thread_var.cpu0.static.s
.. code-block:: text

    ...
    lui  $2, %tp_hi(a)
    ori  $2, $2, %tp_lo(a)
    ...
    lui  $2, %tp_hi(b)
    ori  $2, $2, %tp_lo(b)
    ...

While MIPS uses the `rdhwr` instruction to access thread-local variables, Cpu0
accesses thread-local variables without introducing any new instructions.

Thread-local variables in Cpu0 are stored in a dedicated thread-local memory
region, which is accessed through `%tp_hi` and `%tp_lo`. This memory section is
protected by the kernel, meaning it can only be accessed in kernel mode.

As a result, user-mode programs cannot access this memory region, leaving no room
for potential exploits or malicious programs to interfere with thread-local
storage.

.. code-block:: console

  JonathantekiiMac:input Jonathan$ /Users/Jonathan/llvm/test/build/
  bin/llc -march=mips -relocation-model=static -filetype=asm 
  ch12_thread_var.bc -o -
    ...
    lui $1, %tprel_hi(a)
    ori $1, $1, %tprel_lo(a)
    .set  push
    .set  mips32r2
    rdhwr $3, $29
    .set  pop
    addu  $1, $3, $1
    addiu $2, $zero, 2
    sw  $2, 0($1)
    addiu $2, $zero, 2
    ...

In static mode, the thread variable is similar to global variable. 
In general, they are same in IRs, DAGs and machine code translation. 
List them in the following tables. 
You can check them with debug option enabled.


In static mode, the thread variable behaves similarly to a global variable.
In general, they are the same in terms of LLVM IR, DAG, and machine code
translation.

You can refer to the following tables for a detailed comparison.

To observe this in action, compile and check with debug options enabled.

.. table:: The DAGs of thread varaible of static mode

  ==========================  ===========================
  stage                       DAG
  ==========================  ===========================
  IR                          load i32* @a, align 4;
  Legalized selection DAG     (add Cpu0ISD::Hi Cpu0ISD::Lo);
  Instruction Selection       ori $2, $zero, %tp_lo(a);
  -                           lui $3, %tp_hi(a);
  -                           addu  $3, $3, $2;
  ==========================  ===========================

.. table:: The DAGs of local_thread varaible of static mode

  ==========================  ===========================
  stage                       DAG
  ==========================  ===========================
  IR                          ret i32* @b;
  Legalized selection DAG     %0=(add Cpu0ISD::Hi Cpu0ISD::Lo);...
  Instruction Selection       ori $2, $zero, %tp_lo(a);
  -                           lui $3, %tp_hi(a);
  -                           addu  $3, $3, $2;
  ==========================  ===========================


C++ Memory Order [#cpp-mem-order]_ [#mem-order-wiki]_ [#atomic-stackoverflow]_ 
------------------------------------------------------------------------------

**Memory Order::**

- **Memory Order is the rules that define how operations on shared memory 
  appear to multiple threads — especially the ordering of reads/writes.**

- **These rules ensure correct execution in parallel programs, as demonstrated 
  in the example below.**

.. rubric:: 🧠 Example (Wrong Without Memory Order)
.. code-block:: c++

  volatile bool ready = false;
  ...

  // Thread A
  x = 42;
  ready = true;

  // Thread B
  if (ready)
    std::cout << x; // May print garbage if `x = 42` is reordered!

.. rubric:: 💡 Fixing with Memory Order
.. code-block:: c++

  std::atomic<bool> ready = false;
  ...

  // Thread A
  x.store(42, std::memory_order_relaxed);
  ready.store(true, std::memory_order_release);

  // Thread B
  if (ready.load(std::memory_order_acquire))
    print(x.load(std::memory_order_relaxed));

- memory_order_release tells the compiler/CPU: “All writes before this must be completed first.”
- memory_order_acquire tells the reader: “No reads after this can move before it.”

Now x = 42 is guaranteed to happen before ready = true, as seen by Thread B.

Background
~~~~~~~~~~

Before **C++11**, multi-threaded programming relied on **mutexes, volatile
variables, and platform-specific atomic operations (such as atomic_load(&a) and 
atomic_store(&a, 42)**, which often led to
inefficiencies and undefined behavior.

For RISC CPUs, **only load/store instructions access memory.** Atomic
instructions ensure memory consistency across multiple cores.

CPUs provide atomic operations such as **compare-and-swap** [#cas-wiki]_ or
**ll/sc (load-linked/store-conditional)**, along with **BARRIER** or **SYNC**
instructions to enforce memory ordering. However, **C++03 did not have a language
feature to tell the compiler how to control memory order for load/store
instructions.**

To address this, **C++11 introduced memory orderings via `std::atomic`**, giving
programmers **fine-grained control** over synchronization and memory consistency.

The Problem Before C++11
~~~~~~~~~~~~~~~~~~~~~~~~

**No standard atomic operations → non-portable.**

- Developers had to use compiler-specific intrinsics (like GCC’s __sync_*, 
  MSVC’s Interlocked*) or inline assembly, making code non-portable.

C++11 Memory Model Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: 

   **Non-blocking algorithm:**
   In computer science, an algorithm is called non-blocking if failure or 
   suspension of any thread cannot cause failure or suspension of another thread.

   If a suspended thread can temporarily release its mutex and reacquire it 
   upon resuming, while always producing correct results, then the algorithm is
   non-blocking.

   **Wait-free: no starvation.**
   
   An algorithm is wait-free if every operation has a bound on the 
   number of steps the algorithm will take before the operation completes. In 
   other words, wait-free algorithm has no starvation.

   **Lock-free: progressive, allow starvation.**
   
   Lock-freedom allows 
   individual threads to starve but guarantees system-wide throughput. An 
   algorithm is lock-free if, when the program threads are run for a 
   sufficiently long time, at least one of the threads makes progress (for 
   some sensible definition of progress). 

   All wait-free algorithms are lock-free. 
   In particular, if one thread is suspended, then a lock-free algorithm guarantees 
   that the remaining threads can still make progress. Hence, if two threads can 
   contend for the same mutex lock or spinlock, then the algorithm is not lock-free. 
   (If we suspend one thread that holds the lock, then the second thread will block.) 
   [#lf-wiki]_.

   Based on this definition, any algorithm that retains a mutex without 
   allowing temporary release does not qualify as lock-free.
   
C++11 introduced **a well-defined memory model** and **atomic operations** with 
**memory orderings**, allowing programmers to control hardware-level 
optimizations.

.. table:: C++ memory order A

   ================================================  ===============================================  ========================
   **Feature**                                       **Description**                                  **Benefit**
   ================================================  ===============================================  ========================
   `std::atomic`                                     Provides lock-free atomic variables              Faster than mutexes
   Memory Orderings (`std::memory_order`)            Controls instruction reordering                  Fine-grained optimization
   Sequential Consistency (`memory_order_seq_cst`)   Strongest ordering, default behavior             Prevents race conditions
   Acquire-Release (`memory_order_acquire/release`)  Synchronization without mutexes                  Efficient producer-consumer
   Relaxed Ordering (`memory_order_relaxed`)         Allows reordering for performance                Best for atomic counters
   ================================================  ===============================================  ========================

.. table:: C++ memory order B

   ============================  ==================================================   ================================================
   **Memory Order**              **Description**                                      **Use Cases**
   ============================  ==================================================   ================================================
   `memory_order_relaxed`        No ordering guarantees; only atomicity.              Non-dependent atomic counters, statistics.
   `memory_order_consume`        Data-dependent ordering (deprecated in practice).    Rarely used; intended for pointer chains.
   `memory_order_acquire`        Ensures preceding reads/writes are visible.          Locks, consumer threads.
   `memory_order_release`        Ensures following reads/writes are visible.          Locks, producer threads.
   `memory_order_acq_rel`        Combines acquire + release.                          Read-modify-write operations, synchronization.
   `memory_order_seq_cst`        Strongest ordering; global sequential consistency.   Default behavior, safest but can be slow.
   ============================  ==================================================   ================================================


Explanation:

1. **Sequential Consistency (`memory_order_seq_cst`)**

   - **Prevents reordering globally.**  
   - **Ensures** all threads observe operations in the same order.  
   - **Default behavior** of `std::atomic`.
   - Provides **global order of operations**, preventing out-of-order execution.  
   - The safest but can cause **performance overhead**.

2. **Acquire-Release (`memory_order_acquire/release`)**

   - **Efficient alternative to mutexes.**  
   - `acquire`: Ensures earlier loads are visible.  
   - `release`: Ensures later stores are visible.  

3. **Relaxed Ordering (`memory_order_relaxed`)**

   - Allows **maximum performance** without ordering constraints.  
   - Best for **counters and statistics** that don’t need synchronization.
   - Example: **Atomic counters** that don’t require ordering.

4. **`memory_order_acq_rel`**

  - Used in **atomic read-modify-write operations** like `fetch_add`.  
  - Ensures proper ordering in concurrent updates.

Summary:

- Use **`memory_order_relaxed`** for **performance** when ordering is 
  unnecessary.
- Use **`memory_order_acquire/release`** for **synchronization** between 
  threads.
- Use **`memory_order_seq_cst`** when you need **global ordering but at a 
  performance cost**.

Example code for producer-consumer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example code of C++ code for producer-consumer as follows,

.. rubric:: References/c++/mem-order-ex1.cpp (C++ code of memory order for 
            producer-consumer)
.. literalinclude:: ../References/c++/mem-order-ex1.cpp

Explaining as :numref:`mem_o_hw` and follows,

.. _mem_o_hw:
.. graphviz:: ../Fig/gpu/mem-o-hw.gv
  :caption: Diagram for mem-order-ex1.cpp

Diagram Explanation:

- **CPU Core 1 (Thread 1 - Producer)**

  - Writes 42 to data with memory_order_relaxed (not immediately visible).
  - Writes true to ready with memory_order_release, ensuring prior stores are 
    visible before this write.

- **Main Memory**

  - ready=true propagates to main memory, making it visible to all cores.

- **CPU Core 2 (Thread 2 - Consumer)**

  - Waits until ready=true with memory_order_acquire, ensuring visibility of all
    previous writes.
  - After acquiring ready, loads data, which is now guaranteed to be 42.


Cpu0 implementation for memory-order
------------------------------------

In order to support atomic in C++ and java, llvm provides the atomic IRs and 
memory ordering here [#atomics-llvm]_ [#llvmlang-ordering]_.

The chapter 19 
of book DPC++ [#dpcpp-memorder]_ explains the memory ordering better and I add 
the related code fragment of lbdex/input/atomics.ll to it for explanation as 
follows,

- memory_order::relaxed

Read and write operations can be re-ordered before or after the operation with 
no restrictions. There are no ordering guarantees.

.. code-block:: console
  
  define i8 @load_i8_unordered(i8* %mem) {
  ; CHECK-LABEL: load_i8_unordered
  ; CHECK: ll
  ; CHECK: sc
  ; CHECK-NOT: sync
    %val = load atomic i8, i8* %mem unordered, align 1
    ret i8 %val
  }

No **sync** from CodeGen instructions above.

- memory_order::acquire

Read and write operations appearing after the operation in the program must 
occur after it (i.e., they cannot be re-ordered before the operation).

.. code-block:: console

  define i32 @load_i32_acquire(i32* %mem) {
  ; CHECK-LABEL: load_i32_acquire
  ; CHECK: ll
  ; CHECK: sc
    %val = load atomic i32, i32* %mem acquire, align 4
  ; CHECK: sync
    ret i32 %val
  }

Sync guarantees "load atomic" complete before the next R/W (Read/Write). All 
writes in other threads that release the same atomic variable are visible in the 
current thread.

- memory_order::release

Read and write operations appearing before the operation in the program must 
occur before it (i.e., they cannot be re-ordered after the operation), and 
preceding write operations are guaranteed to be visible to other program 
instances which have been synchronized by a corresponding acquire operation 
(i.e., an atomic operation using the same variable and memory_order::acquire 
or a barrier function).

.. code-block:: console

  define void @store_i32_release(i32* %mem) {
  ; CHECK-LABEL: store_i32_release
  ; CHECK: sync
  ; CHECK: ll
  ; CHECK: sc
    store atomic i32 42, i32* %mem release, align 4
    ret void
  }

Sync guarantees preceding R/W complete before "store atomic". Mips' ll and sc 
guarantee that "store atomic release" is visible to other processors.

- memory_order::acq_rel

The operation acts as both an acquire and a release. Read and write operations 
cannot be re-ordered around the operation, and preceding writes must be made 
visible as previously described for memory_order::release.

.. code-block:: console

  define i32 @cas_strong_i32_acqrel_acquire(i32* %mem) {
  ; CHECK-LABEL: cas_strong_i32_acqrel_acquire
  ; CHECK: ll
  ; CHECK: sc
    %val = cmpxchg i32* %mem, i32 0, i32 1 acq_rel acquire
  ; CHECK: sync
    %loaded = extractvalue { i32, i1} %val, 0
    ret i32 %loaded
  }

Sync guarantees preceding R/W complete before "cmpxchg". Other processors' 
preceding write operations are guaranteed to be visible to this 
"cmpxchg acquire" (Mips's ll and sc quarantee it).

- memory_order::seq_cst

The operation acts as an acquire, release, or both depending on whether it is 
a read, write, or read-modify-write operation, respectively. All operations 
with this memory order are observed in a sequentially consistent order.

.. code-block:: console

  define i8 @cas_strong_i8_sc_sc(i8* %mem) {
  ; CHECK-LABEL: cas_strong_i8_sc_sc
  ; CHECK: sync
  ; CHECK: ll
  ; CHECK: sc
    %val = cmpxchg i8* %mem, i8 0, i8 1 seq_cst seq_cst
  ; CHECK: sync
    %loaded = extractvalue { i8, i1} %val, 0
    ret i8 %loaded
  }

First sync guarantees preceding R/W complete before "cmpxchg seq_cst" and 
visible to "cmpxchg seq_cst". For seq_cst, a store performs a release operation.
Which means "cmpxchg seq_cst" are visible to other threads/processors that 
acquire the same atomic variable as the memory_order_release definition. 
Mips' ll and sc quarantees this feature of "cmpxchg seq_cst". 
Second Sync guarantees "cmpxchg seq_cst" complete before the next R/W.

There are several restrictions on which memory orders are supported by each 
operation. :numref:`c++-f1` (from book Figure 19-10) summarizes which 
combinations are valid.

.. _c++-f1:
.. figure:: ../Fig/c++/Fig-19-10-book-dpc++.png
  :width: 1266 px
  :height: 736 px
  :scale: 50 %
  :align: center

  Supporting atomic operations with memory_order

Load operations do not write values to memory and are therefore incompatible 
with release semantics. Similarly, store operations do not read values from 
memory and are therefore incompatible with acquire semantics. The remaining 
read-modify-write atomic operations and fences are compatible with all memory 
orderings [#dpcpp-memorder]_.

.. note:: **C++ memory_order_consume**

.. _c++-f2:
.. figure:: ../Fig/c++/Fig-19-9-book-dpc++.png
  :width: 1156 px
  :height: 1188 px
  :scale: 50 %
  :align: center

  Comparing standard C++ and SYCL/DPC++ memory models


The C++ memory model additionally includes memory_order::consume, with similar 
behavior to memory_order::acquire. however, the C++17 standard discourages its 
use, noting that its definition is being revised. its inclusion in dpC++ has 
therefore been postponed to a future version.

For a few years now, compilers have treated consume as a synonym for acquire
[#cpp-memorder-consume-as-acquire]_.

The current expectation is that the replacement facility will rely on core 
memory model and atomics definitions very similar to what's currently there. 
Since memory_order_consume does have a profound impact on the memory model, 
removing this text would allow drastic simplification, but conversely would 
make it very difficult to add anything along the lines of memory_order_consume 
back in later, especially if the standard evolves in the meantime, as expected. 
Thus we are not proposing to remove the current wording 
[#cpp-memorder-consume-remove]_.

The following test files are extracted from `memory_checks()` in
`clang/test/Sema/atomic-ops.c`. The `__c11_atomic_xxx` built-in functions used
by Clang are defined in `clang/include/clang/Basic/Builtins.def`. Compiling
these files with Clang produces the same results as shown in :numref:`c++-f1`.

Note: Clang compiles `memory_order_consume` to the same result as
`memory_order_acquire`.

.. rubric:: lbdex/input/ch12_sema_atomic-ops.c
.. literalinclude:: ../lbdex/input/ch12_sema_atomic-ops.c

.. rubric:: lbdex/input/ch12_sema_atomic-fetch.c
.. literalinclude:: ../lbdex/input/ch12_sema_atomic-fetch.c

.. table:: Atomic related between clang's builtin and llvm ir

  ================================  ===========
  clang's builtin                   llvm ir
  ================================  ===========
  __c11_atomic_load                 load atomic
  __c11_atomic_store                store atomic
  __c11_atomic_exchange_xxx         cmpxchg
  atomic_thread_fence               fence
  __c11_atomic_fetch_xxx            atomicrmw xxx
  ================================  ===========

C++ atomic functions are supported by calling implementation functions from the
C++ standard library. These functions eventually call the `__c11_atomic_xxx`
built-in functions for actual implementation.

Therefore, `__c11_atomic_xxx` functions, listed above, provide a lower-level and
higher-performance interface for C++ programmers. An example is shown below:

.. rubric:: lbdex/input/ch12_c++_atomics.cpp
.. literalinclude:: ../lbdex/input/ch12_c++_atomics.cpp


To support LLVM atomic IR instructions, the following code is added to
Chapter12_1.

.. rubric:: lbdex/chapters/Chapter12_1/Disassembler/Cpu0Disassembler.cpp
.. literalinclude:: ../lbdex/Cpu0/Disassembler/Cpu0Disassembler.cpp
    :start-after: //@DecodeMem {
    :end-before: //@DecodeMem body {
.. literalinclude:: ../lbdex/Cpu0/Disassembler/Cpu0Disassembler.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. code-block:: c++

    ...
  }

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0InstrInfo.td
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 5
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 6
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 7
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 8
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 9
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 10
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 11
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 12
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0InstrInfo.td
    :start-after: #if CH >= CH12_1 13
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0ISelLowering.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //2
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //3
    :end-before: #endif
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.h
    :start-after: #if CH >= CH12_1 //4
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0SelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@3_1 1 {
    :end-before: switch (Opcode) {
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //0.5
    :end-before: #endif //#if CH >= CH12_1 //0.5

.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: //@Cpu0TargetLowering {
    :end-before: #if CH >= CH3_2
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //1.5
    :end-before: #endif

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH6_1 //3
    :end-before: #if CH >= CH8_1 //6
.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //7
    :end-before: #endif //#if CH >= CH12_1 //7

.. code-block:: c++

    ...
  }

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //8
    :end-before: #endif

.. literalinclude:: ../lbdex/Cpu0/Cpu0ISelLowering.cpp
    :start-after: #if CH >= CH12_1 //9
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0RegisterInfo.h
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.h
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0RegisterInfo.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0RegisterInfo.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. rubric:: lbdex/chapters/Chapter12_1/Cpu0SEISelLowering.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelLowering.cpp
    :start-after: //@Cpu0SETargetLowering {
    :end-before: //@Cpu0SETargetLowering body {
.. literalinclude:: ../lbdex/Cpu0/Cpu0SEISelLowering.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. code-block:: c++

    ...
  }


.. rubric:: lbdex/chapters/Chapter12_1/Cpu0TargetMachine.cpp
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: //@Cpu0PassConfig {
    :end-before: public:
.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH12_1 //1
    :end-before: #endif

.. code-block:: c++

    ...
  };

.. literalinclude:: ../lbdex/Cpu0/Cpu0TargetMachine.cpp
    :start-after: #if CH >= CH12_1 //2
    :end-before: #endif

Since the `SC` instruction uses `RegisterOperand` type in `Cpu0InstrInfo.td` and 
`SC` uses the `FMem` node whose `DecoderMethod` is `DecodeMem`, the 
`DecodeMem()` function in `Cpu0Disassembler.cpp` needs to be modified 
accordingly.

The atomic node defined in `let usesCustomInserter = 1 in` within 
`Cpu0InstrInfo.td` tells LLVM to call `EmitInstrWithCustomInserter()` in 
`Cpu0ISelLowering.cpp` after the Instruction Selection stage, specifically in 
the `Cpu0TargetLowering::EmitInstrWithCustomInserter()` function invoked during 
the `ExpandISelPseudos::runOnMachineFunction()` phase.

For example, the declaration
`def ATOMIC_LOAD_ADD_I8 : Atomic2Ops<atomic_load_add_8, CPURegs>;`
will trigger a call to `EmitInstrWithCustomInserter()` with the machine 
instruction opcode `ATOMIC_LOAD_ADD_I8` when the IR `load atomic i8*` is 
encountered.

The call to `setInsertFencesForAtomic(true);` in `Cpu0ISelLowering.cpp` will 
trigger the `addIRPasses()` function in `Cpu0TargetMachine.cpp`, which in turn 
invokes `createAtomicExpandPass()` to create the LLVM IR `ATOMIC_FENCE`.

Later, `lowerATOMIC_FENCE()` in `Cpu0ISelLowering.cpp` will emit a 
`Cpu0ISD::Sync` when it sees an `ATOMIC_FENCE` IR, because of the statement 
`setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);` in 
`Cpu0SEISelLowering.cpp`.

Finally, the pattern defined in `Cpu0InstrInfo.td` will translate the DAG node 
into the actual `sync` instruction via `def SYNC` and its alias `SYNC 0`.

This part of the Cpu0 backend code is similar to Mips, except that Cpu0 does 
not include the `nor` instruction.

Below is a table listing the atomic IRs, their corresponding DAG nodes, and 
machine opcodes.

.. table:: The atomic related IRs, their corresponding DAGs and Opcode of Cpu0ISelLowering.cpp

  ==========================  ===========================  ===========================
  IR                          DAG                          Opcode
  ==========================  ===========================  ===========================
  load atomic                 AtomicLoad                   ATOMIC_CMP_SWAP_XXX
  store atomic                AtomicStore                  ATOMIC_SWAP_XXX
  atomicrmw add               AtomicLoadAdd                ATOMIC_LOAD_ADD_XXX
  atomicrmw sub               AtomicLoadSub                ATOMIC_LOAD_SUB_XXX
  atomicrmw xor               AtomicLoadXor                ATOMIC_LOAD_XOR_XXX
  atomicrmw and               AtomicLoadAnd                ATOMIC_LOAD_AND_XXX
  atomicrmw nand              AtomicLoadNand               ATOMIC_LOAD_NAND_XXX
  atomicrmw or                AtomicLoadOr                 ATOMIC_LOAD_OR_XXX
  cmpxchg                     AtomicCmpSwapWithSuccess     ATOMIC_CMP_SWAP_XXX
  atomicrmw xchg              AtomicLoadSwap               ATOMIC_SWAP_XXX
  ==========================  ===========================  ===========================

The input files `atomics.ll` and `atomics-fences.ll` include tests for LLVM 
atomic IRs.

The C++ source files `ch12_atomics.cpp` and `ch12_atomics-fences.cpp` are used 
to generate the corresponding LLVM atomic IRs. To compile these files, use the 
following `clang++` options:

::

  clang++ -pthread -std=c++11


.. [#exception] http://llvm.org/docs/ExceptionHandling.html

.. [#itanium-exception] http://itanium-cxx-abi.github.io/cxx-abi/exceptions.pdf

.. [#thread-wiki] http://en.wikipedia.org/wiki/Thread-local_storage

.. [#cpp-atomic] https://cplusplus.com/reference/atomic/

.. [#cpp-mem-order] https://en.cppreference.com/w/cpp/atomic/memory_order

.. [#mem-order-wiki] https://en.wikipedia.org/wiki/Memory_model_%28programming%29

.. [#atomic-stackoverflow] http://stackoverflow.com/questions/6319146/c11-introduced-a-standardized-memory-model-what-does-it-mean-and-how-is-it-g

.. [#cas-wiki] https://en.wikipedia.org/wiki/Compare-and-swap

.. [#lf-wiki] https://en.wikipedia.org/wiki/Non-blocking_algorithm

.. [#ll-wiki] https://en.wikipedia.org/wiki/Load-link/store-conditional

.. [#mb-wiki] https://en.wikipedia.org/wiki/Memory_barrier

.. [#mips-sync] From page A-158, it is same with ARM's barrier that all instructions before SYNC are completed before issuing the instructions after SYNC. Page 167 (A-155) of https://www.cs.cmu.edu/afs/cs/academic/class/15740-f97/public/doc/mips-isa.pdf

.. [#atomics-llvm] http://llvm.org/docs/Atomics.html

.. [#llvmlang-ordering] http://llvm.org/docs/LangRef.html#ordering

.. [#dpcpp-memorder] Section "The memory_order Enumeration Class" which include figure 19-10 of book https://link.springer.com/book/10.1007/978-1-4842-5574-2

.. [#cpp-memorder-consume-as-acquire] https://stackoverflow.com/questions/65336409/what-does-memory-order-consume-really-do

.. [#cpp-memorder-consume-remove] https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0371r1.html
