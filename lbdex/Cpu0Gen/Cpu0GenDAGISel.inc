/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* DAG Instruction Selector for the Cpu0 target                               *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

// *** NOTE: This file is #included into the middle of the target
// *** instruction selector class.  These functions are really methods.

// If GET_DAGISEL_DECL is #defined with any value, only function
// declarations will be included when this file is included.
// If GET_DAGISEL_BODY is #defined, its value should be the name of
// the instruction selector class. Function bodies will be emitted
// and each function's name will be qualified with the name of the
// class.
//
// When neither of the GET_DAGISEL* macros is defined, the functions
// are emitted inline.

#if defined(GET_DAGISEL_DECL) && defined(GET_DAGISEL_BODY)
#error GET_DAGISEL_DECL and GET_DAGISEL_BODY cannot be both defined, undef both for inline definitions
#endif

#ifdef GET_DAGISEL_BODY
#define LOCAL_DAGISEL_STRINGIZE(X) LOCAL_DAGISEL_STRINGIZE_(X)
#define LOCAL_DAGISEL_STRINGIZE_(X) #X
static_assert(sizeof(LOCAL_DAGISEL_STRINGIZE(GET_DAGISEL_BODY)) > 1,
   "GET_DAGISEL_BODY is empty: it should be defined with the class name");
#undef LOCAL_DAGISEL_STRINGIZE_
#undef LOCAL_DAGISEL_STRINGIZE
#endif

#if !defined(GET_DAGISEL_DECL) && !defined(GET_DAGISEL_BODY)
#define DAGISEL_INLINE 1
#else
#define DAGISEL_INLINE 0
#endif

#if !DAGISEL_INLINE
#define DAGISEL_CLASS_COLONCOLON GET_DAGISEL_BODY ::
#else
#define DAGISEL_CLASS_COLONCOLON
#endif

#ifdef GET_DAGISEL_DECL
void SelectCode(SDNode *N);
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
void DAGISEL_CLASS_COLONCOLON SelectCode(SDNode *N)
{
  // Some target values are emitted as 2 bytes, TARGET_VAL handles
  // this.
  #define TARGET_VAL(X) X & 255, unsigned(X) >> 8
  static const unsigned char MatcherTable[] = {
/*     0*/ OPC_SwitchOpcode /*44 cases */, 36|128,1/*164*/, TARGET_VAL(ISD::LOAD),// ->169
/*     5*/  OPC_RecordMemRef,
/*     6*/  OPC_RecordNode, // #0 = 'ld' chained node
/*     7*/  OPC_RecordChild1, // #1 = $addr
/*     8*/  OPC_CheckPredicate, 0, // Predicate_unindexedload
/*    10*/  OPC_CheckType, MVT::i32,
/*    12*/  OPC_Scope, 16, /*->30*/ // 6 children in Scope
/*    14*/   OPC_CheckPredicate, 1, // Predicate_load
/*    16*/   OPC_CheckPredicate, 2, // Predicate_load_a
/*    18*/   OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectAddr:$addr #2 #3
/*    21*/   OPC_EmitMergeInputChains1_0,
/*    22*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LD), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 2, 3, 
             // Src: (ld:{ *:[i32] } addr:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_load>><<P:Predicate_load_a>> - Complexity = 13
             // Dst: (LD:{ *:[i32] } addr:{ *:[iPTR] }:$addr)
/*    30*/  /*Scope*/ 18, /*->49*/
/*    31*/   OPC_CheckPredicate, 3, // Predicate_sextload
/*    33*/   OPC_CheckPredicate, 4, // Predicate_sextloadi8
/*    35*/   OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*    37*/   OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectAddr:$addr #2 #3
/*    40*/   OPC_EmitMergeInputChains1_0,
/*    41*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LB), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 2, 3, 
             // Src: (ld:{ *:[i32] } addr:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi8>> - Complexity = 13
             // Dst: (LB:{ *:[i32] } addr:{ *:[iPTR] }:$addr)
/*    49*/  /*Scope*/ 18, /*->68*/
/*    50*/   OPC_CheckPredicate, 5, // Predicate_zextload
/*    52*/   OPC_CheckPredicate, 4, // Predicate_zextloadi8
/*    54*/   OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*    56*/   OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectAddr:$addr #2 #3
/*    59*/   OPC_EmitMergeInputChains1_0,
/*    60*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LBu), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 2, 3, 
             // Src: (ld:{ *:[i32] } addr:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi8>> - Complexity = 13
             // Dst: (LBu:{ *:[i32] } addr:{ *:[iPTR] }:$addr)
/*    68*/  /*Scope*/ 20, /*->89*/
/*    69*/   OPC_CheckPredicate, 3, // Predicate_sextload
/*    71*/   OPC_CheckPredicate, 6, // Predicate_sextloadi16
/*    73*/   OPC_CheckPredicate, 2, // Predicate_sextloadi16_a
/*    75*/   OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*    77*/   OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectAddr:$addr #2 #3
/*    80*/   OPC_EmitMergeInputChains1_0,
/*    81*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LH), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 2, 3, 
             // Src: (ld:{ *:[i32] } addr:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_sextload>><<P:Predicate_sextloadi16>><<P:Predicate_sextloadi16_a>> - Complexity = 13
             // Dst: (LH:{ *:[i32] } addr:{ *:[iPTR] }:$addr)
/*    89*/  /*Scope*/ 20, /*->110*/
/*    90*/   OPC_CheckPredicate, 5, // Predicate_zextload
/*    92*/   OPC_CheckPredicate, 6, // Predicate_zextloadi16
/*    94*/   OPC_CheckPredicate, 2, // Predicate_zextloadi16_a
/*    96*/   OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*    98*/   OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectAddr:$addr #2 #3
/*   101*/   OPC_EmitMergeInputChains1_0,
/*   102*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LHu), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 2, 3, 
             // Src: (ld:{ *:[i32] } addr:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedload>><<P:Predicate_zextload>><<P:Predicate_zextloadi16>><<P:Predicate_zextloadi16_a>> - Complexity = 13
             // Dst: (LHu:{ *:[i32] } addr:{ *:[iPTR] }:$addr)
/*   110*/  /*Scope*/ 57, /*->168*/
/*   111*/   OPC_CheckPredicate, 7, // Predicate_extload
/*   113*/   OPC_Scope, 16, /*->131*/ // 3 children in Scope
/*   115*/    OPC_CheckPredicate, 8, // Predicate_extloadi1
/*   117*/    OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*   119*/    OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectAddr:$src #2 #3
/*   122*/    OPC_EmitMergeInputChains1_0,
/*   123*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LBu), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (ld:{ *:[i32] } addr:{ *:[iPTR] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi1>> - Complexity = 13
              // Dst: (LBu:{ *:[i32] } addr:{ *:[iPTR] }:$src)
/*   131*/   /*Scope*/ 16, /*->148*/
/*   132*/    OPC_CheckPredicate, 4, // Predicate_extloadi8
/*   134*/    OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*   136*/    OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectAddr:$src #2 #3
/*   139*/    OPC_EmitMergeInputChains1_0,
/*   140*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LBu), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (ld:{ *:[i32] } addr:{ *:[iPTR] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi8>> - Complexity = 13
              // Dst: (LBu:{ *:[i32] } addr:{ *:[iPTR] }:$src)
/*   148*/   /*Scope*/ 18, /*->167*/
/*   149*/    OPC_CheckPredicate, 6, // Predicate_extloadi16
/*   151*/    OPC_CheckPredicate, 2, // Predicate_extloadi16_a
/*   153*/    OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*   155*/    OPC_CheckComplexPat, /*CP*/0, /*#*/1, // SelectAddr:$src #2 #3
/*   158*/    OPC_EmitMergeInputChains1_0,
/*   159*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LHu), 0|OPFL_Chain|OPFL_MemRefs,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (ld:{ *:[i32] } addr:{ *:[iPTR] }:$src)<<P:Predicate_unindexedload>><<P:Predicate_extload>><<P:Predicate_extloadi16>><<P:Predicate_extloadi16_a>> - Complexity = 13
              // Dst: (LHu:{ *:[i32] } addr:{ *:[iPTR] }:$src)
/*   167*/   0, /*End of Scope*/
/*   168*/  0, /*End of Scope*/
/*   169*/ /*SwitchOpcode*/ 68, TARGET_VAL(ISD::STORE),// ->240
/*   172*/  OPC_RecordMemRef,
/*   173*/  OPC_RecordNode, // #0 = 'st' chained node
/*   174*/  OPC_RecordChild1, // #1 = $ra
/*   175*/  OPC_CheckChild1Type, MVT::i32,
/*   177*/  OPC_RecordChild2, // #2 = $addr
/*   178*/  OPC_CheckPredicate, 9, // Predicate_unindexedstore
/*   180*/  OPC_Scope, 16, /*->198*/ // 2 children in Scope
/*   182*/   OPC_CheckPredicate, 10, // Predicate_store
/*   184*/   OPC_CheckPredicate, 11, // Predicate_store_a
/*   186*/   OPC_CheckComplexPat, /*CP*/0, /*#*/2, // SelectAddr:$addr #3 #4
/*   189*/   OPC_EmitMergeInputChains1_0,
/*   190*/   OPC_MorphNodeTo0, TARGET_VAL(Cpu0::ST), 0|OPFL_Chain|OPFL_MemRefs,
                 3/*#Ops*/, 1, 3, 4, 
             // Src: (st GPROut:{ *:[i32] }:$ra, addr:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_store>><<P:Predicate_store_a>> - Complexity = 13
             // Dst: (ST GPROut:{ *:[i32] }:$ra, addr:{ *:[iPTR] }:$addr)
/*   198*/  /*Scope*/ 40, /*->239*/
/*   199*/   OPC_CheckPredicate, 12, // Predicate_truncstore
/*   201*/   OPC_Scope, 16, /*->219*/ // 2 children in Scope
/*   203*/    OPC_CheckPredicate, 4, // Predicate_truncstorei8
/*   205*/    OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*   207*/    OPC_CheckComplexPat, /*CP*/0, /*#*/2, // SelectAddr:$addr #3 #4
/*   210*/    OPC_EmitMergeInputChains1_0,
/*   211*/    OPC_MorphNodeTo0, TARGET_VAL(Cpu0::SB), 0|OPFL_Chain|OPFL_MemRefs,
                  3/*#Ops*/, 1, 3, 4, 
              // Src: (st GPROut:{ *:[i32] }:$ra, addr:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei8>> - Complexity = 13
              // Dst: (SB GPROut:{ *:[i32] }:$ra, addr:{ *:[iPTR] }:$addr)
/*   219*/   /*Scope*/ 18, /*->238*/
/*   220*/    OPC_CheckPredicate, 6, // Predicate_truncstorei16
/*   222*/    OPC_CheckPredicate, 11, // Predicate_truncstorei16_a
/*   224*/    OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*   226*/    OPC_CheckComplexPat, /*CP*/0, /*#*/2, // SelectAddr:$addr #3 #4
/*   229*/    OPC_EmitMergeInputChains1_0,
/*   230*/    OPC_MorphNodeTo0, TARGET_VAL(Cpu0::SH), 0|OPFL_Chain|OPFL_MemRefs,
                  3/*#Ops*/, 1, 3, 4, 
              // Src: (st GPROut:{ *:[i32] }:$ra, addr:{ *:[iPTR] }:$addr)<<P:Predicate_unindexedstore>><<P:Predicate_truncstore>><<P:Predicate_truncstorei16>><<P:Predicate_truncstorei16_a>> - Complexity = 13
              // Dst: (SH GPROut:{ *:[i32] }:$ra, addr:{ *:[iPTR] }:$addr)
/*   238*/   0, /*End of Scope*/
/*   239*/  0, /*End of Scope*/
/*   240*/ /*SwitchOpcode*/ 98, TARGET_VAL(ISD::XOR),// ->341
/*   243*/  OPC_Scope, 28, /*->273*/ // 2 children in Scope
/*   245*/   OPC_MoveChild0,
/*   246*/   OPC_CheckOpcode, TARGET_VAL(ISD::OR),
/*   249*/   OPC_RecordChild0, // #0 = $rb
/*   250*/   OPC_RecordChild1, // #1 = $rc
/*   251*/   OPC_MoveParent,
/*   252*/   OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   263*/   OPC_CheckPatternPredicate, 1, // (Subtarget->hasChapter4_2())
/*   265*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::NOR), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (xor:{ *:[i32] } (or:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc), -1:{ *:[i32] }) - Complexity = 11
             // Dst: (NOR:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*   273*/  /*Scope*/ 66, /*->340*/
/*   274*/   OPC_RecordChild0, // #0 = $in
/*   275*/   OPC_Scope, 24, /*->301*/ // 2 children in Scope
/*   277*/    OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   288*/    OPC_CheckPatternPredicate, 1, // (Subtarget->hasChapter4_2())
/*   290*/    OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   293*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::NOR), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (xor:{ *:[i32] } CPURegs:{ *:[i32] }:$in, -1:{ *:[i32] }) - Complexity = 8
              // Dst: (NOR:{ *:[i32] } CPURegs:{ *:[i32] }:$in, ZERO:{ *:[i32] })
/*   301*/   /*Scope*/ 37, /*->339*/
/*   302*/    OPC_RecordChild1, // #1 = $imm16
/*   303*/    OPC_Scope, 22, /*->327*/ // 2 children in Scope
/*   305*/     OPC_MoveChild1,
/*   306*/     OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*   309*/     OPC_CheckPredicate, 13, // Predicate_immZExt16
/*   311*/     OPC_MoveParent,
/*   312*/     OPC_CheckPatternPredicate, 1, // (Subtarget->hasChapter4_2())
/*   314*/     OPC_EmitConvertToTarget, 1,
/*   316*/     OPC_EmitNodeXForm, 0, 2, // LO16
/*   319*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                   MVT::i32, 2/*#Ops*/, 0, 3, 
               // Src: (xor:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immZExt16>><<X:LO16>>:$imm16) - Complexity = 7
               // Dst: (XORi:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (LO16:{ *:[i32] } (imm:{ *:[i32] }):$imm16))
/*   327*/    /*Scope*/ 10, /*->338*/
/*   328*/     OPC_CheckPatternPredicate, 1, // (Subtarget->hasChapter4_2())
/*   330*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XOR), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (xor:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
               // Dst: (XOR:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*   338*/    0, /*End of Scope*/
/*   339*/   0, /*End of Scope*/
/*   340*/  0, /*End of Scope*/
/*   341*/ /*SwitchOpcode*/ 40, TARGET_VAL(ISD::CTLZ),// ->384
/*   344*/  OPC_Scope, 26, /*->372*/ // 2 children in Scope
/*   346*/   OPC_MoveChild0,
/*   347*/   OPC_CheckOpcode, TARGET_VAL(ISD::XOR),
/*   350*/   OPC_RecordChild0, // #0 = $rb
/*   351*/   OPC_CheckChild1Integer, 127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,127|128,1/*18446744073709551615*/, 
/*   362*/   OPC_MoveParent,
/*   363*/   OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*   365*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::CLO), 0,
                 MVT::i32, 1/*#Ops*/, 0, 
             // Src: (ctlz:{ *:[i32] } (xor:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, -1:{ *:[i32] })) - Complexity = 11
             // Dst: (CLO:{ *:[i32] } CPURegs:{ *:[i32] }:$rb)
/*   372*/  /*Scope*/ 10, /*->383*/
/*   373*/   OPC_RecordChild0, // #0 = $rb
/*   374*/   OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*   376*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::CLZ), 0,
                 MVT::i32, 1/*#Ops*/, 0, 
             // Src: (ctlz:{ *:[i32] } CPURegs:{ *:[i32] }:$rb) - Complexity = 3
             // Dst: (CLZ:{ *:[i32] } CPURegs:{ *:[i32] }:$rb)
/*   383*/  0, /*End of Scope*/
/*   384*/ /*SwitchOpcode*/ 86|128,6/*854*/, TARGET_VAL(ISD::BRCOND),// ->1242
/*   388*/  OPC_RecordNode, // #0 = 'brcond' chained node
/*   389*/  OPC_Scope, 33|128,6/*801*/, /*->1193*/ // 2 children in Scope
/*   392*/   OPC_MoveChild1,
/*   393*/   OPC_CheckOpcode, TARGET_VAL(ISD::SETCC),
/*   396*/   OPC_RecordChild0, // #1 = $lhs
/*   397*/   OPC_Scope, 54, /*->453*/ // 2 children in Scope
/*   399*/    OPC_CheckChild1Integer, 0, 
/*   401*/    OPC_Scope, 24, /*->427*/ // 2 children in Scope
/*   403*/     OPC_CheckChild2CondCode, ISD::SETNE,
/*   405*/     OPC_MoveParent,
/*   406*/     OPC_RecordChild2, // #2 = $dst
/*   407*/     OPC_MoveChild2,
/*   408*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   411*/     OPC_MoveParent,
/*   412*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   414*/     OPC_EmitMergeInputChains1_0,
/*   415*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   418*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BNE), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 1, 3, 2, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, 0:{ *:[i32] }, SETNE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 11
               // Dst: (BNE:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   427*/    /*Scope*/ 24, /*->452*/
/*   428*/     OPC_CheckChild2CondCode, ISD::SETEQ,
/*   430*/     OPC_MoveParent,
/*   431*/     OPC_RecordChild2, // #2 = $dst
/*   432*/     OPC_MoveChild2,
/*   433*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   436*/     OPC_MoveParent,
/*   437*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   439*/     OPC_EmitMergeInputChains1_0,
/*   440*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   443*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BEQ), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 1, 3, 2, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, 0:{ *:[i32] }, SETEQ:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 11
               // Dst: (BEQ:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   452*/    0, /*End of Scope*/
/*   453*/   /*Scope*/ 97|128,5/*737*/, /*->1192*/
/*   455*/    OPC_CheckChild0Type, MVT::i32,
/*   457*/    OPC_RecordChild1, // #2 = $rb
/*   458*/    OPC_Scope, 21, /*->481*/ // 26 children in Scope
/*   460*/     OPC_CheckChild2CondCode, ISD::SETEQ,
/*   462*/     OPC_MoveParent,
/*   463*/     OPC_RecordChild2, // #3 = $imm16
/*   464*/     OPC_MoveChild2,
/*   465*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   468*/     OPC_MoveParent,
/*   469*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   471*/     OPC_EmitMergeInputChains1_0,
/*   472*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BEQ), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 1, 2, 3, 
               // Src: (brcond (setcc:{ *:[i32] } GPROut:{ *:[i32] }:$ra, GPROut:{ *:[i32] }:$rb, SETEQ:{ *:[Other] }), (bb:{ *:[Other] }):$imm16) - Complexity = 6
               // Dst: (BEQ:{ *:[i32] } GPROut:{ *:[i32] }:$ra, GPROut:{ *:[i32] }:$rb, (bb:{ *:[Other] }):$imm16)
/*   481*/    /*Scope*/ 21, /*->503*/
/*   482*/     OPC_CheckChild2CondCode, ISD::SETNE,
/*   484*/     OPC_MoveParent,
/*   485*/     OPC_RecordChild2, // #3 = $imm16
/*   486*/     OPC_MoveChild2,
/*   487*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   490*/     OPC_MoveParent,
/*   491*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   493*/     OPC_EmitMergeInputChains1_0,
/*   494*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BNE), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 1, 2, 3, 
               // Src: (brcond (setcc:{ *:[i32] } GPROut:{ *:[i32] }:$ra, GPROut:{ *:[i32] }:$rb, SETNE:{ *:[Other] }), (bb:{ *:[Other] }):$imm16) - Complexity = 6
               // Dst: (BNE:{ *:[i32] } GPROut:{ *:[i32] }:$ra, GPROut:{ *:[i32] }:$rb, (bb:{ *:[Other] }):$imm16)
/*   503*/    /*Scope*/ 21, /*->525*/
/*   504*/     OPC_CheckChild2CondCode, ISD::SETEQ,
/*   506*/     OPC_MoveParent,
/*   507*/     OPC_RecordChild2, // #3 = $dst
/*   508*/     OPC_MoveChild2,
/*   509*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   512*/     OPC_MoveParent,
/*   513*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   515*/     OPC_EmitMergeInputChains1_0,
/*   516*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BEQ), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 1, 2, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETEQ:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BEQ:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, (bb:{ *:[Other] }):$dst)
/*   525*/    /*Scope*/ 21, /*->547*/
/*   526*/     OPC_CheckChild2CondCode, ISD::SETUEQ,
/*   528*/     OPC_MoveParent,
/*   529*/     OPC_RecordChild2, // #3 = $dst
/*   530*/     OPC_MoveChild2,
/*   531*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   534*/     OPC_MoveParent,
/*   535*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   537*/     OPC_EmitMergeInputChains1_0,
/*   538*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BEQ), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 1, 2, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUEQ:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BEQ:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, (bb:{ *:[Other] }):$dst)
/*   547*/    /*Scope*/ 21, /*->569*/
/*   548*/     OPC_CheckChild2CondCode, ISD::SETNE,
/*   550*/     OPC_MoveParent,
/*   551*/     OPC_RecordChild2, // #3 = $dst
/*   552*/     OPC_MoveChild2,
/*   553*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   556*/     OPC_MoveParent,
/*   557*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   559*/     OPC_EmitMergeInputChains1_0,
/*   560*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BNE), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 1, 2, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETNE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BNE:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, (bb:{ *:[Other] }):$dst)
/*   569*/    /*Scope*/ 21, /*->591*/
/*   570*/     OPC_CheckChild2CondCode, ISD::SETUNE,
/*   572*/     OPC_MoveParent,
/*   573*/     OPC_RecordChild2, // #3 = $dst
/*   574*/     OPC_MoveChild2,
/*   575*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   578*/     OPC_MoveParent,
/*   579*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   581*/     OPC_EmitMergeInputChains1_0,
/*   582*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BNE), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 1, 2, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUNE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BNE:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, (bb:{ *:[Other] }):$dst)
/*   591*/    /*Scope*/ 32, /*->624*/
/*   592*/     OPC_CheckChild2CondCode, ISD::SETLT,
/*   594*/     OPC_MoveParent,
/*   595*/     OPC_RecordChild2, // #3 = $dst
/*   596*/     OPC_MoveChild2,
/*   597*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   600*/     OPC_MoveParent,
/*   601*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   603*/     OPC_EmitMergeInputChains1_0,
/*   604*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLT), 0,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*   612*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   615*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BNE), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 4, 5, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETLT:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BNE:{ *:[i32] } (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   624*/    /*Scope*/ 32, /*->657*/
/*   625*/     OPC_CheckChild2CondCode, ISD::SETULT,
/*   627*/     OPC_MoveParent,
/*   628*/     OPC_RecordChild2, // #3 = $dst
/*   629*/     OPC_MoveChild2,
/*   630*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   633*/     OPC_MoveParent,
/*   634*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   636*/     OPC_EmitMergeInputChains1_0,
/*   637*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLTu), 0,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*   645*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   648*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BNE), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 4, 5, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETULT:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BNE:{ *:[i32] } (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   657*/    /*Scope*/ 32, /*->690*/
/*   658*/     OPC_CheckChild2CondCode, ISD::SETGT,
/*   660*/     OPC_MoveParent,
/*   661*/     OPC_RecordChild2, // #3 = $dst
/*   662*/     OPC_MoveChild2,
/*   663*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   666*/     OPC_MoveParent,
/*   667*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   669*/     OPC_EmitMergeInputChains1_0,
/*   670*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLT), 0,
                   MVT::i32, 2/*#Ops*/, 2, 1,  // Results = #4
/*   678*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   681*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BNE), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 4, 5, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETGT:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BNE:{ *:[i32] } (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   690*/    /*Scope*/ 32, /*->723*/
/*   691*/     OPC_CheckChild2CondCode, ISD::SETUGT,
/*   693*/     OPC_MoveParent,
/*   694*/     OPC_RecordChild2, // #3 = $dst
/*   695*/     OPC_MoveChild2,
/*   696*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   699*/     OPC_MoveParent,
/*   700*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   702*/     OPC_EmitMergeInputChains1_0,
/*   703*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLTu), 0,
                   MVT::i32, 2/*#Ops*/, 2, 1,  // Results = #4
/*   711*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   714*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BNE), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 4, 5, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUGT:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BNE:{ *:[i32] } (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   723*/    /*Scope*/ 32, /*->756*/
/*   724*/     OPC_CheckChild2CondCode, ISD::SETLE,
/*   726*/     OPC_MoveParent,
/*   727*/     OPC_RecordChild2, // #3 = $dst
/*   728*/     OPC_MoveChild2,
/*   729*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   732*/     OPC_MoveParent,
/*   733*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   735*/     OPC_EmitMergeInputChains1_0,
/*   736*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLT), 0,
                   MVT::i32, 2/*#Ops*/, 2, 1,  // Results = #4
/*   744*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   747*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BEQ), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 4, 5, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETLE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BEQ:{ *:[i32] } (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   756*/    /*Scope*/ 32, /*->789*/
/*   757*/     OPC_CheckChild2CondCode, ISD::SETULE,
/*   759*/     OPC_MoveParent,
/*   760*/     OPC_RecordChild2, // #3 = $dst
/*   761*/     OPC_MoveChild2,
/*   762*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   765*/     OPC_MoveParent,
/*   766*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   768*/     OPC_EmitMergeInputChains1_0,
/*   769*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLTu), 0,
                   MVT::i32, 2/*#Ops*/, 2, 1,  // Results = #4
/*   777*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   780*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BEQ), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 4, 5, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETULE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BEQ:{ *:[i32] } (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   789*/    /*Scope*/ 32, /*->822*/
/*   790*/     OPC_CheckChild2CondCode, ISD::SETGE,
/*   792*/     OPC_MoveParent,
/*   793*/     OPC_RecordChild2, // #3 = $dst
/*   794*/     OPC_MoveChild2,
/*   795*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   798*/     OPC_MoveParent,
/*   799*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   801*/     OPC_EmitMergeInputChains1_0,
/*   802*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLT), 0,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*   810*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   813*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BEQ), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 4, 5, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETGE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BEQ:{ *:[i32] } (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   822*/    /*Scope*/ 32, /*->855*/
/*   823*/     OPC_CheckChild2CondCode, ISD::SETUGE,
/*   825*/     OPC_MoveParent,
/*   826*/     OPC_RecordChild2, // #3 = $dst
/*   827*/     OPC_MoveChild2,
/*   828*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   831*/     OPC_MoveParent,
/*   832*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*   834*/     OPC_EmitMergeInputChains1_0,
/*   835*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLTu), 0,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*   843*/     OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*   846*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BEQ), 0|OPFL_Chain,
                   MVT::i32, 3/*#Ops*/, 4, 5, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUGE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (BEQ:{ *:[i32] } (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*   855*/    /*Scope*/ 27, /*->883*/
/*   856*/     OPC_CheckChild2CondCode, ISD::SETEQ,
/*   858*/     OPC_MoveParent,
/*   859*/     OPC_RecordChild2, // #3 = $dst
/*   860*/     OPC_MoveChild2,
/*   861*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   864*/     OPC_MoveParent,
/*   865*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*   867*/     OPC_EmitMergeInputChains1_0,
/*   868*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*   876*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JEQ), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETEQ:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JEQ (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*   883*/    /*Scope*/ 27, /*->911*/
/*   884*/     OPC_CheckChild2CondCode, ISD::SETUEQ,
/*   886*/     OPC_MoveParent,
/*   887*/     OPC_RecordChild2, // #3 = $dst
/*   888*/     OPC_MoveChild2,
/*   889*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   892*/     OPC_MoveParent,
/*   893*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*   895*/     OPC_EmitMergeInputChains1_0,
/*   896*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMPu), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*   904*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JEQ), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUEQ:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JEQ (CMPu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*   911*/    /*Scope*/ 27, /*->939*/
/*   912*/     OPC_CheckChild2CondCode, ISD::SETNE,
/*   914*/     OPC_MoveParent,
/*   915*/     OPC_RecordChild2, // #3 = $dst
/*   916*/     OPC_MoveChild2,
/*   917*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   920*/     OPC_MoveParent,
/*   921*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*   923*/     OPC_EmitMergeInputChains1_0,
/*   924*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*   932*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JNE), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETNE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JNE (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*   939*/    /*Scope*/ 27, /*->967*/
/*   940*/     OPC_CheckChild2CondCode, ISD::SETUNE,
/*   942*/     OPC_MoveParent,
/*   943*/     OPC_RecordChild2, // #3 = $dst
/*   944*/     OPC_MoveChild2,
/*   945*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   948*/     OPC_MoveParent,
/*   949*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*   951*/     OPC_EmitMergeInputChains1_0,
/*   952*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMPu), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*   960*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JNE), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUNE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JNE (CMPu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*   967*/    /*Scope*/ 27, /*->995*/
/*   968*/     OPC_CheckChild2CondCode, ISD::SETLT,
/*   970*/     OPC_MoveParent,
/*   971*/     OPC_RecordChild2, // #3 = $dst
/*   972*/     OPC_MoveChild2,
/*   973*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*   976*/     OPC_MoveParent,
/*   977*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*   979*/     OPC_EmitMergeInputChains1_0,
/*   980*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*   988*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JLT), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETLT:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JLT (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*   995*/    /*Scope*/ 27, /*->1023*/
/*   996*/     OPC_CheckChild2CondCode, ISD::SETULT,
/*   998*/     OPC_MoveParent,
/*   999*/     OPC_RecordChild2, // #3 = $dst
/*  1000*/     OPC_MoveChild2,
/*  1001*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1004*/     OPC_MoveParent,
/*  1005*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  1007*/     OPC_EmitMergeInputChains1_0,
/*  1008*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMPu), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*  1016*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JLT), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETULT:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JLT (CMPu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*  1023*/    /*Scope*/ 27, /*->1051*/
/*  1024*/     OPC_CheckChild2CondCode, ISD::SETGT,
/*  1026*/     OPC_MoveParent,
/*  1027*/     OPC_RecordChild2, // #3 = $dst
/*  1028*/     OPC_MoveChild2,
/*  1029*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1032*/     OPC_MoveParent,
/*  1033*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  1035*/     OPC_EmitMergeInputChains1_0,
/*  1036*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*  1044*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JGT), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETGT:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JGT (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*  1051*/    /*Scope*/ 27, /*->1079*/
/*  1052*/     OPC_CheckChild2CondCode, ISD::SETUGT,
/*  1054*/     OPC_MoveParent,
/*  1055*/     OPC_RecordChild2, // #3 = $dst
/*  1056*/     OPC_MoveChild2,
/*  1057*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1060*/     OPC_MoveParent,
/*  1061*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  1063*/     OPC_EmitMergeInputChains1_0,
/*  1064*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMPu), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*  1072*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JGT), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUGT:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JGT (CMPu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*  1079*/    /*Scope*/ 27, /*->1107*/
/*  1080*/     OPC_CheckChild2CondCode, ISD::SETLE,
/*  1082*/     OPC_MoveParent,
/*  1083*/     OPC_RecordChild2, // #3 = $dst
/*  1084*/     OPC_MoveChild2,
/*  1085*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1088*/     OPC_MoveParent,
/*  1089*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  1091*/     OPC_EmitMergeInputChains1_0,
/*  1092*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*  1100*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JLE), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETLE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JLE (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*  1107*/    /*Scope*/ 27, /*->1135*/
/*  1108*/     OPC_CheckChild2CondCode, ISD::SETULE,
/*  1110*/     OPC_MoveParent,
/*  1111*/     OPC_RecordChild2, // #3 = $dst
/*  1112*/     OPC_MoveChild2,
/*  1113*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1116*/     OPC_MoveParent,
/*  1117*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  1119*/     OPC_EmitMergeInputChains1_0,
/*  1120*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMPu), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*  1128*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JLE), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETULE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JLE (CMPu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*  1135*/    /*Scope*/ 27, /*->1163*/
/*  1136*/     OPC_CheckChild2CondCode, ISD::SETGE,
/*  1138*/     OPC_MoveParent,
/*  1139*/     OPC_RecordChild2, // #3 = $dst
/*  1140*/     OPC_MoveChild2,
/*  1141*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1144*/     OPC_MoveParent,
/*  1145*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  1147*/     OPC_EmitMergeInputChains1_0,
/*  1148*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*  1156*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JGE), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETGE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JGE (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*  1163*/    /*Scope*/ 27, /*->1191*/
/*  1164*/     OPC_CheckChild2CondCode, ISD::SETUGE,
/*  1166*/     OPC_MoveParent,
/*  1167*/     OPC_RecordChild2, // #3 = $dst
/*  1168*/     OPC_MoveChild2,
/*  1169*/     OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1172*/     OPC_MoveParent,
/*  1173*/     OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  1175*/     OPC_EmitMergeInputChains1_0,
/*  1176*/     OPC_EmitNode1, TARGET_VAL(Cpu0::CMPu), 0|OPFL_Chain,
                   MVT::i32, 2/*#Ops*/, 1, 2,  // Results = #4
/*  1184*/     OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JGE), 0|OPFL_Chain,
                   2/*#Ops*/, 4, 3, 
               // Src: (brcond (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUGE:{ *:[Other] }), (bb:{ *:[Other] }):$dst) - Complexity = 6
               // Dst: (JGE (CMPu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), (bb:{ *:[Other] }):$dst)
/*  1191*/    0, /*End of Scope*/
/*  1192*/   0, /*End of Scope*/
/*  1193*/  /*Scope*/ 47, /*->1241*/
/*  1194*/   OPC_RecordChild1, // #1 = $cond
/*  1195*/   OPC_RecordChild2, // #2 = $dst
/*  1196*/   OPC_MoveChild2,
/*  1197*/   OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  1200*/   OPC_MoveParent,
/*  1201*/   OPC_Scope, 15, /*->1218*/ // 2 children in Scope
/*  1203*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  1205*/    OPC_EmitMergeInputChains1_0,
/*  1206*/    OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*  1209*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::BNE), 0|OPFL_Chain,
                  MVT::i32, 3/*#Ops*/, 1, 3, 2, 
              // Src: (brcond CPURegs:{ *:[i32] }:$cond, (bb:{ *:[Other] }):$dst) - Complexity = 3
              // Dst: (BNE:{ *:[i32] } CPURegs:{ *:[i32] }:$cond, ZERO:{ *:[i32] }, (bb:{ *:[Other] }):$dst)
/*  1218*/   /*Scope*/ 21, /*->1240*/
/*  1219*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  1221*/    OPC_EmitMergeInputChains1_0,
/*  1222*/    OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*  1225*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0|OPFL_Chain,
                  MVT::i32, 2/*#Ops*/, 1, 3,  // Results = #4
/*  1233*/    OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JNE), 0|OPFL_Chain,
                  2/*#Ops*/, 4, 2, 
              // Src: (brcond CPURegs:{ *:[i32] }:$cond, (bb:{ *:[Other] }):$dst) - Complexity = 3
              // Dst: (JNE (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$cond, ZERO:{ *:[i32] }), (bb:{ *:[Other] }):$dst)
/*  1240*/   0, /*End of Scope*/
/*  1241*/  0, /*End of Scope*/
/*  1242*/ /*SwitchOpcode*/ 53|128,2/*309*/, TARGET_VAL(ISD::SELECT),// ->1555
/*  1246*/  OPC_Scope, 32|128,2/*288*/, /*->1537*/ // 2 children in Scope
/*  1249*/   OPC_MoveChild0,
/*  1250*/   OPC_CheckOpcode, TARGET_VAL(ISD::SETCC),
/*  1253*/   OPC_RecordChild0, // #0 = $lhs
/*  1254*/   OPC_Scope, 42, /*->1298*/ // 3 children in Scope
/*  1256*/    OPC_CheckChild1Integer, 0, 
/*  1258*/    OPC_Scope, 18, /*->1278*/ // 2 children in Scope
/*  1260*/     OPC_CheckChild2CondCode, ISD::SETEQ,
/*  1262*/     OPC_MoveParent,
/*  1263*/     OPC_RecordChild1, // #1 = $T
/*  1264*/     OPC_RecordChild2, // #2 = $F
/*  1265*/     OPC_CheckType, MVT::i32,
/*  1267*/     OPC_CheckPatternPredicate, 4, // (Subtarget->hasChapter8_2())
/*  1269*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVZ_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 1, 0, 2, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, 0:{ *:[i32] }, SETEQ:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 11
               // Dst: (MOVZ_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$F)
/*  1278*/    /*Scope*/ 18, /*->1297*/
/*  1279*/     OPC_CheckChild2CondCode, ISD::SETNE,
/*  1281*/     OPC_MoveParent,
/*  1282*/     OPC_RecordChild1, // #1 = $T
/*  1283*/     OPC_RecordChild2, // #2 = $F
/*  1284*/     OPC_CheckType, MVT::i32,
/*  1286*/     OPC_CheckPatternPredicate, 4, // (Subtarget->hasChapter8_2())
/*  1288*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVN_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 1, 0, 2, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, 0:{ *:[i32] }, SETNE:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 11
               // Dst: (MOVN_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$F)
/*  1297*/    0, /*End of Scope*/
/*  1298*/   /*Scope*/ 68, /*->1367*/
/*  1299*/    OPC_RecordChild1, // #1 = $rhs
/*  1300*/    OPC_MoveChild1,
/*  1301*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1304*/    OPC_CheckPredicate, 14, // Predicate_immSExt16
/*  1306*/    OPC_MoveParent,
/*  1307*/    OPC_Scope, 28, /*->1337*/ // 2 children in Scope
/*  1309*/     OPC_CheckChild2CondCode, ISD::SETGE,
/*  1311*/     OPC_MoveParent,
/*  1312*/     OPC_RecordChild1, // #2 = $T
/*  1313*/     OPC_RecordChild2, // #3 = $F
/*  1314*/     OPC_CheckType, MVT::i32,
/*  1316*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  1318*/     OPC_EmitConvertToTarget, 1,
/*  1320*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLTi), 0,
                   MVT::i32, 2/*#Ops*/, 0, 4,  // Results = #5
/*  1328*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVZ_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 2, 5, 3, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$rhs, SETGE:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 10
               // Dst: (MOVZ_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, (SLTi:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$rhs), CPURegs:{ *:[i32] }:$F)
/*  1337*/    /*Scope*/ 28, /*->1366*/
/*  1338*/     OPC_CheckChild2CondCode, ISD::SETUGE,
/*  1340*/     OPC_MoveParent,
/*  1341*/     OPC_RecordChild1, // #2 = $T
/*  1342*/     OPC_RecordChild2, // #3 = $F
/*  1343*/     OPC_CheckType, MVT::i32,
/*  1345*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  1347*/     OPC_EmitConvertToTarget, 1,
/*  1349*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLTiu), 0,
                   MVT::i32, 2/*#Ops*/, 0, 4,  // Results = #5
/*  1357*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVZ_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 2, 5, 3, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lh, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$rh, SETUGE:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 10
               // Dst: (MOVZ_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, (SLTiu:{ *:[i32] } CPURegs:{ *:[i32] }:$lh, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$rh), CPURegs:{ *:[i32] }:$F)
/*  1366*/    0, /*End of Scope*/
/*  1367*/   /*Scope*/ 39|128,1/*167*/, /*->1536*/
/*  1369*/    OPC_CheckChild0Type, MVT::i32,
/*  1371*/    OPC_RecordChild1, // #1 = $rhs
/*  1372*/    OPC_Scope, 26, /*->1400*/ // 6 children in Scope
/*  1374*/     OPC_CheckChild2CondCode, ISD::SETGE,
/*  1376*/     OPC_MoveParent,
/*  1377*/     OPC_RecordChild1, // #2 = $T
/*  1378*/     OPC_RecordChild2, // #3 = $F
/*  1379*/     OPC_CheckType, MVT::i32,
/*  1381*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  1383*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLT), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #4
/*  1391*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVZ_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 2, 4, 3, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETGE:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 6
               // Dst: (MOVZ_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), CPURegs:{ *:[i32] }:$F)
/*  1400*/    /*Scope*/ 26, /*->1427*/
/*  1401*/     OPC_CheckChild2CondCode, ISD::SETUGE,
/*  1403*/     OPC_MoveParent,
/*  1404*/     OPC_RecordChild1, // #2 = $T
/*  1405*/     OPC_RecordChild2, // #3 = $F
/*  1406*/     OPC_CheckType, MVT::i32,
/*  1408*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  1410*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLTu), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #4
/*  1418*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVZ_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 2, 4, 3, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUGE:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 6
               // Dst: (MOVZ_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), CPURegs:{ *:[i32] }:$F)
/*  1427*/    /*Scope*/ 26, /*->1454*/
/*  1428*/     OPC_CheckChild2CondCode, ISD::SETLE,
/*  1430*/     OPC_MoveParent,
/*  1431*/     OPC_RecordChild1, // #2 = $T
/*  1432*/     OPC_RecordChild2, // #3 = $F
/*  1433*/     OPC_CheckType, MVT::i32,
/*  1435*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  1437*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLT), 0,
                   MVT::i32, 2/*#Ops*/, 1, 0,  // Results = #4
/*  1445*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVZ_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 2, 4, 3, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETLE:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 6
               // Dst: (MOVZ_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), CPURegs:{ *:[i32] }:$F)
/*  1454*/    /*Scope*/ 26, /*->1481*/
/*  1455*/     OPC_CheckChild2CondCode, ISD::SETULE,
/*  1457*/     OPC_MoveParent,
/*  1458*/     OPC_RecordChild1, // #2 = $T
/*  1459*/     OPC_RecordChild2, // #3 = $F
/*  1460*/     OPC_CheckType, MVT::i32,
/*  1462*/     OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  1464*/     OPC_EmitNode1, TARGET_VAL(Cpu0::SLTu), 0,
                   MVT::i32, 2/*#Ops*/, 1, 0,  // Results = #4
/*  1472*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVZ_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 2, 4, 3, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETULE:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 6
               // Dst: (MOVZ_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), CPURegs:{ *:[i32] }:$F)
/*  1481*/    /*Scope*/ 26, /*->1508*/
/*  1482*/     OPC_CheckChild2CondCode, ISD::SETEQ,
/*  1484*/     OPC_MoveParent,
/*  1485*/     OPC_RecordChild1, // #2 = $T
/*  1486*/     OPC_RecordChild2, // #3 = $F
/*  1487*/     OPC_CheckType, MVT::i32,
/*  1489*/     OPC_CheckPatternPredicate, 4, // (Subtarget->hasChapter8_2())
/*  1491*/     OPC_EmitNode1, TARGET_VAL(Cpu0::XOR), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #4
/*  1499*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVZ_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 2, 4, 3, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETEQ:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 6
               // Dst: (MOVZ_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, (XOR:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), CPURegs:{ *:[i32] }:$F)
/*  1508*/    /*Scope*/ 26, /*->1535*/
/*  1509*/     OPC_CheckChild2CondCode, ISD::SETNE,
/*  1511*/     OPC_MoveParent,
/*  1512*/     OPC_RecordChild1, // #2 = $T
/*  1513*/     OPC_RecordChild2, // #3 = $F
/*  1514*/     OPC_CheckType, MVT::i32,
/*  1516*/     OPC_CheckPatternPredicate, 4, // (Subtarget->hasChapter8_2())
/*  1518*/     OPC_EmitNode1, TARGET_VAL(Cpu0::XOR), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #4
/*  1526*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVN_I_I), 0,
                   MVT::i32, 3/*#Ops*/, 2, 4, 3, 
               // Src: (select:{ *:[i32] } (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETNE:{ *:[Other] }), CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 6
               // Dst: (MOVN_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, (XOR:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), CPURegs:{ *:[i32] }:$F)
/*  1535*/    0, /*End of Scope*/
/*  1536*/   0, /*End of Scope*/
/*  1537*/  /*Scope*/ 16, /*->1554*/
/*  1538*/   OPC_RecordChild0, // #0 = $cond
/*  1539*/   OPC_RecordChild1, // #1 = $T
/*  1540*/   OPC_RecordChild2, // #2 = $F
/*  1541*/   OPC_CheckType, MVT::i32,
/*  1543*/   OPC_CheckPatternPredicate, 4, // (Subtarget->hasChapter8_2())
/*  1545*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MOVN_I_I), 0,
                 MVT::i32, 3/*#Ops*/, 1, 0, 2, 
             // Src: (select:{ *:[i32] } CPURegs:{ *:[i32] }:$cond, CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$F) - Complexity = 3
             // Dst: (MOVN_I_I:{ *:[i32] } CPURegs:{ *:[i32] }:$T, CPURegs:{ *:[i32] }:$cond, CPURegs:{ *:[i32] }:$F)
/*  1554*/  0, /*End of Scope*/
/*  1555*/ /*SwitchOpcode*/ 24, TARGET_VAL(ISD::CALLSEQ_START),// ->1582
/*  1558*/  OPC_RecordNode, // #0 = 'callseq_start' chained node
/*  1559*/  OPC_RecordChild1, // #1 = $amt1
/*  1560*/  OPC_MoveChild1,
/*  1561*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/*  1564*/  OPC_MoveParent,
/*  1565*/  OPC_RecordChild2, // #2 = $amt2
/*  1566*/  OPC_MoveChild2,
/*  1567*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/*  1570*/  OPC_MoveParent,
/*  1571*/  OPC_CheckPatternPredicate, 5, // (Subtarget->hasChapter9_2())
/*  1573*/  OPC_EmitMergeInputChains1_0,
/*  1574*/  OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ADJCALLSTACKDOWN), 0|OPFL_Chain|OPFL_GlueOutput,
                MVT::i32, 2/*#Ops*/, 1, 2, 
            // Src: (callseq_start (timm:{ *:[i32] }):$amt1, (timm:{ *:[i32] }):$amt2) - Complexity = 9
            // Dst: (ADJCALLSTACKDOWN:{ *:[i32] } (timm:{ *:[i32] }):$amt1, (timm:{ *:[i32] }):$amt2)
/*  1582*/ /*SwitchOpcode*/ 25, TARGET_VAL(ISD::CALLSEQ_END),// ->1610
/*  1585*/  OPC_RecordNode, // #0 = 'callseq_end' chained node
/*  1586*/  OPC_CaptureGlueInput,
/*  1587*/  OPC_RecordChild1, // #1 = $amt1
/*  1588*/  OPC_MoveChild1,
/*  1589*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/*  1592*/  OPC_MoveParent,
/*  1593*/  OPC_RecordChild2, // #2 = $amt2
/*  1594*/  OPC_MoveChild2,
/*  1595*/  OPC_CheckOpcode, TARGET_VAL(ISD::TargetConstant),
/*  1598*/  OPC_MoveParent,
/*  1599*/  OPC_CheckPatternPredicate, 5, // (Subtarget->hasChapter9_2())
/*  1601*/  OPC_EmitMergeInputChains1_0,
/*  1602*/  OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ADJCALLSTACKUP), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput,
                MVT::i32, 2/*#Ops*/, 1, 2, 
            // Src: (callseq_end (timm:{ *:[i32] }):$amt1, (timm:{ *:[i32] }):$amt2) - Complexity = 9
            // Dst: (ADJCALLSTACKUP:{ *:[i32] } (timm:{ *:[i32] }):$amt1, (timm:{ *:[i32] }):$amt2)
/*  1610*/ /*SwitchOpcode*/ 16, TARGET_VAL(ISD::FrameIndex),// ->1629
/*  1613*/  OPC_RecordNode, // #0 = $addr
/*  1614*/  OPC_CheckType, MVT::i32,
/*  1616*/  OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*  1618*/  OPC_CheckComplexPat, /*CP*/0, /*#*/0, // SelectAddr:$addr #1 #2
/*  1621*/  OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LEA_ADDiu), 0,
                MVT::i32, 2/*#Ops*/, 1, 2, 
            // Src: addr:{ *:[i32] }:$addr - Complexity = 9
            // Dst: (LEA_ADDiu:{ *:[i32] } addr:{ *:[i32] }:$addr)
/*  1629*/ /*SwitchOpcode*/ 127|128,1/*255*/, TARGET_VAL(ISD::ADD),// ->1888
/*  1633*/  OPC_Scope, 101, /*->1736*/ // 3 children in Scope
/*  1635*/   OPC_RecordChild0, // #0 = $hi
/*  1636*/   OPC_MoveChild1,
/*  1637*/   OPC_SwitchOpcode /*2 cases */, 72, TARGET_VAL(Cpu0ISD::Lo),// ->1713
/*  1641*/    OPC_RecordChild0, // #1 = $lo
/*  1642*/    OPC_MoveChild0,
/*  1643*/    OPC_SwitchOpcode /*4 cases */, 14, TARGET_VAL(ISD::TargetGlobalAddress),// ->1661
/*  1647*/     OPC_MoveParent,
/*  1648*/     OPC_MoveParent,
/*  1649*/     OPC_CheckType, MVT::i32,
/*  1651*/     OPC_CheckPatternPredicate, 6, // (Subtarget->hasChapter6_1())
/*  1653*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (add:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (Cpu0Lo:{ *:[i32] } (tglobaladdr:{ *:[i32] }):$lo)) - Complexity = 9
               // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (tglobaladdr:{ *:[i32] }):$lo)
/*  1661*/    /*SwitchOpcode*/ 14, TARGET_VAL(ISD::TargetBlockAddress),// ->1678
/*  1664*/     OPC_MoveParent,
/*  1665*/     OPC_MoveParent,
/*  1666*/     OPC_CheckType, MVT::i32,
/*  1668*/     OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  1670*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (add:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (Cpu0Lo:{ *:[i32] } (tblockaddress:{ *:[i32] }):$lo)) - Complexity = 9
               // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (tblockaddress:{ *:[i32] }):$lo)
/*  1678*/    /*SwitchOpcode*/ 14, TARGET_VAL(ISD::TargetJumpTable),// ->1695
/*  1681*/     OPC_MoveParent,
/*  1682*/     OPC_MoveParent,
/*  1683*/     OPC_CheckType, MVT::i32,
/*  1685*/     OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  1687*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (add:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (Cpu0Lo:{ *:[i32] } (tjumptable:{ *:[i32] }):$lo)) - Complexity = 9
               // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (tjumptable:{ *:[i32] }):$lo)
/*  1695*/    /*SwitchOpcode*/ 14, TARGET_VAL(ISD::TargetGlobalTLSAddress),// ->1712
/*  1698*/     OPC_MoveParent,
/*  1699*/     OPC_MoveParent,
/*  1700*/     OPC_CheckType, MVT::i32,
/*  1702*/     OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  1704*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                   MVT::i32, 2/*#Ops*/, 0, 1, 
               // Src: (add:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (Cpu0Lo:{ *:[i32] } (tglobaltlsaddr:{ *:[i32] }):$lo)) - Complexity = 9
               // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (tglobaltlsaddr:{ *:[i32] }):$lo)
/*  1712*/    0, // EndSwitchOpcode
/*  1713*/   /*SwitchOpcode*/ 19, TARGET_VAL(Cpu0ISD::GPRel),// ->1735
/*  1716*/    OPC_RecordChild0, // #1 = $in
/*  1717*/    OPC_MoveChild0,
/*  1718*/    OPC_CheckOpcode, TARGET_VAL(ISD::TargetGlobalAddress),
/*  1721*/    OPC_MoveParent,
/*  1722*/    OPC_MoveParent,
/*  1723*/    OPC_CheckType, MVT::i32,
/*  1725*/    OPC_CheckPatternPredicate, 6, // (Subtarget->hasChapter6_1())
/*  1727*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (add:{ *:[i32] } CPURegs:{ *:[i32] }:$gp, (Cpu0GPRel:{ *:[i32] } (tglobaladdr:{ *:[i32] }):$in)) - Complexity = 9
              // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$gp, (tglobaladdr:{ *:[i32] }):$in)
/*  1735*/   0, // EndSwitchOpcode
/*  1736*/  /*Scope*/ 105, /*->1842*/
/*  1737*/   OPC_MoveChild0,
/*  1738*/   OPC_SwitchOpcode /*2 cases */, 76, TARGET_VAL(Cpu0ISD::Lo),// ->1818
/*  1742*/    OPC_RecordChild0, // #0 = $lo
/*  1743*/    OPC_MoveChild0,
/*  1744*/    OPC_SwitchOpcode /*4 cases */, 15, TARGET_VAL(ISD::TargetGlobalAddress),// ->1763
/*  1748*/     OPC_MoveParent,
/*  1749*/     OPC_MoveParent,
/*  1750*/     OPC_RecordChild1, // #1 = $hi
/*  1751*/     OPC_CheckType, MVT::i32,
/*  1753*/     OPC_CheckPatternPredicate, 6, // (Subtarget->hasChapter6_1())
/*  1755*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                   MVT::i32, 2/*#Ops*/, 1, 0, 
               // Src: (add:{ *:[i32] } (Cpu0Lo:{ *:[i32] } (tglobaladdr:{ *:[i32] }):$lo), CPURegs:{ *:[i32] }:$hi) - Complexity = 9
               // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (tglobaladdr:{ *:[i32] }):$lo)
/*  1763*/    /*SwitchOpcode*/ 15, TARGET_VAL(ISD::TargetBlockAddress),// ->1781
/*  1766*/     OPC_MoveParent,
/*  1767*/     OPC_MoveParent,
/*  1768*/     OPC_RecordChild1, // #1 = $hi
/*  1769*/     OPC_CheckType, MVT::i32,
/*  1771*/     OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  1773*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                   MVT::i32, 2/*#Ops*/, 1, 0, 
               // Src: (add:{ *:[i32] } (Cpu0Lo:{ *:[i32] } (tblockaddress:{ *:[i32] }):$lo), CPURegs:{ *:[i32] }:$hi) - Complexity = 9
               // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (tblockaddress:{ *:[i32] }):$lo)
/*  1781*/    /*SwitchOpcode*/ 15, TARGET_VAL(ISD::TargetJumpTable),// ->1799
/*  1784*/     OPC_MoveParent,
/*  1785*/     OPC_MoveParent,
/*  1786*/     OPC_RecordChild1, // #1 = $hi
/*  1787*/     OPC_CheckType, MVT::i32,
/*  1789*/     OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  1791*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                   MVT::i32, 2/*#Ops*/, 1, 0, 
               // Src: (add:{ *:[i32] } (Cpu0Lo:{ *:[i32] } (tjumptable:{ *:[i32] }):$lo), CPURegs:{ *:[i32] }:$hi) - Complexity = 9
               // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (tjumptable:{ *:[i32] }):$lo)
/*  1799*/    /*SwitchOpcode*/ 15, TARGET_VAL(ISD::TargetGlobalTLSAddress),// ->1817
/*  1802*/     OPC_MoveParent,
/*  1803*/     OPC_MoveParent,
/*  1804*/     OPC_RecordChild1, // #1 = $hi
/*  1805*/     OPC_CheckType, MVT::i32,
/*  1807*/     OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  1809*/     OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                   MVT::i32, 2/*#Ops*/, 1, 0, 
               // Src: (add:{ *:[i32] } (Cpu0Lo:{ *:[i32] } (tglobaltlsaddr:{ *:[i32] }):$lo), CPURegs:{ *:[i32] }:$hi) - Complexity = 9
               // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$hi, (tglobaltlsaddr:{ *:[i32] }):$lo)
/*  1817*/    0, // EndSwitchOpcode
/*  1818*/   /*SwitchOpcode*/ 20, TARGET_VAL(Cpu0ISD::GPRel),// ->1841
/*  1821*/    OPC_RecordChild0, // #0 = $in
/*  1822*/    OPC_MoveChild0,
/*  1823*/    OPC_CheckOpcode, TARGET_VAL(ISD::TargetGlobalAddress),
/*  1826*/    OPC_MoveParent,
/*  1827*/    OPC_MoveParent,
/*  1828*/    OPC_RecordChild1, // #1 = $gp
/*  1829*/    OPC_CheckType, MVT::i32,
/*  1831*/    OPC_CheckPatternPredicate, 6, // (Subtarget->hasChapter6_1())
/*  1833*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                  MVT::i32, 2/*#Ops*/, 1, 0, 
              // Src: (add:{ *:[i32] } (Cpu0GPRel:{ *:[i32] } (tglobaladdr:{ *:[i32] }):$in), CPURegs:{ *:[i32] }:$gp) - Complexity = 9
              // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$gp, (tglobaladdr:{ *:[i32] }):$in)
/*  1841*/   0, // EndSwitchOpcode
/*  1842*/  /*Scope*/ 44, /*->1887*/
/*  1843*/   OPC_RecordChild0, // #0 = $rb
/*  1844*/   OPC_RecordChild1, // #1 = $imm16
/*  1845*/   OPC_Scope, 17, /*->1864*/ // 3 children in Scope
/*  1847*/    OPC_MoveChild1,
/*  1848*/    OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1851*/    OPC_CheckPredicate, 14, // Predicate_immSExt16
/*  1853*/    OPC_MoveParent,
/*  1854*/    OPC_EmitConvertToTarget, 1,
/*  1856*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ADDiu), 0,
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (add:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$imm16) - Complexity = 7
              // Dst: (ADDiu:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] }):$imm16)
/*  1864*/   /*Scope*/ 10, /*->1875*/
/*  1865*/    OPC_CheckPatternPredicate, 9, // (Subtarget->disableOverflow())
/*  1867*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ADDu), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (add:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
              // Dst: (ADDu:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  1875*/   /*Scope*/ 10, /*->1886*/
/*  1876*/    OPC_CheckPatternPredicate, 10, // (Subtarget->enableOverflow())
/*  1878*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ADD), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (add:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
              // Dst: (ADD:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  1886*/   0, /*End of Scope*/
/*  1887*/  0, /*End of Scope*/
/*  1888*/ /*SwitchOpcode*/ 12, TARGET_VAL(ISD::INTRINSIC_WO_CHAIN),// ->1903
/*  1891*/  OPC_CheckChild0Integer, 49, 
/*  1893*/  OPC_RecordChild1, // #0 = $rb
/*  1894*/  OPC_RecordChild2, // #1 = $rc
/*  1895*/  OPC_MorphNodeTo1, TARGET_VAL(Cpu0::GCD), 0,
                MVT::i32, 2/*#Ops*/, 0, 1, 
            // Src: (intrinsic_wo_chain:{ *:[i32] } 49:{ *:[iPTR] }, CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 8
            // Dst: (GCD:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  1903*/ /*SwitchOpcode*/ 38, TARGET_VAL(ISD::AND),// ->1944
/*  1906*/  OPC_RecordChild0, // #0 = $rb
/*  1907*/  OPC_RecordChild1, // #1 = $imm16
/*  1908*/  OPC_Scope, 22, /*->1932*/ // 2 children in Scope
/*  1910*/   OPC_MoveChild1,
/*  1911*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1914*/   OPC_CheckPredicate, 13, // Predicate_immZExt16
/*  1916*/   OPC_MoveParent,
/*  1917*/   OPC_CheckPatternPredicate, 1, // (Subtarget->hasChapter4_2())
/*  1919*/   OPC_EmitConvertToTarget, 1,
/*  1921*/   OPC_EmitNodeXForm, 0, 2, // LO16
/*  1924*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ANDi), 0,
                 MVT::i32, 2/*#Ops*/, 0, 3, 
             // Src: (and:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immZExt16>><<X:LO16>>:$imm16) - Complexity = 7
             // Dst: (ANDi:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (LO16:{ *:[i32] } (imm:{ *:[i32] }):$imm16))
/*  1932*/  /*Scope*/ 10, /*->1943*/
/*  1933*/   OPC_CheckPatternPredicate, 1, // (Subtarget->hasChapter4_2())
/*  1935*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::AND), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (and:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
             // Dst: (AND:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  1943*/  0, /*End of Scope*/
/*  1944*/ /*SwitchOpcode*/ 38, TARGET_VAL(ISD::OR),// ->1985
/*  1947*/  OPC_RecordChild0, // #0 = $rb
/*  1948*/  OPC_RecordChild1, // #1 = $imm16
/*  1949*/  OPC_Scope, 22, /*->1973*/ // 2 children in Scope
/*  1951*/   OPC_MoveChild1,
/*  1952*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1955*/   OPC_CheckPredicate, 13, // Predicate_immZExt16
/*  1957*/   OPC_MoveParent,
/*  1958*/   OPC_CheckPatternPredicate, 11, // (Subtarget->hasChapter3_5())
/*  1960*/   OPC_EmitConvertToTarget, 1,
/*  1962*/   OPC_EmitNodeXForm, 0, 2, // LO16
/*  1965*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 0, 3, 
             // Src: (or:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immZExt16>><<X:LO16>>:$imm16) - Complexity = 7
             // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (LO16:{ *:[i32] } (imm:{ *:[i32] }):$imm16))
/*  1973*/  /*Scope*/ 10, /*->1984*/
/*  1974*/   OPC_CheckPatternPredicate, 1, // (Subtarget->hasChapter4_2())
/*  1976*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::OR), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (or:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
             // Dst: (OR:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  1984*/  0, /*End of Scope*/
/*  1985*/ /*SwitchOpcode*/ 35, TARGET_VAL(ISD::ROTL),// ->2023
/*  1988*/  OPC_RecordChild0, // #0 = $rb
/*  1989*/  OPC_RecordChild1, // #1 = $shamt
/*  1990*/  OPC_Scope, 19, /*->2011*/ // 2 children in Scope
/*  1992*/   OPC_MoveChild1,
/*  1993*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  1996*/   OPC_CheckPredicate, 15, // Predicate_immZExt5
/*  1998*/   OPC_MoveParent,
/*  1999*/   OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  2001*/   OPC_EmitConvertToTarget, 1,
/*  2003*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ROL), 0,
                 MVT::i32, 2/*#Ops*/, 0, 2, 
             // Src: (rotl:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immZExt5>>:$shamt) - Complexity = 7
             // Dst: (ROL:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] }):$shamt)
/*  2011*/  /*Scope*/ 10, /*->2022*/
/*  2012*/   OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  2014*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ROLV), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (rotl:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
             // Dst: (ROLV:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  2022*/  0, /*End of Scope*/
/*  2023*/ /*SwitchOpcode*/ 35, TARGET_VAL(ISD::ROTR),// ->2061
/*  2026*/  OPC_RecordChild0, // #0 = $rb
/*  2027*/  OPC_RecordChild1, // #1 = $shamt
/*  2028*/  OPC_Scope, 19, /*->2049*/ // 2 children in Scope
/*  2030*/   OPC_MoveChild1,
/*  2031*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2034*/   OPC_CheckPredicate, 15, // Predicate_immZExt5
/*  2036*/   OPC_MoveParent,
/*  2037*/   OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  2039*/   OPC_EmitConvertToTarget, 1,
/*  2041*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ROR), 0,
                 MVT::i32, 2/*#Ops*/, 0, 2, 
             // Src: (rotr:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immZExt5>>:$shamt) - Complexity = 7
             // Dst: (ROR:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] }):$shamt)
/*  2049*/  /*Scope*/ 10, /*->2060*/
/*  2050*/   OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  2052*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::RORV), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (rotr:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
             // Dst: (RORV:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  2060*/  0, /*End of Scope*/
/*  2061*/ /*SwitchOpcode*/ 35, TARGET_VAL(ISD::SHL),// ->2099
/*  2064*/  OPC_RecordChild0, // #0 = $rb
/*  2065*/  OPC_RecordChild1, // #1 = $shamt
/*  2066*/  OPC_Scope, 19, /*->2087*/ // 2 children in Scope
/*  2068*/   OPC_MoveChild1,
/*  2069*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2072*/   OPC_CheckPredicate, 15, // Predicate_immZExt5
/*  2074*/   OPC_MoveParent,
/*  2075*/   OPC_CheckPatternPredicate, 11, // (Subtarget->hasChapter3_5())
/*  2077*/   OPC_EmitConvertToTarget, 1,
/*  2079*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SHL), 0,
                 MVT::i32, 2/*#Ops*/, 0, 2, 
             // Src: (shl:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immZExt5>>:$shamt) - Complexity = 7
             // Dst: (SHL:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] }):$shamt)
/*  2087*/  /*Scope*/ 10, /*->2098*/
/*  2088*/   OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  2090*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SHLV), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (shl:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
             // Dst: (SHLV:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  2098*/  0, /*End of Scope*/
/*  2099*/ /*SwitchOpcode*/ 35, TARGET_VAL(ISD::SRL),// ->2137
/*  2102*/  OPC_RecordChild0, // #0 = $rb
/*  2103*/  OPC_RecordChild1, // #1 = $shamt
/*  2104*/  OPC_Scope, 19, /*->2125*/ // 2 children in Scope
/*  2106*/   OPC_MoveChild1,
/*  2107*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2110*/   OPC_CheckPredicate, 15, // Predicate_immZExt5
/*  2112*/   OPC_MoveParent,
/*  2113*/   OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  2115*/   OPC_EmitConvertToTarget, 1,
/*  2117*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SHR), 0,
                 MVT::i32, 2/*#Ops*/, 0, 2, 
             // Src: (srl:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immZExt5>>:$shamt) - Complexity = 7
             // Dst: (SHR:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] }):$shamt)
/*  2125*/  /*Scope*/ 10, /*->2136*/
/*  2126*/   OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  2128*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SHRV), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (srl:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
             // Dst: (SHRV:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  2136*/  0, /*End of Scope*/
/*  2137*/ /*SwitchOpcode*/ 35, TARGET_VAL(ISD::SRA),// ->2175
/*  2140*/  OPC_RecordChild0, // #0 = $rb
/*  2141*/  OPC_RecordChild1, // #1 = $shamt
/*  2142*/  OPC_Scope, 19, /*->2163*/ // 2 children in Scope
/*  2144*/   OPC_MoveChild1,
/*  2145*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2148*/   OPC_CheckPredicate, 15, // Predicate_immZExt5
/*  2150*/   OPC_MoveParent,
/*  2151*/   OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  2153*/   OPC_EmitConvertToTarget, 1,
/*  2155*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SRA), 0,
                 MVT::i32, 2/*#Ops*/, 0, 2, 
             // Src: (sra:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immZExt5>>:$shamt) - Complexity = 7
             // Dst: (SRA:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] }):$shamt)
/*  2163*/  /*Scope*/ 10, /*->2174*/
/*  2164*/   OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  2166*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SRAV), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (sra:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
             // Dst: (SRAV:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  2174*/  0, /*End of Scope*/
/*  2175*/ /*SwitchOpcode*/ 104|128,4/*616*/, TARGET_VAL(ISD::SETCC),// ->2795
/*  2179*/  OPC_RecordChild0, // #0 = $rb
/*  2180*/  OPC_Scope, 92, /*->2274*/ // 2 children in Scope
/*  2182*/   OPC_RecordChild1, // #1 = $imm16
/*  2183*/   OPC_MoveChild1,
/*  2184*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2187*/   OPC_CheckPredicate, 14, // Predicate_immSExt16
/*  2189*/   OPC_MoveParent,
/*  2190*/   OPC_Scope, 14, /*->2206*/ // 4 children in Scope
/*  2192*/    OPC_CheckChild2CondCode, ISD::SETLT,
/*  2194*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2196*/    OPC_EmitConvertToTarget, 1,
/*  2198*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SLTi), 0,
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$imm16, SETLT:{ *:[Other] }) - Complexity = 7
              // Dst: (SLTi:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] }):$imm16)
/*  2206*/   /*Scope*/ 14, /*->2221*/
/*  2207*/    OPC_CheckChild2CondCode, ISD::SETULT,
/*  2209*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2211*/    OPC_EmitConvertToTarget, 1,
/*  2213*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SLTiu), 0,
                  MVT::i32, 2/*#Ops*/, 0, 2, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$imm16, SETULT:{ *:[Other] }) - Complexity = 7
              // Dst: (SLTiu:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, (imm:{ *:[i32] }):$imm16)
/*  2221*/   /*Scope*/ 25, /*->2247*/
/*  2222*/    OPC_CheckChild2CondCode, ISD::SETGE,
/*  2224*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2226*/    OPC_EmitConvertToTarget, 1,
/*  2228*/    OPC_EmitNode1, TARGET_VAL(Cpu0::SLTi), 0,
                  MVT::i32, 2/*#Ops*/, 0, 2,  // Results = #3
/*  2236*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2239*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 3, 4, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$rhs, SETGE:{ *:[Other] }) - Complexity = 7
              // Dst: (XORi:{ *:[i32] } (SLTi:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$rhs), 1:{ *:[i32] })
/*  2247*/   /*Scope*/ 25, /*->2273*/
/*  2248*/    OPC_CheckChild2CondCode, ISD::SETUGE,
/*  2250*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2252*/    OPC_EmitConvertToTarget, 1,
/*  2254*/    OPC_EmitNode1, TARGET_VAL(Cpu0::SLTiu), 0,
                  MVT::i32, 2/*#Ops*/, 0, 2,  // Results = #3
/*  2262*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2265*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 3, 4, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$rhs, SETUGE:{ *:[Other] }) - Complexity = 7
              // Dst: (XORi:{ *:[i32] } (SLTiu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$rhs), 1:{ *:[i32] })
/*  2273*/   0, /*End of Scope*/
/*  2274*/  /*Scope*/ 6|128,4/*518*/, /*->2794*/
/*  2276*/   OPC_CheckChild0Type, MVT::i32,
/*  2278*/   OPC_RecordChild1, // #1 = $rc
/*  2279*/   OPC_Scope, 12, /*->2293*/ // 20 children in Scope
/*  2281*/    OPC_CheckChild2CondCode, ISD::SETLT,
/*  2283*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2285*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SLT), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc, SETLT:{ *:[Other] }) - Complexity = 3
              // Dst: (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  2293*/   /*Scope*/ 12, /*->2306*/
/*  2294*/    OPC_CheckChild2CondCode, ISD::SETULT,
/*  2296*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2298*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SLTu), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc, SETULT:{ *:[Other] }) - Complexity = 3
              // Dst: (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  2306*/   /*Scope*/ 12, /*->2319*/
/*  2307*/    OPC_CheckChild2CondCode, ISD::SETGT,
/*  2309*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2311*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SLT), 0,
                  MVT::i32, 2/*#Ops*/, 1, 0, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETGT:{ *:[Other] }) - Complexity = 3
              // Dst: (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs)
/*  2319*/   /*Scope*/ 12, /*->2332*/
/*  2320*/    OPC_CheckChild2CondCode, ISD::SETUGT,
/*  2322*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2324*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SLTu), 0,
                  MVT::i32, 2/*#Ops*/, 1, 0, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUGT:{ *:[Other] }) - Complexity = 3
              // Dst: (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs)
/*  2332*/   /*Scope*/ 23, /*->2356*/
/*  2333*/    OPC_CheckChild2CondCode, ISD::SETEQ,
/*  2335*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2337*/    OPC_EmitNode1, TARGET_VAL(Cpu0::XOR), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2345*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2348*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SLTiu), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETEQ:{ *:[Other] }) - Complexity = 3
              // Dst: (SLTiu:{ *:[i32] } (XOR:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), 1:{ *:[i32] })
/*  2356*/   /*Scope*/ 23, /*->2380*/
/*  2357*/    OPC_CheckChild2CondCode, ISD::SETNE,
/*  2359*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2361*/    OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*  2364*/    OPC_EmitNode1, TARGET_VAL(Cpu0::XOR), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #3
/*  2372*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SLTu), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETNE:{ *:[Other] }) - Complexity = 3
              // Dst: (SLTu:{ *:[i32] } ZERO:{ *:[i32] }, (XOR:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs))
/*  2380*/   /*Scope*/ 23, /*->2404*/
/*  2381*/    OPC_CheckChild2CondCode, ISD::SETLE,
/*  2383*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2385*/    OPC_EmitNode1, TARGET_VAL(Cpu0::SLT), 0,
                  MVT::i32, 2/*#Ops*/, 1, 0,  // Results = #2
/*  2393*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2396*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETLE:{ *:[Other] }) - Complexity = 3
              // Dst: (XORi:{ *:[i32] } (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), 1:{ *:[i32] })
/*  2404*/   /*Scope*/ 23, /*->2428*/
/*  2405*/    OPC_CheckChild2CondCode, ISD::SETULE,
/*  2407*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2409*/    OPC_EmitNode1, TARGET_VAL(Cpu0::SLTu), 0,
                  MVT::i32, 2/*#Ops*/, 1, 0,  // Results = #2
/*  2417*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2420*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETULE:{ *:[Other] }) - Complexity = 3
              // Dst: (XORi:{ *:[i32] } (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), 1:{ *:[i32] })
/*  2428*/   /*Scope*/ 23, /*->2452*/
/*  2429*/    OPC_CheckChild2CondCode, ISD::SETGE,
/*  2431*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2433*/    OPC_EmitNode1, TARGET_VAL(Cpu0::SLT), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2441*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2444*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETGE:{ *:[Other] }) - Complexity = 3
              // Dst: (XORi:{ *:[i32] } (SLT:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), 1:{ *:[i32] })
/*  2452*/   /*Scope*/ 23, /*->2476*/
/*  2453*/    OPC_CheckChild2CondCode, ISD::SETUGE,
/*  2455*/    OPC_CheckPatternPredicate, 2, // (Subtarget->hasSlt())
/*  2457*/    OPC_EmitNode1, TARGET_VAL(Cpu0::SLTu), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2465*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2468*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUGE:{ *:[Other] }) - Complexity = 3
              // Dst: (XORi:{ *:[i32] } (SLTu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), 1:{ *:[i32] })
/*  2476*/   /*Scope*/ 23, /*->2500*/
/*  2477*/    OPC_CheckChild2CondCode, ISD::SETLT,
/*  2479*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2481*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2489*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2492*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETLT:{ *:[Other] }) - Complexity = 3
              // Dst: (ANDi:{ *:[i32] } (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), 1:{ *:[i32] })
/*  2500*/   /*Scope*/ 23, /*->2524*/
/*  2501*/    OPC_CheckChild2CondCode, ISD::SETULT,
/*  2503*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2505*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMPu), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2513*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2516*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETULT:{ *:[Other] }) - Complexity = 3
              // Dst: (ANDi:{ *:[i32] } (CMPu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), 1:{ *:[i32] })
/*  2524*/   /*Scope*/ 23, /*->2548*/
/*  2525*/    OPC_CheckChild2CondCode, ISD::SETGT,
/*  2527*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2529*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0,
                  MVT::i32, 2/*#Ops*/, 1, 0,  // Results = #2
/*  2537*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2540*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETGT:{ *:[Other] }) - Complexity = 3
              // Dst: (ANDi:{ *:[i32] } (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), 1:{ *:[i32] })
/*  2548*/   /*Scope*/ 23, /*->2572*/
/*  2549*/    OPC_CheckChild2CondCode, ISD::SETUGT,
/*  2551*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2553*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMPu), 0,
                  MVT::i32, 2/*#Ops*/, 1, 0,  // Results = #2
/*  2561*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2564*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUGT:{ *:[Other] }) - Complexity = 3
              // Dst: (ANDi:{ *:[i32] } (CMPu:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), 1:{ *:[i32] })
/*  2572*/   /*Scope*/ 34, /*->2607*/
/*  2573*/    OPC_CheckChild2CondCode, ISD::SETEQ,
/*  2575*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2577*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2585*/    OPC_EmitInteger, MVT::i32, 2, 
/*  2588*/    OPC_EmitNode1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #4
/*  2596*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2599*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SHR), 0,
                  MVT::i32, 2/*#Ops*/, 4, 5, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETEQ:{ *:[Other] }) - Complexity = 3
              // Dst: (SHR:{ *:[i32] } (ANDi:{ *:[i32] } (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), 2:{ *:[i32] }), 1:{ *:[i32] })
/*  2607*/   /*Scope*/ 34, /*->2642*/
/*  2608*/    OPC_CheckChild2CondCode, ISD::SETLE,
/*  2610*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2612*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0,
                  MVT::i32, 2/*#Ops*/, 1, 0,  // Results = #2
/*  2620*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2623*/    OPC_EmitNode1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #4
/*  2631*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2634*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 4, 5, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETLE:{ *:[Other] }) - Complexity = 3
              // Dst: (XORi:{ *:[i32] } (ANDi:{ *:[i32] } (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), 1:{ *:[i32] }), 1:{ *:[i32] })
/*  2642*/   /*Scope*/ 34, /*->2677*/
/*  2643*/    OPC_CheckChild2CondCode, ISD::SETULE,
/*  2645*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2647*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0,
                  MVT::i32, 2/*#Ops*/, 1, 0,  // Results = #2
/*  2655*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2658*/    OPC_EmitNode1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #4
/*  2666*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2669*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 4, 5, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETULE:{ *:[Other] }) - Complexity = 3
              // Dst: (XORi:{ *:[i32] } (ANDi:{ *:[i32] } (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$rhs, CPURegs:{ *:[i32] }:$lhs), 1:{ *:[i32] }), 1:{ *:[i32] })
/*  2677*/   /*Scope*/ 34, /*->2712*/
/*  2678*/    OPC_CheckChild2CondCode, ISD::SETGE,
/*  2680*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2682*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2690*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2693*/    OPC_EmitNode1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #4
/*  2701*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2704*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 4, 5, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETGE:{ *:[Other] }) - Complexity = 3
              // Dst: (XORi:{ *:[i32] } (ANDi:{ *:[i32] } (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), 1:{ *:[i32] }), 1:{ *:[i32] })
/*  2712*/   /*Scope*/ 34, /*->2747*/
/*  2713*/    OPC_CheckChild2CondCode, ISD::SETUGE,
/*  2715*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2717*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMPu), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2725*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2728*/    OPC_EmitNode1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #4
/*  2736*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2739*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 4, 5, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETUGE:{ *:[Other] }) - Complexity = 3
              // Dst: (XORi:{ *:[i32] } (ANDi:{ *:[i32] } (CMPu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), 1:{ *:[i32] }), 1:{ *:[i32] })
/*  2747*/   /*Scope*/ 45, /*->2793*/
/*  2748*/    OPC_CheckChild2CondCode, ISD::SETNE,
/*  2750*/    OPC_CheckPatternPredicate, 3, // (Subtarget->hasCmp())
/*  2752*/    OPC_EmitNode1, TARGET_VAL(Cpu0::CMP), 0,
                  MVT::i32, 2/*#Ops*/, 0, 1,  // Results = #2
/*  2760*/    OPC_EmitInteger, MVT::i32, 2, 
/*  2763*/    OPC_EmitNode1, TARGET_VAL(Cpu0::ANDi), 0,
                  MVT::i32, 2/*#Ops*/, 2, 3,  // Results = #4
/*  2771*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2774*/    OPC_EmitNode1, TARGET_VAL(Cpu0::SHR), 0,
                  MVT::i32, 2/*#Ops*/, 4, 5,  // Results = #6
/*  2782*/    OPC_EmitInteger, MVT::i32, 1, 
/*  2785*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::XORi), 0,
                  MVT::i32, 2/*#Ops*/, 6, 7, 
              // Src: (setcc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs, SETNE:{ *:[Other] }) - Complexity = 3
              // Dst: (XORi:{ *:[i32] } (SHR:{ *:[i32] } (ANDi:{ *:[i32] } (CMP:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs), 2:{ *:[i32] }), 1:{ *:[i32] }), 1:{ *:[i32] })
/*  2793*/   0, /*End of Scope*/
/*  2794*/  0, /*End of Scope*/
/*  2795*/ /*SwitchOpcode*/ 35, TARGET_VAL(ISD::ADDC),// ->2833
/*  2798*/  OPC_RecordChild0, // #0 = $src
/*  2799*/  OPC_RecordChild1, // #1 = $imm
/*  2800*/  OPC_Scope, 19, /*->2821*/ // 2 children in Scope
/*  2802*/   OPC_MoveChild1,
/*  2803*/   OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2806*/   OPC_CheckPredicate, 14, // Predicate_immSExt16
/*  2808*/   OPC_MoveParent,
/*  2809*/   OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*  2811*/   OPC_EmitConvertToTarget, 1,
/*  2813*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ADDiu), 0|OPFL_GlueOutput,
                 MVT::i32, 2/*#Ops*/, 0, 2, 
             // Src: (addc:{ *:[i32] } CPURegs:{ *:[i32] }:$src, (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$imm) - Complexity = 7
             // Dst: (ADDiu:{ *:[i32] } CPURegs:{ *:[i32] }:$src, (imm:{ *:[i32] }):$imm)
/*  2821*/  /*Scope*/ 10, /*->2832*/
/*  2822*/   OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*  2824*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ADDu), 0|OPFL_GlueOutput,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (addc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs) - Complexity = 3
             // Dst: (ADDu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs)
/*  2832*/  0, /*End of Scope*/
/*  2833*/ /*SwitchOpcode*/ 18, TARGET_VAL(Cpu0ISD::Sync),// ->2854
/*  2836*/  OPC_RecordNode, // #0 = 'Cpu0Sync' chained node
/*  2837*/  OPC_RecordChild1, // #1 = $stype
/*  2838*/  OPC_MoveChild1,
/*  2839*/  OPC_CheckOpcode, TARGET_VAL(ISD::Constant),
/*  2842*/  OPC_MoveParent,
/*  2843*/  OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  2845*/  OPC_EmitMergeInputChains1_0,
/*  2846*/  OPC_EmitConvertToTarget, 1,
/*  2848*/  OPC_MorphNodeTo0, TARGET_VAL(Cpu0::SYNC), 0|OPFL_Chain,
                1/*#Ops*/, 2, 
            // Src: (Cpu0Sync (imm:{ *:[i32] }):$stype) - Complexity = 6
            // Dst: (SYNC (imm:{ *:[i32] }):$stype)
/*  2854*/ /*SwitchOpcode*/ 66, TARGET_VAL(Cpu0ISD::JmpLink),// ->2923
/*  2857*/  OPC_RecordNode, // #0 = 'Cpu0JmpLink' chained node
/*  2858*/  OPC_CaptureGlueInput,
/*  2859*/  OPC_RecordChild1, // #1 = $target
/*  2860*/  OPC_Scope, 48, /*->2910*/ // 2 children in Scope
/*  2862*/   OPC_MoveChild1,
/*  2863*/   OPC_SwitchOpcode /*3 cases */, 12, TARGET_VAL(ISD::Constant),// ->2879
/*  2867*/    OPC_MoveParent,
/*  2868*/    OPC_CheckPatternPredicate, 13, // (Subtarget->hasChapter9_1())
/*  2870*/    OPC_EmitMergeInputChains1_0,
/*  2871*/    OPC_EmitConvertToTarget, 1,
/*  2873*/    OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JSUB), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                  1/*#Ops*/, 2, 
              // Src: (Cpu0JmpLink (imm:{ *:[i32] }):$target) - Complexity = 6
              // Dst: (JSUB (imm:{ *:[i32] }):$target)
/*  2879*/   /*SwitchOpcode*/ 12, TARGET_VAL(ISD::TargetGlobalAddress),// ->2894
/*  2882*/    OPC_CheckType, MVT::i32,
/*  2884*/    OPC_MoveParent,
/*  2885*/    OPC_CheckPatternPredicate, 13, // (Subtarget->hasChapter9_1())
/*  2887*/    OPC_EmitMergeInputChains1_0,
/*  2888*/    OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JSUB), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                  1/*#Ops*/, 1, 
              // Src: (Cpu0JmpLink (tglobaladdr:{ *:[i32] }):$dst) - Complexity = 6
              // Dst: (JSUB (tglobaladdr:{ *:[i32] }):$dst)
/*  2894*/   /*SwitchOpcode*/ 12, TARGET_VAL(ISD::TargetExternalSymbol),// ->2909
/*  2897*/    OPC_CheckType, MVT::i32,
/*  2899*/    OPC_MoveParent,
/*  2900*/    OPC_CheckPatternPredicate, 13, // (Subtarget->hasChapter9_1())
/*  2902*/    OPC_EmitMergeInputChains1_0,
/*  2903*/    OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JSUB), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                  1/*#Ops*/, 1, 
              // Src: (Cpu0JmpLink (texternalsym:{ *:[i32] }):$dst) - Complexity = 6
              // Dst: (JSUB (texternalsym:{ *:[i32] }):$dst)
/*  2909*/   0, // EndSwitchOpcode
/*  2910*/  /*Scope*/ 11, /*->2922*/
/*  2911*/   OPC_CheckChild1Type, MVT::i32,
/*  2913*/   OPC_CheckPatternPredicate, 13, // (Subtarget->hasChapter9_1())
/*  2915*/   OPC_EmitMergeInputChains1_0,
/*  2916*/   OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JALR), 0|OPFL_Chain|OPFL_GlueInput|OPFL_GlueOutput|OPFL_Variadic1,
                 1/*#Ops*/, 1, 
             // Src: (Cpu0JmpLink GPROut:{ *:[i32] }:$rb) - Complexity = 3
             // Dst: (JALR GPROut:{ *:[i32] }:$rb)
/*  2922*/  0, /*End of Scope*/
/*  2923*/ /*SwitchOpcode*/ 50, TARGET_VAL(Cpu0ISD::TailCall),// ->2976
/*  2926*/  OPC_RecordNode, // #0 = 'Cpu0TailCall' chained node
/*  2927*/  OPC_CaptureGlueInput,
/*  2928*/  OPC_RecordChild1, // #1 = $dst
/*  2929*/  OPC_Scope, 31, /*->2962*/ // 2 children in Scope
/*  2931*/   OPC_MoveChild1,
/*  2932*/   OPC_SwitchOpcode /*2 cases */, 11, TARGET_VAL(ISD::TargetGlobalAddress),// ->2947
/*  2936*/    OPC_MoveParent,
/*  2937*/    OPC_CheckPatternPredicate, 13, // (Subtarget->hasChapter9_1())
/*  2939*/    OPC_EmitMergeInputChains1_0,
/*  2940*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::TAILCALL), 0|OPFL_Chain|OPFL_GlueInput|OPFL_Variadic1,
                  MVT::i32, 1/*#Ops*/, 1, 
              // Src: (Cpu0TailCall (tglobaladdr:{ *:[iPTR] }):$dst) - Complexity = 6
              // Dst: (TAILCALL:{ *:[i32] } (tglobaladdr:{ *:[iPTR] }):$dst)
/*  2947*/   /*SwitchOpcode*/ 11, TARGET_VAL(ISD::TargetExternalSymbol),// ->2961
/*  2950*/    OPC_MoveParent,
/*  2951*/    OPC_CheckPatternPredicate, 13, // (Subtarget->hasChapter9_1())
/*  2953*/    OPC_EmitMergeInputChains1_0,
/*  2954*/    OPC_MorphNodeTo1, TARGET_VAL(Cpu0::TAILCALL), 0|OPFL_Chain|OPFL_GlueInput|OPFL_Variadic1,
                  MVT::i32, 1/*#Ops*/, 1, 
              // Src: (Cpu0TailCall (texternalsym:{ *:[iPTR] }):$dst) - Complexity = 6
              // Dst: (TAILCALL:{ *:[i32] } (texternalsym:{ *:[iPTR] }):$dst)
/*  2961*/   0, // EndSwitchOpcode
/*  2962*/  /*Scope*/ 12, /*->2975*/
/*  2963*/   OPC_CheckChild1Type, MVT::i32,
/*  2965*/   OPC_CheckPatternPredicate, 13, // (Subtarget->hasChapter9_1())
/*  2967*/   OPC_EmitMergeInputChains1_0,
/*  2968*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::TAILCALL_R), 0|OPFL_Chain|OPFL_GlueInput|OPFL_Variadic1,
                 MVT::i32, 1/*#Ops*/, 1, 
             // Src: (Cpu0TailCall GPROut:{ *:[i32] }:$rs) - Complexity = 3
             // Dst: (TAILCALL_R:{ *:[i32] } GPROut:{ *:[i32] }:$rs)
/*  2975*/  0, /*End of Scope*/
/*  2976*/ /*SwitchOpcode*/ 64, TARGET_VAL(Cpu0ISD::Hi),// ->3043
/*  2979*/  OPC_RecordChild0, // #0 = $in
/*  2980*/  OPC_MoveChild0,
/*  2981*/  OPC_SwitchOpcode /*4 cases */, 12, TARGET_VAL(ISD::TargetGlobalAddress),// ->2997
/*  2985*/   OPC_MoveParent,
/*  2986*/   OPC_CheckType, MVT::i32,
/*  2988*/   OPC_CheckPatternPredicate, 6, // (Subtarget->hasChapter6_1())
/*  2990*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LUi), 0,
                 MVT::i32, 1/*#Ops*/, 0, 
             // Src: (Cpu0Hi:{ *:[i32] } (tglobaladdr:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (LUi:{ *:[i32] } (tglobaladdr:{ *:[i32] }):$in)
/*  2997*/  /*SwitchOpcode*/ 12, TARGET_VAL(ISD::TargetBlockAddress),// ->3012
/*  3000*/   OPC_MoveParent,
/*  3001*/   OPC_CheckType, MVT::i32,
/*  3003*/   OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  3005*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LUi), 0,
                 MVT::i32, 1/*#Ops*/, 0, 
             // Src: (Cpu0Hi:{ *:[i32] } (tblockaddress:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (LUi:{ *:[i32] } (tblockaddress:{ *:[i32] }):$in)
/*  3012*/  /*SwitchOpcode*/ 12, TARGET_VAL(ISD::TargetJumpTable),// ->3027
/*  3015*/   OPC_MoveParent,
/*  3016*/   OPC_CheckType, MVT::i32,
/*  3018*/   OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  3020*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LUi), 0,
                 MVT::i32, 1/*#Ops*/, 0, 
             // Src: (Cpu0Hi:{ *:[i32] } (tjumptable:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (LUi:{ *:[i32] } (tjumptable:{ *:[i32] }):$in)
/*  3027*/  /*SwitchOpcode*/ 12, TARGET_VAL(ISD::TargetGlobalTLSAddress),// ->3042
/*  3030*/   OPC_MoveParent,
/*  3031*/   OPC_CheckType, MVT::i32,
/*  3033*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3035*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LUi), 0,
                 MVT::i32, 1/*#Ops*/, 0, 
             // Src: (Cpu0Hi:{ *:[i32] } (tglobaltlsaddr:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (LUi:{ *:[i32] } (tglobaltlsaddr:{ *:[i32] }):$in)
/*  3042*/  0, // EndSwitchOpcode
/*  3043*/ /*SwitchOpcode*/ 80, TARGET_VAL(Cpu0ISD::Lo),// ->3126
/*  3046*/  OPC_RecordChild0, // #0 = $in
/*  3047*/  OPC_MoveChild0,
/*  3048*/  OPC_SwitchOpcode /*4 cases */, 16, TARGET_VAL(ISD::TargetGlobalAddress),// ->3068
/*  3052*/   OPC_MoveParent,
/*  3053*/   OPC_CheckType, MVT::i32,
/*  3055*/   OPC_CheckPatternPredicate, 6, // (Subtarget->hasChapter6_1())
/*  3057*/   OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*  3060*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 1, 0, 
             // Src: (Cpu0Lo:{ *:[i32] } (tglobaladdr:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (ORi:{ *:[i32] } ZERO:{ *:[i32] }, (tglobaladdr:{ *:[i32] }):$in)
/*  3068*/  /*SwitchOpcode*/ 16, TARGET_VAL(ISD::TargetBlockAddress),// ->3087
/*  3071*/   OPC_MoveParent,
/*  3072*/   OPC_CheckType, MVT::i32,
/*  3074*/   OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  3076*/   OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*  3079*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 1, 0, 
             // Src: (Cpu0Lo:{ *:[i32] } (tblockaddress:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (ORi:{ *:[i32] } ZERO:{ *:[i32] }, (tblockaddress:{ *:[i32] }):$in)
/*  3087*/  /*SwitchOpcode*/ 16, TARGET_VAL(ISD::TargetJumpTable),// ->3106
/*  3090*/   OPC_MoveParent,
/*  3091*/   OPC_CheckType, MVT::i32,
/*  3093*/   OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  3095*/   OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*  3098*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 1, 0, 
             // Src: (Cpu0Lo:{ *:[i32] } (tjumptable:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (ORi:{ *:[i32] } ZERO:{ *:[i32] }, (tjumptable:{ *:[i32] }):$in)
/*  3106*/  /*SwitchOpcode*/ 16, TARGET_VAL(ISD::TargetGlobalTLSAddress),// ->3125
/*  3109*/   OPC_MoveParent,
/*  3110*/   OPC_CheckType, MVT::i32,
/*  3112*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3114*/   OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*  3117*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 1, 0, 
             // Src: (Cpu0Lo:{ *:[i32] } (tglobaltlsaddr:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (ORi:{ *:[i32] } ZERO:{ *:[i32] }, (tglobaltlsaddr:{ *:[i32] }):$in)
/*  3125*/  0, // EndSwitchOpcode
/*  3126*/ /*SwitchOpcode*/ 53, TARGET_VAL(Cpu0ISD::Wrapper),// ->3182
/*  3129*/  OPC_RecordChild0, // #0 = $gp
/*  3130*/  OPC_RecordChild1, // #1 = $in
/*  3131*/  OPC_MoveChild1,
/*  3132*/  OPC_SwitchOpcode /*3 cases */, 13, TARGET_VAL(ISD::TargetGlobalAddress),// ->3149
/*  3136*/   OPC_MoveParent,
/*  3137*/   OPC_CheckType, MVT::i32,
/*  3139*/   OPC_CheckPatternPredicate, 6, // (Subtarget->hasChapter6_1())
/*  3141*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (Cpu0Wrapper:{ *:[i32] } GPROut:{ *:[i32] }:$gp, (tglobaladdr:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (ORi:{ *:[i32] } GPROut:{ *:[i32] }:$gp, (tglobaladdr:{ *:[i32] }):$in)
/*  3149*/  /*SwitchOpcode*/ 13, TARGET_VAL(ISD::TargetJumpTable),// ->3165
/*  3152*/   OPC_MoveParent,
/*  3153*/   OPC_CheckType, MVT::i32,
/*  3155*/   OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  3157*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (Cpu0Wrapper:{ *:[i32] } GPROut:{ *:[i32] }:$gp, (tjumptable:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (ORi:{ *:[i32] } GPROut:{ *:[i32] }:$gp, (tjumptable:{ *:[i32] }):$in)
/*  3165*/  /*SwitchOpcode*/ 13, TARGET_VAL(ISD::TargetGlobalTLSAddress),// ->3181
/*  3168*/   OPC_MoveParent,
/*  3169*/   OPC_CheckType, MVT::i32,
/*  3171*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3173*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (Cpu0Wrapper:{ *:[i32] } CPURegs:{ *:[i32] }:$gp, (tglobaltlsaddr:{ *:[i32] }):$in) - Complexity = 6
             // Dst: (ORi:{ *:[i32] } CPURegs:{ *:[i32] }:$gp, (tglobaltlsaddr:{ *:[i32] }):$in)
/*  3181*/  0, // EndSwitchOpcode
/*  3182*/ /*SwitchOpcode*/ 82, TARGET_VAL(ISD::Constant),// ->3267
/*  3185*/  OPC_RecordNode, // #0 = $in
/*  3186*/  OPC_Scope, 15, /*->3203*/ // 4 children in Scope
/*  3188*/   OPC_CheckPredicate, 14, // Predicate_immSExt16
/*  3190*/   OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*  3193*/   OPC_EmitConvertToTarget, 0,
/*  3195*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ADDiu), 0,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (imm:{ *:[i32] })<<P:Predicate_immSExt16>>:$in - Complexity = 4
             // Dst: (ADDiu:{ *:[i32] } ZERO:{ *:[i32] }, (imm:{ *:[i32] }):$in)
/*  3203*/  /*Scope*/ 17, /*->3221*/
/*  3204*/   OPC_CheckPredicate, 13, // Predicate_immZExt16
/*  3206*/   OPC_CheckPatternPredicate, 11, // (Subtarget->hasChapter3_5())
/*  3208*/   OPC_EmitRegister, MVT::i32, Cpu0::ZERO,
/*  3211*/   OPC_EmitConvertToTarget, 0,
/*  3213*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (imm:{ *:[i32] })<<P:Predicate_immZExt16>><<X:LO16>>:$in - Complexity = 4
             // Dst: (ORi:{ *:[i32] } ZERO:{ *:[i32] }, (imm:{ *:[i32] }):$in)
/*  3221*/  /*Scope*/ 16, /*->3238*/
/*  3222*/   OPC_CheckPredicate, 16, // Predicate_immLow16Zero
/*  3224*/   OPC_CheckPatternPredicate, 11, // (Subtarget->hasChapter3_5())
/*  3226*/   OPC_EmitConvertToTarget, 0,
/*  3228*/   OPC_EmitNodeXForm, 1, 1, // HI16
/*  3231*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::LUi), 0,
                 MVT::i32, 1/*#Ops*/, 2, 
             // Src: (imm:{ *:[i32] })<<P:Predicate_immLow16Zero>>:$in - Complexity = 4
             // Dst: (LUi:{ *:[i32] } (HI16:{ *:[i32] } (imm:{ *:[i32] }):$in))
/*  3238*/  /*Scope*/ 27, /*->3266*/
/*  3239*/   OPC_CheckPatternPredicate, 11, // (Subtarget->hasChapter3_5())
/*  3241*/   OPC_EmitConvertToTarget, 0,
/*  3243*/   OPC_EmitNodeXForm, 1, 1, // HI16
/*  3246*/   OPC_EmitNode1, TARGET_VAL(Cpu0::LUi), 0,
                 MVT::i32, 1/*#Ops*/, 2,  // Results = #3
/*  3253*/   OPC_EmitConvertToTarget, 0,
/*  3255*/   OPC_EmitNodeXForm, 0, 4, // LO16
/*  3258*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ORi), 0,
                 MVT::i32, 2/*#Ops*/, 3, 5, 
             // Src: (imm:{ *:[i32] }):$imm - Complexity = 3
             // Dst: (ORi:{ *:[i32] } (LUi:{ *:[i32] } (HI16:{ *:[i32] } (imm:{ *:[i32] }):$imm)), (LO16:{ *:[i32] } (imm:{ *:[i32] }):$imm))
/*  3266*/  0, /*End of Scope*/
/*  3267*/ /*SwitchOpcode*/ 48, TARGET_VAL(ISD::ATOMIC_LOAD_ADD),// ->3318
/*  3270*/  OPC_RecordMemRef,
/*  3271*/  OPC_RecordNode, // #0 = 'atomic_load_add' chained node
/*  3272*/  OPC_RecordChild1, // #1 = $ptr
/*  3273*/  OPC_RecordChild2, // #2 = $incr
/*  3274*/  OPC_Scope, 13, /*->3289*/ // 3 children in Scope
/*  3276*/   OPC_CheckPredicate, 4, // Predicate_atomic_load_add_8
/*  3278*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3280*/   OPC_EmitMergeInputChains1_0,
/*  3281*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_ADD_I8), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_add:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_add_8>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_ADD_I8:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3289*/  /*Scope*/ 13, /*->3303*/
/*  3290*/   OPC_CheckPredicate, 6, // Predicate_atomic_load_add_16
/*  3292*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3294*/   OPC_EmitMergeInputChains1_0,
/*  3295*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_ADD_I16), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_add:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_add_16>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_ADD_I16:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3303*/  /*Scope*/ 13, /*->3317*/
/*  3304*/   OPC_CheckPredicate, 17, // Predicate_atomic_load_add_32
/*  3306*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3308*/   OPC_EmitMergeInputChains1_0,
/*  3309*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_ADD_I32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_add:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_add_32>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_ADD_I32:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3317*/  0, /*End of Scope*/
/*  3318*/ /*SwitchOpcode*/ 48, TARGET_VAL(ISD::ATOMIC_LOAD_SUB),// ->3369
/*  3321*/  OPC_RecordMemRef,
/*  3322*/  OPC_RecordNode, // #0 = 'atomic_load_sub' chained node
/*  3323*/  OPC_RecordChild1, // #1 = $ptr
/*  3324*/  OPC_RecordChild2, // #2 = $incr
/*  3325*/  OPC_Scope, 13, /*->3340*/ // 3 children in Scope
/*  3327*/   OPC_CheckPredicate, 4, // Predicate_atomic_load_sub_8
/*  3329*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3331*/   OPC_EmitMergeInputChains1_0,
/*  3332*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_SUB_I8), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_sub:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_sub_8>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_SUB_I8:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3340*/  /*Scope*/ 13, /*->3354*/
/*  3341*/   OPC_CheckPredicate, 6, // Predicate_atomic_load_sub_16
/*  3343*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3345*/   OPC_EmitMergeInputChains1_0,
/*  3346*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_SUB_I16), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_sub:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_sub_16>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_SUB_I16:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3354*/  /*Scope*/ 13, /*->3368*/
/*  3355*/   OPC_CheckPredicate, 17, // Predicate_atomic_load_sub_32
/*  3357*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3359*/   OPC_EmitMergeInputChains1_0,
/*  3360*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_SUB_I32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_sub:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_sub_32>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_SUB_I32:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3368*/  0, /*End of Scope*/
/*  3369*/ /*SwitchOpcode*/ 48, TARGET_VAL(ISD::ATOMIC_LOAD_AND),// ->3420
/*  3372*/  OPC_RecordMemRef,
/*  3373*/  OPC_RecordNode, // #0 = 'atomic_load_and' chained node
/*  3374*/  OPC_RecordChild1, // #1 = $ptr
/*  3375*/  OPC_RecordChild2, // #2 = $incr
/*  3376*/  OPC_Scope, 13, /*->3391*/ // 3 children in Scope
/*  3378*/   OPC_CheckPredicate, 4, // Predicate_atomic_load_and_8
/*  3380*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3382*/   OPC_EmitMergeInputChains1_0,
/*  3383*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_AND_I8), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_and:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_and_8>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_AND_I8:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3391*/  /*Scope*/ 13, /*->3405*/
/*  3392*/   OPC_CheckPredicate, 6, // Predicate_atomic_load_and_16
/*  3394*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3396*/   OPC_EmitMergeInputChains1_0,
/*  3397*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_AND_I16), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_and:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_and_16>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_AND_I16:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3405*/  /*Scope*/ 13, /*->3419*/
/*  3406*/   OPC_CheckPredicate, 17, // Predicate_atomic_load_and_32
/*  3408*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3410*/   OPC_EmitMergeInputChains1_0,
/*  3411*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_AND_I32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_and:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_and_32>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_AND_I32:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3419*/  0, /*End of Scope*/
/*  3420*/ /*SwitchOpcode*/ 48, TARGET_VAL(ISD::ATOMIC_LOAD_OR),// ->3471
/*  3423*/  OPC_RecordMemRef,
/*  3424*/  OPC_RecordNode, // #0 = 'atomic_load_or' chained node
/*  3425*/  OPC_RecordChild1, // #1 = $ptr
/*  3426*/  OPC_RecordChild2, // #2 = $incr
/*  3427*/  OPC_Scope, 13, /*->3442*/ // 3 children in Scope
/*  3429*/   OPC_CheckPredicate, 4, // Predicate_atomic_load_or_8
/*  3431*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3433*/   OPC_EmitMergeInputChains1_0,
/*  3434*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_OR_I8), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_or:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_or_8>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_OR_I8:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3442*/  /*Scope*/ 13, /*->3456*/
/*  3443*/   OPC_CheckPredicate, 6, // Predicate_atomic_load_or_16
/*  3445*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3447*/   OPC_EmitMergeInputChains1_0,
/*  3448*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_OR_I16), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_or:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_or_16>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_OR_I16:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3456*/  /*Scope*/ 13, /*->3470*/
/*  3457*/   OPC_CheckPredicate, 17, // Predicate_atomic_load_or_32
/*  3459*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3461*/   OPC_EmitMergeInputChains1_0,
/*  3462*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_OR_I32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_or:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_or_32>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_OR_I32:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3470*/  0, /*End of Scope*/
/*  3471*/ /*SwitchOpcode*/ 48, TARGET_VAL(ISD::ATOMIC_LOAD_XOR),// ->3522
/*  3474*/  OPC_RecordMemRef,
/*  3475*/  OPC_RecordNode, // #0 = 'atomic_load_xor' chained node
/*  3476*/  OPC_RecordChild1, // #1 = $ptr
/*  3477*/  OPC_RecordChild2, // #2 = $incr
/*  3478*/  OPC_Scope, 13, /*->3493*/ // 3 children in Scope
/*  3480*/   OPC_CheckPredicate, 4, // Predicate_atomic_load_xor_8
/*  3482*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3484*/   OPC_EmitMergeInputChains1_0,
/*  3485*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_XOR_I8), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_xor:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_xor_8>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_XOR_I8:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3493*/  /*Scope*/ 13, /*->3507*/
/*  3494*/   OPC_CheckPredicate, 6, // Predicate_atomic_load_xor_16
/*  3496*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3498*/   OPC_EmitMergeInputChains1_0,
/*  3499*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_XOR_I16), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_xor:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_xor_16>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_XOR_I16:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3507*/  /*Scope*/ 13, /*->3521*/
/*  3508*/   OPC_CheckPredicate, 17, // Predicate_atomic_load_xor_32
/*  3510*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3512*/   OPC_EmitMergeInputChains1_0,
/*  3513*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_XOR_I32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_xor:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_xor_32>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_XOR_I32:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3521*/  0, /*End of Scope*/
/*  3522*/ /*SwitchOpcode*/ 48, TARGET_VAL(ISD::ATOMIC_LOAD_NAND),// ->3573
/*  3525*/  OPC_RecordMemRef,
/*  3526*/  OPC_RecordNode, // #0 = 'atomic_load_nand' chained node
/*  3527*/  OPC_RecordChild1, // #1 = $ptr
/*  3528*/  OPC_RecordChild2, // #2 = $incr
/*  3529*/  OPC_Scope, 13, /*->3544*/ // 3 children in Scope
/*  3531*/   OPC_CheckPredicate, 4, // Predicate_atomic_load_nand_8
/*  3533*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3535*/   OPC_EmitMergeInputChains1_0,
/*  3536*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_NAND_I8), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_nand:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_nand_8>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_NAND_I8:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3544*/  /*Scope*/ 13, /*->3558*/
/*  3545*/   OPC_CheckPredicate, 6, // Predicate_atomic_load_nand_16
/*  3547*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3549*/   OPC_EmitMergeInputChains1_0,
/*  3550*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_NAND_I16), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_nand:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_nand_16>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_NAND_I16:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3558*/  /*Scope*/ 13, /*->3572*/
/*  3559*/   OPC_CheckPredicate, 17, // Predicate_atomic_load_nand_32
/*  3561*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3563*/   OPC_EmitMergeInputChains1_0,
/*  3564*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_LOAD_NAND_I32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_load_nand:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_load_nand_32>> - Complexity = 4
             // Dst: (ATOMIC_LOAD_NAND_I32:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3572*/  0, /*End of Scope*/
/*  3573*/ /*SwitchOpcode*/ 48, TARGET_VAL(ISD::ATOMIC_SWAP),// ->3624
/*  3576*/  OPC_RecordMemRef,
/*  3577*/  OPC_RecordNode, // #0 = 'atomic_swap' chained node
/*  3578*/  OPC_RecordChild1, // #1 = $ptr
/*  3579*/  OPC_RecordChild2, // #2 = $incr
/*  3580*/  OPC_Scope, 13, /*->3595*/ // 3 children in Scope
/*  3582*/   OPC_CheckPredicate, 4, // Predicate_atomic_swap_8
/*  3584*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3586*/   OPC_EmitMergeInputChains1_0,
/*  3587*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_SWAP_I8), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_swap:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_swap_8>> - Complexity = 4
             // Dst: (ATOMIC_SWAP_I8:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3595*/  /*Scope*/ 13, /*->3609*/
/*  3596*/   OPC_CheckPredicate, 6, // Predicate_atomic_swap_16
/*  3598*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3600*/   OPC_EmitMergeInputChains1_0,
/*  3601*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_SWAP_I16), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_swap:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_swap_16>> - Complexity = 4
             // Dst: (ATOMIC_SWAP_I16:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3609*/  /*Scope*/ 13, /*->3623*/
/*  3610*/   OPC_CheckPredicate, 17, // Predicate_atomic_swap_32
/*  3612*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3614*/   OPC_EmitMergeInputChains1_0,
/*  3615*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_SWAP_I32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 2/*#Ops*/, 1, 2, 
             // Src: (atomic_swap:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)<<P:Predicate_atomic_swap_32>> - Complexity = 4
             // Dst: (ATOMIC_SWAP_I32:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$incr)
/*  3623*/  0, /*End of Scope*/
/*  3624*/ /*SwitchOpcode*/ 52, TARGET_VAL(ISD::ATOMIC_CMP_SWAP),// ->3679
/*  3627*/  OPC_RecordMemRef,
/*  3628*/  OPC_RecordNode, // #0 = 'atomic_cmp_swap' chained node
/*  3629*/  OPC_RecordChild1, // #1 = $ptr
/*  3630*/  OPC_RecordChild2, // #2 = $cmp
/*  3631*/  OPC_RecordChild3, // #3 = $swap
/*  3632*/  OPC_Scope, 14, /*->3648*/ // 3 children in Scope
/*  3634*/   OPC_CheckPredicate, 4, // Predicate_atomic_cmp_swap_8
/*  3636*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3638*/   OPC_EmitMergeInputChains1_0,
/*  3639*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_CMP_SWAP_I8), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 1, 2, 3, 
             // Src: (atomic_cmp_swap:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$cmp, CPURegs:{ *:[i32] }:$swap)<<P:Predicate_atomic_cmp_swap_8>> - Complexity = 4
             // Dst: (ATOMIC_CMP_SWAP_I8:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$cmp, CPURegs:{ *:[i32] }:$swap)
/*  3648*/  /*Scope*/ 14, /*->3663*/
/*  3649*/   OPC_CheckPredicate, 6, // Predicate_atomic_cmp_swap_16
/*  3651*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3653*/   OPC_EmitMergeInputChains1_0,
/*  3654*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_CMP_SWAP_I16), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 1, 2, 3, 
             // Src: (atomic_cmp_swap:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$cmp, CPURegs:{ *:[i32] }:$swap)<<P:Predicate_atomic_cmp_swap_16>> - Complexity = 4
             // Dst: (ATOMIC_CMP_SWAP_I16:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$cmp, CPURegs:{ *:[i32] }:$swap)
/*  3663*/  /*Scope*/ 14, /*->3678*/
/*  3664*/   OPC_CheckPredicate, 17, // Predicate_atomic_cmp_swap_32
/*  3666*/   OPC_CheckPatternPredicate, 8, // (Subtarget->hasChapter12_1())
/*  3668*/   OPC_EmitMergeInputChains1_0,
/*  3669*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::ATOMIC_CMP_SWAP_I32), 0|OPFL_Chain|OPFL_MemRefs,
                 MVT::i32, 3/*#Ops*/, 1, 2, 3, 
             // Src: (atomic_cmp_swap:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$cmp, CPURegs:{ *:[i32] }:$swap)<<P:Predicate_atomic_cmp_swap_32>> - Complexity = 4
             // Dst: (ATOMIC_CMP_SWAP_I32:{ *:[i32] } iPTR:{ *:[iPTR] }:$ptr, CPURegs:{ *:[i32] }:$cmp, CPURegs:{ *:[i32] }:$swap)
/*  3678*/  0, /*End of Scope*/
/*  3679*/ /*SwitchOpcode*/ 26, TARGET_VAL(ISD::SUB),// ->3708
/*  3682*/  OPC_RecordChild0, // #0 = $rb
/*  3683*/  OPC_RecordChild1, // #1 = $rc
/*  3684*/  OPC_Scope, 10, /*->3696*/ // 2 children in Scope
/*  3686*/   OPC_CheckPatternPredicate, 9, // (Subtarget->disableOverflow())
/*  3688*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SUBu), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (sub:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
             // Dst: (SUBu:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  3696*/  /*Scope*/ 10, /*->3707*/
/*  3697*/   OPC_CheckPatternPredicate, 10, // (Subtarget->enableOverflow())
/*  3699*/   OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SUB), 0,
                 MVT::i32, 2/*#Ops*/, 0, 1, 
             // Src: (sub:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
             // Dst: (SUB:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  3707*/  0, /*End of Scope*/
/*  3708*/ /*SwitchOpcode*/ 12, TARGET_VAL(ISD::MUL),// ->3723
/*  3711*/  OPC_RecordChild0, // #0 = $rb
/*  3712*/  OPC_RecordChild1, // #1 = $rc
/*  3713*/  OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  3715*/  OPC_MorphNodeTo1, TARGET_VAL(Cpu0::MUL), 0,
                MVT::i32, 2/*#Ops*/, 0, 1, 
            // Src: (mul:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc) - Complexity = 3
            // Dst: (MUL:{ *:[i32] } CPURegs:{ *:[i32] }:$rb, CPURegs:{ *:[i32] }:$rc)
/*  3723*/ /*SwitchOpcode*/ 16, TARGET_VAL(ISD::BR),// ->3742
/*  3726*/  OPC_RecordNode, // #0 = 'br' chained node
/*  3727*/  OPC_RecordChild1, // #1 = $addr
/*  3728*/  OPC_MoveChild1,
/*  3729*/  OPC_CheckOpcode, TARGET_VAL(ISD::BasicBlock),
/*  3732*/  OPC_MoveParent,
/*  3733*/  OPC_CheckPatternPredicate, 7, // (Subtarget->hasChapter8_1())
/*  3735*/  OPC_EmitMergeInputChains1_0,
/*  3736*/  OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JMP), 0|OPFL_Chain,
                1/*#Ops*/, 1, 
            // Src: (br (bb:{ *:[Other] }):$addr) - Complexity = 3
            // Dst: (JMP (bb:{ *:[Other] }):$addr)
/*  3742*/ /*SwitchOpcode*/ 21, TARGET_VAL(ISD::BRIND),// ->3766
/*  3745*/  OPC_RecordNode, // #0 = 'brind' chained node
/*  3746*/  OPC_RecordChild1, // #1 = $ra
/*  3747*/  OPC_CheckChild1Type, MVT::i32,
/*  3749*/  OPC_EmitMergeInputChains1_0,
/*  3750*/  OPC_Scope, 6, /*->3758*/ // 2 children in Scope
/*  3752*/   OPC_MorphNodeTo0, TARGET_VAL(Cpu0::JR), 0|OPFL_Chain,
                 1/*#Ops*/, 1, 
             // Src: (brind GPROut:{ *:[i32] }:$ra) - Complexity = 3
             // Dst: (JR GPROut:{ *:[i32] }:$ra)
/*  3758*/  /*Scope*/ 6, /*->3765*/
/*  3759*/   OPC_MorphNodeTo0, TARGET_VAL(Cpu0::RET), 0|OPFL_Chain,
                 1/*#Ops*/, 1, 
             // Src: (brind GPROut:{ *:[i32] }:$ra) - Complexity = 3
             // Dst: (RET GPROut:{ *:[i32] }:$ra)
/*  3765*/  0, /*End of Scope*/
/*  3766*/ /*SwitchOpcode*/ 10, TARGET_VAL(Cpu0ISD::Ret),// ->3779
/*  3769*/  OPC_RecordNode, // #0 = 'Cpu0Ret' chained node
/*  3770*/  OPC_CaptureGlueInput,
/*  3771*/  OPC_CheckPatternPredicate, 14, // (Subtarget->hasChapter3_4())
/*  3773*/  OPC_EmitMergeInputChains1_0,
/*  3774*/  OPC_MorphNodeTo0, TARGET_VAL(Cpu0::RetLR), 0|OPFL_Chain|OPFL_GlueInput|OPFL_Variadic0,
                0/*#Ops*/, 
            // Src: (Cpu0Ret) - Complexity = 3
            // Dst: (RetLR)
/*  3779*/ /*SwitchOpcode*/ 12, TARGET_VAL(Cpu0ISD::DivRem),// ->3794
/*  3782*/  OPC_RecordChild0, // #0 = $ra
/*  3783*/  OPC_RecordChild1, // #1 = $rb
/*  3784*/  OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  3786*/  OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SDIV), 0|OPFL_GlueOutput,
                MVT::i32, 2/*#Ops*/, 0, 1, 
            // Src: (Cpu0DivRem CPURegs:{ *:[i32] }:$ra, CPURegs:{ *:[i32] }:$rb) - Complexity = 3
            // Dst: (SDIV:{ *:[i32] } CPURegs:{ *:[i32] }:$ra, CPURegs:{ *:[i32] }:$rb)
/*  3794*/ /*SwitchOpcode*/ 12, TARGET_VAL(Cpu0ISD::DivRemU),// ->3809
/*  3797*/  OPC_RecordChild0, // #0 = $ra
/*  3798*/  OPC_RecordChild1, // #1 = $rb
/*  3799*/  OPC_CheckPatternPredicate, 12, // (Subtarget->hasChapter4_1())
/*  3801*/  OPC_MorphNodeTo1, TARGET_VAL(Cpu0::UDIV), 0|OPFL_GlueOutput,
                MVT::i32, 2/*#Ops*/, 0, 1, 
            // Src: (Cpu0DivRemU CPURegs:{ *:[i32] }:$ra, CPURegs:{ *:[i32] }:$rb) - Complexity = 3
            // Dst: (UDIV:{ *:[i32] } CPURegs:{ *:[i32] }:$ra, CPURegs:{ *:[i32] }:$rb)
/*  3809*/ /*SwitchOpcode*/ 14, TARGET_VAL(Cpu0ISD::EH_RETURN),// ->3826
/*  3812*/  OPC_RecordNode, // #0 = 'CPU0ehret' chained node
/*  3813*/  OPC_CaptureGlueInput,
/*  3814*/  OPC_RecordChild1, // #1 = $spoff
/*  3815*/  OPC_RecordChild2, // #2 = $dst
/*  3816*/  OPC_CheckChild2Type, MVT::i32,
/*  3818*/  OPC_EmitMergeInputChains1_0,
/*  3819*/  OPC_MorphNodeTo0, TARGET_VAL(Cpu0::CPU0eh_return32), 0|OPFL_Chain|OPFL_GlueInput|OPFL_Variadic2,
                2/*#Ops*/, 1, 2, 
            // Src: (CPU0ehret GPROut:{ *:[i32] }:$spoff, GPROut:{ *:[i32] }:$dst) - Complexity = 3
            // Dst: (CPU0eh_return32 GPROut:{ *:[i32] }:$spoff, GPROut:{ *:[i32] }:$dst)
/*  3826*/ /*SwitchOpcode*/ 12, TARGET_VAL(ISD::SUBC),// ->3841
/*  3829*/  OPC_RecordChild0, // #0 = $lhs
/*  3830*/  OPC_RecordChild1, // #1 = $rhs
/*  3831*/  OPC_CheckPatternPredicate, 0, // (Subtarget->hasChapter7_1())
/*  3833*/  OPC_MorphNodeTo1, TARGET_VAL(Cpu0::SUBu), 0|OPFL_GlueOutput,
                MVT::i32, 2/*#Ops*/, 0, 1, 
            // Src: (subc:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs) - Complexity = 3
            // Dst: (SUBu:{ *:[i32] } CPURegs:{ *:[i32] }:$lhs, CPURegs:{ *:[i32] }:$rhs)
/*  3841*/ 0, // EndSwitchOpcode
    0
  }; // Total Array size is 3843 bytes

  // Opcode Histogram:
  // #OPC_Scope                                = 44
  // #OPC_RecordNode                           = 22
  // #OPC_RecordChild                          = 142
  // #OPC_RecordMemRef                         = 10
  // #OPC_CaptureGlueInput                     = 5
  // #OPC_MoveChild                            = 62
  // #OPC_MoveParent                           = 123
  // #OPC_CheckSame                            = 0
  // #OPC_CheckChildSame                       = 0
  // #OPC_CheckPatternPredicate                = 165
  // #OPC_CheckPredicate                       = 64
  // #OPC_CheckOpcode                          = 53
  // #OPC_SwitchOpcode                         = 10
  // #OPC_CheckType                            = 36
  // #OPC_SwitchType                           = 0
  // #OPC_CheckChildType                       = 8
  // #OPC_CheckInteger                         = 0
  // #OPC_CheckChildInteger                    = 6
  // #OPC_CheckCondCode                        = 0
  // #OPC_CheckChild2CondCode                  = 62
  // #OPC_CheckValueType                       = 0
  // #OPC_CheckComplexPat                      = 12
  // #OPC_CheckAndImm                          = 0
  // #OPC_CheckOrImm                           = 0
  // #OPC_CheckImmAllOnesV                     = 0
  // #OPC_CheckImmAllZerosV                    = 0
  // #OPC_CheckFoldableChainNode               = 0
  // #OPC_EmitInteger                          = 24
  // #OPC_EmitStringInteger                    = 0
  // #OPC_EmitRegister                         = 20
  // #OPC_EmitConvertToTarget                  = 23
  // #OPC_EmitMergeInputChains                 = 79
  // #OPC_EmitCopyToReg                        = 0
  // #OPC_EmitNode                             = 55
  // #OPC_EmitNodeXForm                        = 6
  // #OPC_CompleteMatch                        = 0
  // #OPC_MorphNodeTo                          = 173

  #undef TARGET_VAL
  SelectCodeCommon(N, MatcherTable,sizeof(MatcherTable));
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
bool CheckPatternPredicate(unsigned PredNo) const override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
bool DAGISEL_CLASS_COLONCOLON CheckPatternPredicate(unsigned PredNo) const
#if DAGISEL_INLINE
  override
#endif
{
  switch (PredNo) {
  default: llvm_unreachable("Invalid predicate in table?");
  case 0: return (Subtarget->hasChapter7_1());
  case 1: return (Subtarget->hasChapter4_2());
  case 2: return (Subtarget->hasSlt());
  case 3: return (Subtarget->hasCmp());
  case 4: return (Subtarget->hasChapter8_2());
  case 5: return (Subtarget->hasChapter9_2());
  case 6: return (Subtarget->hasChapter6_1());
  case 7: return (Subtarget->hasChapter8_1());
  case 8: return (Subtarget->hasChapter12_1());
  case 9: return (Subtarget->disableOverflow());
  case 10: return (Subtarget->enableOverflow());
  case 11: return (Subtarget->hasChapter3_5());
  case 12: return (Subtarget->hasChapter4_1());
  case 13: return (Subtarget->hasChapter9_1());
  case 14: return (Subtarget->hasChapter3_4());
  }
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
bool CheckNodePredicate(SDNode *Node, unsigned PredNo) const override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
bool DAGISEL_CLASS_COLONCOLON CheckNodePredicate(SDNode *Node, unsigned PredNo) const
#if DAGISEL_INLINE
  override
#endif
{
  switch (PredNo) {
  default: llvm_unreachable("Invalid predicate in table?");
  case 0: {
    // Predicate_unindexedload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getAddressingMode() != ISD::UNINDEXED) return false;
return true;

  }
  case 1: {
    // Predicate_load
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::NON_EXTLOAD) return false;
return true;

  }
  case 2: {
    // Predicate_load_a
    // Predicate_sextloadi16_a
    // Predicate_zextloadi16_a
    // Predicate_extloadi16_a
    SDNode *N = Node;
    (void)N;

  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getMemoryVT().getSizeInBits()/8 <= LD->getAlignment();

  }
  case 3: {
    // Predicate_sextload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::SEXTLOAD) return false;
return true;

  }
  case 4: {
    // Predicate_sextloadi8
    // Predicate_zextloadi8
    // Predicate_extloadi8
    // Predicate_truncstorei8
    // Predicate_atomic_load_add_8
    // Predicate_atomic_load_sub_8
    // Predicate_atomic_load_and_8
    // Predicate_atomic_load_or_8
    // Predicate_atomic_load_xor_8
    // Predicate_atomic_load_nand_8
    // Predicate_atomic_swap_8
    // Predicate_atomic_cmp_swap_8
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i8) return false;
return true;

  }
  case 5: {
    // Predicate_zextload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::ZEXTLOAD) return false;
return true;

  }
  case 6: {
    // Predicate_sextloadi16
    // Predicate_zextloadi16
    // Predicate_extloadi16
    // Predicate_truncstorei16
    // Predicate_atomic_load_add_16
    // Predicate_atomic_load_sub_16
    // Predicate_atomic_load_and_16
    // Predicate_atomic_load_or_16
    // Predicate_atomic_load_xor_16
    // Predicate_atomic_load_nand_16
    // Predicate_atomic_swap_16
    // Predicate_atomic_cmp_swap_16
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i16) return false;
return true;

  }
  case 7: {
    // Predicate_extload
    SDNode *N = Node;
    (void)N;
if (cast<LoadSDNode>(N)->getExtensionType() != ISD::EXTLOAD) return false;
return true;

  }
  case 8: {
    // Predicate_extloadi1
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i1) return false;
return true;

  }
  case 9: {
    // Predicate_unindexedstore
    SDNode *N = Node;
    (void)N;
if (cast<StoreSDNode>(N)->getAddressingMode() != ISD::UNINDEXED) return false;
return true;

  }
  case 10: {
    // Predicate_store
    SDNode *N = Node;
    (void)N;
 if (cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 11: {
    // Predicate_store_a
    // Predicate_truncstorei16_a
    SDNode *N = Node;
    (void)N;

  StoreSDNode *SD = cast<StoreSDNode>(N);
  return SD->getMemoryVT().getSizeInBits()/8 <= SD->getAlignment();

  }
  case 12: {
    // Predicate_truncstore
    SDNode *N = Node;
    (void)N;
 if (!cast<StoreSDNode>(N)->isTruncatingStore()) return false;
return true;

  }
  case 13: {
    // Predicate_immZExt16
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;

  if (N->getValueType(0) == MVT::i32)
    return (uint32_t)N->getZExtValue() == (unsigned short)N->getZExtValue();
  else
    return (uint64_t)N->getZExtValue() == (unsigned short)N->getZExtValue();

  }
  case 14: {
    // Predicate_immSExt16
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;
 return isInt<16>(N->getSExtValue()); 
  }
  case 15: {
    // Predicate_immZExt5
    int64_t Imm = cast<ConstantSDNode>(Node)->getSExtValue();
return Imm == (Imm & 0x1f);
  }
  case 16: {
    // Predicate_immLow16Zero
    auto *N = cast<ConstantSDNode>(Node);
    (void)N;

  int64_t Val = N->getSExtValue();
  return isInt<32>(Val) && !(Val & 0xffff);

  }
  case 17: {
    // Predicate_atomic_load_add_32
    // Predicate_atomic_load_sub_32
    // Predicate_atomic_load_and_32
    // Predicate_atomic_load_or_32
    // Predicate_atomic_load_xor_32
    // Predicate_atomic_load_nand_32
    // Predicate_atomic_swap_32
    // Predicate_atomic_cmp_swap_32
    SDNode *N = Node;
    (void)N;
if (cast<MemSDNode>(N)->getMemoryVT() != MVT::i32) return false;
return true;

  }
  }
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
bool CheckComplexPattern(SDNode *Root, SDNode *Parent,
      SDValue N, unsigned PatternNo,
      SmallVectorImpl<std::pair<SDValue, SDNode *>> &Result) override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
bool DAGISEL_CLASS_COLONCOLON CheckComplexPattern(SDNode *Root, SDNode *Parent,
      SDValue N, unsigned PatternNo,
      SmallVectorImpl<std::pair<SDValue, SDNode *>> &Result)
#if DAGISEL_INLINE
  override
#endif
{
  unsigned NextRes = Result.size();
  switch (PatternNo) {
  default: llvm_unreachable("Invalid pattern # in table?");
  case 0:
    Result.resize(NextRes+2);
  return SelectAddr(Parent, N, Result[NextRes+0].first, Result[NextRes+1].first);
  }
}
#endif // GET_DAGISEL_BODY

#ifdef GET_DAGISEL_DECL
SDValue RunSDNodeXForm(SDValue V, unsigned XFormNo) override;
#endif
#if defined(GET_DAGISEL_BODY) || DAGISEL_INLINE
SDValue DAGISEL_CLASS_COLONCOLON RunSDNodeXForm(SDValue V, unsigned XFormNo)
#if DAGISEL_INLINE
  override
#endif
{
  switch (XFormNo) {
  default: llvm_unreachable("Invalid xform # in table?");
  case 0: {  // LO16
    ConstantSDNode *N = cast<ConstantSDNode>(V.getNode());

  return getImm(N, N->getZExtValue() & 0xffff);

  }
  case 1: {  // HI16
    ConstantSDNode *N = cast<ConstantSDNode>(V.getNode());

  return getImm(N, (N->getZExtValue() >> 16) & 0xffff);

  }
  }
}
#endif // GET_DAGISEL_BODY


#ifdef DAGISEL_INLINE
#undef DAGISEL_INLINE
#endif
#ifdef DAGISEL_CLASS_COLONCOLON
#undef DAGISEL_CLASS_COLONCOLON
#endif
#ifdef GET_DAGISEL_DECL
#undef GET_DAGISEL_DECL
#endif
#ifdef GET_DAGISEL_BODY
#undef GET_DAGISEL_BODY
#endif
