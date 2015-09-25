//===-- Cpu0RegisterInfo.td - Cpu0 Register defs -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the CPU0 register file
//===----------------------------------------------------------------------===//


// We have banks of 16 registers each.
class Cpu0Reg<string n> : Register<n> {
  field bits<4> Num;
  let Namespace = "Cpu0";
}

// Cpu0 CPU Registers
class Cpu0GPRReg<bits<4> num, string n> : Cpu0Reg<n> {
  let Num = num;
}

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//
// The register string, such as "9" or "gp" will show on "llvm-objdump -d"
let Namespace = "Cpu0" in {
  // General Purpose Registers
  def ZERO : Cpu0GPRReg< 0, "zero">, DwarfRegNum<[0]>;
  def AT   : Cpu0GPRReg< 1, "1">,   DwarfRegNum<[1]>;
  def V0   : Cpu0GPRReg< 2, "2">,    DwarfRegNum<[2]>;
  def V1   : Cpu0GPRReg< 3, "3">,    DwarfRegNum<[3]>;
  def A0   : Cpu0GPRReg< 4, "4">,    DwarfRegNum<[6]>;
  def A1   : Cpu0GPRReg< 5, "5">,    DwarfRegNum<[7]>;
  def T9   : Cpu0GPRReg< 6, "t9">,   DwarfRegNum<[6]>;
  def S0   : Cpu0GPRReg< 7, "7">,    DwarfRegNum<[7]>;
  def S1   : Cpu0GPRReg< 8, "8">,    DwarfRegNum<[8]>;
  def S2   : Cpu0GPRReg< 9, "9">,    DwarfRegNum<[9]>;
  def GP   : Cpu0GPRReg< 10, "gp">,  DwarfRegNum<[10]>;
  def FP   : Cpu0GPRReg< 11, "fp">,  DwarfRegNum<[11]>;
  def SW   : Cpu0GPRReg< 12, "sw">,   DwarfRegNum<[12]>;
  def SP   : Cpu0GPRReg< 13, "sp">,   DwarfRegNum<[13]>;
  def LR   : Cpu0GPRReg< 14, "lr">,   DwarfRegNum<[14]>;
  def PC   : Cpu0GPRReg< 15, "pc">,   DwarfRegNum<[15]>;
//  def MAR  : Register< 16, "mar">,  DwarfRegNum<[16]>;
//  def MDR  : Register< 17, "mdr">,  DwarfRegNum<[17]>;

  // Hi/Lo registers
  def HI   : Register<"hi">, DwarfRegNum<[18]>;
  def LO   : Register<"lo">, DwarfRegNum<[19]>;
}

//===----------------------------------------------------------------------===//
// Register Classes
//===----------------------------------------------------------------------===//

def CPURegs : RegisterClass<"Cpu0", [i32], 32, (add
  // Reserved
  ZERO, AT, 
  // Return Values and Arguments
  V0, V1, A0, A1, 
  // Not preserved across procedure calls
  T9, 
  // Callee save
  S0, S1, S2, 
  // Reserved
  GP, FP, 
  // Not preserved across procedure calls
  SW, 
  // Reserved
  SP, LR, PC)>;

// Hi/Lo Registers
def HILO : RegisterClass<"Cpu0", [i32], 32, (add HI, LO)>;

// Status Registers
def SR   : RegisterClass<"Cpu0", [i32], 32, (add SW)>;
