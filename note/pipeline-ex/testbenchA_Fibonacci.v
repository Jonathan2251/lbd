//Appendix A: Verilog Code & Test-bench.
`define TEST_A

//Pipeline Registers Modules:
//IF/ID:
module IFID(flush,clock,IFIDWrite,PC_Plus4,Inst,InstReg,PC_Plus4Reg);
  input [31:0] PC_Plus4,Inst;
  input clock,IFIDWrite,flush;
  output [31:0] InstReg, PC_Plus4Reg;
  reg [31:0] InstReg, PC_Plus4Reg;
  initial begin
    InstReg = 0;
    PC_Plus4Reg = 0;
  end
  always@(posedge clock)
  begin
    if(flush)
    begin
      InstReg <= 0;
      PC_Plus4Reg <=0;
    end
    else if(IFIDWrite)
    begin
      InstReg <= Inst;
      PC_Plus4Reg <= PC_Plus4;
    end
  end
endmodule

//ID/EX:
module IDEX(clock,WB,M,EX,DataA,DataB,imm_value,RegRs,RegRt,RegRd,WBreg,Mreg,EXreg,DataAreg,
DataBreg,imm_valuereg,RegRsreg,RegRtreg,RegRdreg);
  input clock;
  input [1:0] WB;
  input [2:0] M;
  input [3:0] EX;
  input [4:0] RegRs,RegRt,RegRd;
  input [31:0] DataA,DataB,imm_value;
  output [1:0] WBreg;
  output [2:0] Mreg;
  output [3:0] EXreg;
  output [4:0] RegRsreg,RegRtreg,RegRdreg;
  output [31:0] DataAreg,DataBreg,imm_valuereg;

  reg [1:0] WBreg;
  reg [2:0] Mreg;
  reg [3:0] EXreg;
  reg [31:0] DataAreg,DataBreg,imm_valuereg;
  reg [4:0] RegRsreg,RegRtreg,RegRdreg;

  initial begin
    WBreg = 0;
    Mreg = 0;
    EXreg = 0;
    DataAreg = 0;
    DataBreg = 0;
    imm_valuereg = 0;
    RegRsreg = 0;
    RegRtreg = 0;
    RegRdreg = 0;
  end
  always@(posedge clock)
  begin
    WBreg <= WB;
    Mreg <= M;
    EXreg <= EX;
    DataAreg <= DataA;
    DataBreg <= DataB;
    imm_valuereg <= imm_value;
    RegRsreg <= RegRs;
    RegRtreg <= RegRt;
    RegRdreg <= RegRd;
  end
endmodule

//EX/MEM:
module EXMEM(clock,WB,M,ALUOut,RegRD,WriteDataIn,Mreg,WBreg,ALUreg,RegRDreg,WriteDataOut);
  input clock;
  input [1:0] WB;
  input [2:0] M;
  input [4:0] RegRD;
  input [31:0] ALUOut,WriteDataIn;
  output [1:0] WBreg;
  output [2:0] Mreg;
  output [31:0] ALUreg,WriteDataOut;
  output [4:0] RegRDreg;
  reg [1:0] WBreg;
  reg [2:0] Mreg;
  reg [31:0] ALUreg,WriteDataOut;
  reg [4:0] RegRDreg;

  initial begin
    WBreg=0;
    Mreg=0;
    ALUreg=0;
    WriteDataOut=0;
    RegRDreg=0;
  end
  always@(posedge clock)
  begin
    WBreg <= WB;
    Mreg <= M;
    ALUreg <= ALUOut;
    RegRDreg <= RegRD;
    WriteDataOut <= WriteDataIn;
  end
endmodule

//1MEM/WB:
module MEMWB(clock,WB,Memout,ALUOut,RegRD,WBreg,Memreg,ALUreg,RegRDreg);
  input clock;
  input [1:0] WB;
  input [4:0] RegRD;
  input [31:0] Memout,ALUOut;
  output [1:0] WBreg;
  output [31:0] Memreg,ALUreg;
  output [4:0] RegRDreg;
  reg [1:0] WBreg;
  reg [31:0] Memreg,ALUreg;
  reg [4:0] RegRDreg;
  initial begin
    WBreg = 0;
    Memreg = 0;
    ALUreg = 0;
    RegRDreg = 0;
  end
  always@(posedge clock)
  begin
    WBreg <= WB;
    Memreg <= Memout;
    ALUreg <= ALUOut;
    RegRDreg <= RegRD;
  end
endmodule

//Instruction Memory Module:
module InstructMem(PC,Inst);
  input [31:0] PC;
  output [31:0] Inst;
  reg [31:0] regfile[511:0];//32 32-bit register
  assign Inst = regfile[PC]; //assigns output to instruction
endmodule

//Register File Module:
module Registers(clock,WE,InData,WrReg,ReadA,ReadB,OutA,OutB);
  input [4:0] WrReg, ReadA, ReadB;
  input WE,clock;
  input [31:0] InData;
  output [31:0] OutA,OutB;
  reg [31:0] OutA, OutB;//2 32-bit output reg
  reg [31:0] regfile[31:0];//32 32-bit registers
  initial begin
    OutA = -20572; //random values for initial
    OutB = -398567;
  end
  always@(clock,InData,WrReg,WE)
  begin
    if(WE && clock)
    begin
      regfile[WrReg]<=InData;//write to register
      $display("Does WrReg: %d Data: %d",WrReg,InData);
    end
  end
  always @ (clock,ReadA,ReadB,WrReg)
  begin
    if(~clock)
    begin
      OutA <= regfile[ReadA];//read values from registers
      OutB <= regfile[ReadB];
      $monitor ("R3: %d R4: %d R5 %d R6: %d R7: %d R8 %d R9: %d R10: %d R11 %d R12: %d R13: %d R14%d", regfile[3],regfile[4],regfile[5],regfile[6],regfile[7],regfile[8],regfile[9],regfile[10], regfile[11],regfile[12],regfile[13],regfile[14]);
    end
  end
endmodule

//ALU Module:
module ALU(ALUCon,DataA,DataB,Result);
  input [3:0] ALUCon;
  input [31:0] DataA,DataB;
  output [31:0] Result;
  reg [31:0] Result;
  reg Zero;
  initial begin
    Result = 32'd0;
  end
  always@(ALUCon,DataA,DataB)
  begin
  case(ALUCon)
    4'b0000://and
      Result <= DataA&DataB;
    4'b0001://or
      Result <= DataA|DataB;
    4'b0010://add
      Result <= DataA+DataB;
    4'b0011://multiply
      Result <= DataA*DataB;
    4'b0100://nor
    begin
      Result[0] <= !(DataA[0]|DataB[0]);
      Result[1] <= !(DataA[1]|DataB[1]);
      Result[2] <= !(DataA[2]|DataB[2]);
      Result[3] <= !(DataA[3]|DataB[3]);
      Result[4] <= !(DataA[4]|DataB[4]);
      Result[5] <= !(DataA[5]|DataB[5]);
      Result[6] <= !(DataA[6]|DataB[6]);
      Result[7] <= !(DataA[7]|DataB[7]);
      Result[8] <= !(DataA[8]|DataB[8]);
      Result[9] <= !(DataA[9]|DataB[9]);
      Result[10] <= !(DataA[10]|DataB[10]);
      Result[11] <= !(DataA[11]|DataB[11]);
      Result[12] <= !(DataA[12]|DataB[12]);
      Result[13] <= !(DataA[13]|DataB[13]);
      Result[14] <= !(DataA[14]|DataB[14]);
      Result[15] <= !(DataA[15]|DataB[15]);
      Result[16] <= !(DataA[16]|DataB[16]);
      Result[17] <= !(DataA[17]|DataB[17]);
      Result[18] <= !(DataA[18]|DataB[18]);
      Result[19] <= !(DataA[19]|DataB[19]);
      Result[20] <= !(DataA[20]|DataB[20]);
      Result[21] <= !(DataA[21]|DataB[21]);
      Result[22] <= !(DataA[22]|DataB[22]);
      Result[23] <= !(DataA[23]|DataB[23]);
      Result[24] <= !(DataA[24]|DataB[24]);
      Result[25] <= !(DataA[25]|DataB[25]);
      Result[26] <= !(DataA[26]|DataB[26]);
      Result[27] <= !(DataA[27]|DataB[27]);
      Result[28] <= !(DataA[28]|DataB[28]);
      Result[29] <= !(DataA[29]|DataB[29]);
      Result[30] <= !(DataA[30]|DataB[30]);
      Result[31] <= !(DataA[31]|DataB[31]);
    end
    4'b0101://divide
      Result <= DataA/DataB;
    4'b0110://sub
      Result <= DataA-DataB;
    4'b0111://slt
      Result = DataA<DataB ? 1:0;
    4'b1000://sll
      Result <= (DataA<<DataB);
    4'b0110://srl
      Result <= (DataA>>DataB);
    default: //error
    begin
      $display("ALUERROR");
      Result = 0;
    end
  endcase
  end
endmodule

//Data Memory Module:
module DATAMEM(MemWrite,MemRead,Addr,Wdata,Rdata);
  input [31:0] Addr,Wdata;
  input MemWrite,MemRead;
  output [31:0] Rdata;
  reg [31:0] Rdata;
  reg [31:0] regfile[511:0];//32 32-bit registers
  always@(Addr,Wdata,MemWrite,MemRead)
  if(MemWrite)
  begin
    $display("Writing %d -> Addr: %d",Wdata,Addr);
    regfile[Addr]<=Wdata; //memory write
  end
  always@(Addr,Wdata,MemWrite,MemRead)
  if(MemRead)
    Rdata <= regfile[Addr];//memory read
endmodule

//ALU Control Module:
module ALUControl(andi,ori,addi,ALUOp,funct,ALUCon);
  input [1:0] ALUOp;
  input [5:0] funct;
  input andi,ori,addi;
  output [3:0] ALUCon;
  reg [3:0] ALUCon;
  always@(ALUOp or funct or andi or ori or addi)
  begin
  case(ALUOp)
  2'b00://lw or sw
    ALUCon = 4'b0010;
  2'b01://beq
    ALUCon = 4'b0110;
  2'b10://R-type
  begin
    if(funct==6'b100100)
      ALUCon = 4'b0000;//and
    if(funct==6'b100101)
      ALUCon = 4'b0001;//or
    if(funct==6'b100000)
      ALUCon = 4'b0010;//add
    if(funct==6'b011000)
      ALUCon = 4'b0011;//multi
    if(funct==6'b100111)
      ALUCon = 4'b0100;//nor
    if(funct==6'b011010)
      ALUCon = 4'b0101;//div
    if(funct==6'b100010)
      ALUCon = 4'b0110;//sub
    if(funct==6'b101010)
      ALUCon = 4'b0111;//slt
  end
  2'b11://immediate
  begin
    if(andi)begin
      ALUCon = 4'b0000;//andi
    end
    if(ori) begin
      ALUCon = 4'b0001;//ori
    end
    if(addi)
      ALUCon = 4'b0010;//addi
  end
  endcase
  end
endmodule

//Control Module:
module Control(Op,Out,j,bne,imm,andi,ori,addi);
  input [5:0] Op;
  output[8:0] Out;
  output j,bne,imm,andi,ori,addi;
  wire regdst,alusrc,memtoreg,regwrite,memread,memwrite,branch;
  //determines type of instruction
  wire r = ~Op[5]&~Op[4]&~Op[3]&~Op[2]&~Op[1]&~Op[0];
  wire lw = Op[5]&~Op[4]&~Op[3]&~Op[2]&Op[1]&Op[0];
  wire sw = Op[5]&~Op[4]&Op[3]&~Op[2]&Op[1]&Op[0];
  wire beq = ~Op[5]&~Op[4]&~Op[3]&Op[2]&~Op[1]&~Op[0];
  wire bne = ~Op[5]&~Op[4]&~Op[3]&Op[2]&~Op[1]&Op[0];
  wire j = ~Op[5]&~Op[4]&~Op[3]&~Op[2]&Op[1]&~Op[0];
  wire andi = ~Op[5]&~Op[4]&Op[3]&Op[2]&~Op[1]&~Op[0];
  wire ori = ~Op[5]&~Op[4]&Op[3]&Op[2]&~Op[1]&Op[0];
  wire addi = ~Op[5]&~Op[4]&Op[3]&~Op[2]&~Op[1]&~Op[0];
  wire imm = andi|ori|addi; //immediate value type
  //seperate control arrays for reference
  wire [3:0] EXE;
  wire [2:0] M;
  wire [1:0] WB;
  // microcode control
  assign regdst = r;
  assign alusrc = lw|sw|imm;
  assign memtoreg = lw;
  assign regwrite = r|lw|imm;
  assign memread = lw;
  assign memwrite = sw;
  assign branch = beq;
  // EXE control
  assign EXE[3] = regdst;
  assign EXE[2] = alusrc;
  assign EXE[1] = r;
  assign EXE[0] = beq;
  //M control
  assign M[2] = branch;
  assign M[1] = memread;
  assign M[0] = memwrite;
  //WB control
  assign WB[1] = memtoreg; //not same as diagram
  assign WB[0] = regwrite;
  //output control
  assign Out[8:7] = WB;
  assign Out[6:4] = M;
  assign Out[3:0] = EXE;
endmodule

//Forwarding Unit Module:
module ForwardUnit(MEMRegRd,WBRegRd,EXRegRs,EXRegRt, MEM_RegWrite, WB_RegWrite, ForwardA, ForwardB);
input[4:0] MEMRegRd,WBRegRd,EXRegRs,EXRegRt;
input MEM_RegWrite, WB_RegWrite;
output[1:0] ForwardA, ForwardB;
reg[1:0] ForwardA, ForwardB;
//Forward A
always@(MEM_RegWrite or MEMRegRd or EXRegRs or WB_RegWrite or WBRegRd)
begin
if((MEM_RegWrite)&&(MEMRegRd != 0)&&(MEMRegRd == EXRegRs))
ForwardA = 2'b10;
else if((WB_RegWrite)&&(WBRegRd != 0)&&(WBRegRd == EXRegRs)&&(MEMRegRd != EXRegRs) )
ForwardA = 2'b01;
else
ForwardA = 2'b00;
end
//Forward B
always@(WB_RegWrite or WBRegRd or EXRegRt or MEMRegRd or MEM_RegWrite)
begin
if((WB_RegWrite)&&(WBRegRd != 0)&&(WBRegRd == EXRegRt)&&(MEMRegRd != EXRegRt) )
ForwardB = 2'b01;
else if((MEM_RegWrite)&&(MEMRegRd != 0)&&(MEMRegRd == EXRegRt))
ForwardB = 2'b10;
else
ForwardB = 2'b00;
end
endmodule

//Hazard Detection Unit Module:
module HazardUnit(IDRegRs,IDRegRt,EXRegRt,EXMemRead,PCWrite,IFIDWrite,HazMuxCon);
  input [4:0] IDRegRs,IDRegRt,EXRegRt;
  input EXMemRead;
  output PCWrite, IFIDWrite, HazMuxCon;
  reg PCWrite, IFIDWrite, HazMuxCon;
  always@(IDRegRs,IDRegRt,EXRegRt,EXMemRead)
  if(EXMemRead&((EXRegRt == IDRegRs)|(EXRegRt == IDRegRt)))
  begin//stall
    PCWrite = 0;
    IFIDWrite = 0;
    HazMuxCon = 1;
  end
  else
  begin//no stall
    PCWrite = 1;
    IFIDWrite = 1;
    HazMuxCon = 1;
  end
endmodule

//Multiplexer Module:
module BIGMUX2(A,X0,X1,X2,X3,Out);//non-clocked mux
  input [1:0] A;
  input [31:0] X3,X2,X1,X0;
  output [31:0] Out;
  reg [31:0] Out;
  always@(A,X3,X2,X1,X0)
  begin
  case(A)
  2'b00:
    Out <= X0;
  2'b01:
    Out <= X1;
  2'b10:
    Out <= X2;
  2'b11:
    Out <= X3;
  endcase
  end
endmodule

//Top level CPU Module:
module cpu(clock);
  input clock;
  //debugging vars
  reg [31:0] cycle;
  //IF vars
  wire [31:0] nextpc,IFpc_plus_4,IFinst;
  reg [31:0] pc;
  //ID vars
  wire PCSrc;
  wire [4:0] IDRegRs,IDRegRt,IDRegRd;
  wire [31:0] IDpc_plus_4,IDinst;
  wire [31:0] IDRegAout, IDRegBout;
  wire [31:0] IDimm_value,BranchAddr,PCMuxOut,JumpTarget;

  //control vars in ID stage
  wire PCWrite,IFIDWrite,HazMuxCon,jump,bne,imm,andi,ori,addi;
  wire [8:0] IDcontrol,ConOut;

  //EX vars
  wire [1:0] EXWB,ForwardA,ForwardB,aluop;
  wire [2:0] EXM;
  wire [3:0] EXEX,ALUCon;
  wire [4:0] EXRegRs,EXRegRt,EXRegRd,regtopass;
  wire [31:0] EXRegAout,EXRegBout,EXimm_value, b_value;
  wire [31:0] EXALUOut,ALUSrcA,ALUSrcB;

  //MEM vars
  wire [1:0] MEMWB;
  wire [2:0] MEMM;
  wire [4:0] MEMRegRd;
  wire [31:0] MEMALUOut,MEMWriteData,MEMReadData;

  //WB vars
  wire [1:0] WBWB;
  wire [4:0] WBRegRd;
  wire [31:0] datatowrite,WBReadData,WBALUOut;

  //initial conditions
  initial begin
  pc = 0;
  cycle = 0;
  end

  //debugging variable
  always@(posedge clock)
  begin
  cycle = cycle + 1;
  end

  /**
  * Instruction Fetch (IF)
  */
  assign PCSrc = ((IDRegAout==IDRegBout)&IDcontrol[6])|((IDRegAout!=IDRegBout)&bne);
  assign IFFlush = PCSrc|jump;
  assign IFpc_plus_4 = pc + 4;

  assign nextpc = PCSrc ? BranchAddr : PCMuxOut;

  always @ (posedge clock) begin
    if(PCWrite)
    begin
      pc = nextpc; //update pc
      $display("PC: %d",pc);
    end
    else
      $display("Skipped writting to PC - nop"); //nop dont update
  end

  InstructMem IM(pc,IFinst);

  IFID IFIDreg(IFFlush,clock,IFIDWrite,IFpc_plus_4,IFinst,IDinst,IDpc_plus_4);
  /**
   * Instruction Decode (ID)
   */
  assign IDRegRs[4:0]=IDinst[25:21];
  assign IDRegRt[4:0]=IDinst[20:16];
  assign IDRegRd[4:0]=IDinst[15:11];
  assign IDimm_value =
  {IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15],IDinst[15:0]};
  assign BranchAddr = (IDimm_value << 2) + IDpc_plus_4;
  assign JumpTarget[31:28] = IFpc_plus_4[31:28];
  assign JumpTarget[27:2] = IDinst[25:0];
  assign JumpTarget[1:0] = 0;
  assign IDcontrol = HazMuxCon ? ConOut : 0;
  assign PCMuxOut = jump ? JumpTarget : IFpc_plus_4;
  HazardUnit HU(IDRegRs,IDRegRt,EXRegRt,EXM[1],PCWrite,IFIDWrite,HazMuxCon);
  Control thecontrol(IDinst[31:26],ConOut,jump,bne,imm,andi,ori,addi);
  Registers
  piperegs(clock,WBWB[0],datatowrite,WBRegRd,IDRegRs,IDRegRt,IDRegAout,IDRegBout);
  IDEX
  IDEXreg(clock,IDcontrol[8:7],IDcontrol[6:4],IDcontrol[3:0],IDRegAout,IDRegBout,
  IDimm_value,IDRegRs,IDRegRt,IDRegRd,EXWB,EXM,EXEX,EXRegAout,EXRegBout,
  EXimm_value,EXRegRs,EXRegRt,EXRegRd
  );
  /**
  * Execution (EX)
  */
  assign regtopass = EXEX[3] ? EXRegRd : EXRegRt;
  assign b_value = EXEX[2] ? EXimm_value : EXRegBout;
  BIGMUX2 MUX0(ForwardA,EXRegAout,datatowrite,MEMALUOut,0,ALUSrcA);
  BIGMUX2 MUX1(ForwardB,b_value,datatowrite,MEMALUOut,0,ALUSrcB);
  ForwardUnit FU(MEMRegRd,WBRegRd,EXRegRs, EXRegRt, MEMWB[0], WBWB[0], ForwardA,
  ForwardB);
  // ALU control
  assign aluop[0] =
  (~IDinst[31]&~IDinst[30]&~IDinst[29]&IDinst[28]&~IDinst[27]&~IDinst[26])|(imm);
  assign aluop[1] =
  (~IDinst[31]&~IDinst[30]&~IDinst[29]&~IDinst[28]&~IDinst[27]&~IDinst[26])|(imm);
  ALUControl ALUcontrol(andi,ori,addi,EXEX[1:0],EXimm_value[5:0],ALUCon);
  ALU theALU(ALUCon,ALUSrcA,ALUSrcB,EXALUOut);
  EXMEM
  EXMEMreg(clock,EXWB,EXM,EXALUOut,regtopass,EXRegBout,MEMM,MEMWB,MEMALUOut,
  MEMRegRd,MEMWriteData);
  /**
  * Memory (Mem)
  */
  DATAMEM DM(MEMM[0],MEMM[1],MEMALUOut,MEMWriteData,MEMReadData);
  MEMWB
  MEMWBreg(clock,MEMWB,MEMReadData,MEMALUOut,MEMRegRd,WBWB,WBReadData,WBALUOut,WBRegRd);
  /**
  * Write Back (WB)
  */
  assign datatowrite = WBWB[1] ? WBReadData : WBALUOut;
endmodule

`ifdef TEST_A
//Test Bench A - Fibonacci:
module Pipelined_TestBench;
reg Clock;
integer i;
initial begin
Clock = 1;
end
//clock controls
always begin
Clock = ~Clock;
#25;
end
initial begin
// Instr Memory intialization
pipelined.IM.regfile[0] = 32'h8c030000;
pipelined.IM.regfile[4] = 32'h8c040001;
pipelined.IM.regfile[8] = 32'h8c050002;
pipelined.IM.regfile[12] = 32'h8c010002;
pipelined.IM.regfile[16] = 32'h10600004;
pipelined.IM.regfile[20] = 32'h00852020;
pipelined.IM.regfile[24] = 32'h00852822;
pipelined.IM.regfile[28] = 32'h00611820;
pipelined.IM.regfile[32] = 32'h1000fffb;
pipelined.IM.regfile[36] = 32'hac040006;
// Data Memory intialization
pipelined.DM.regfile[0] = 32'd8;
pipelined.DM.regfile[1] = 32'd1;
pipelined.DM.regfile[2] = -32'd1;
pipelined.DM.regfile[3] = 0;
pipelined.piperegs.regfile[0] = 0;
// Register File initialization
for (i = 0; i < 32; i = i + 1)
pipelined.piperegs.regfile[i] = 32'd0;
end
//Instantiate cpu
cpu pipelined(Clock);
endmodule
`else
//Test Bench B – Program that tests all 15 instructions (does not do anything useful):
module Pipelined_TestBench;
reg Clock;
integer i;
initial begin
Clock = 1;
end
//clock controls
always begin
Clock = ~Clock;
#25;
end
initial begin
// Instr Memory intialization
pipelined.IM.regfile[0] = 32'h8C030000; //lw R3,0(R1)
pipelined.IM.regfile[4] = 32'h8C040001;//lw R4,1(R0)
pipelined.IM.regfile[8] = 32'h00642820;//add R5,R3,R4
pipelined.IM.regfile[12] = 32'h00A43022;//sub R6,R5,R4
pipelined.IM.regfile[16] = 32'h00643824;//and R7,R3,R4
pipelined.IM.regfile[20] = 32'h00644025;//or R8,R3,R4
pipelined.IM.regfile[24] = 32'h00644827;//nor R9,R3,R4
pipelined.IM.regfile[28] = 32'h00C5502A;//slt R10,R6,R5
pipelined.IM.regfile[32] = 32'h80000008;//j startloop
pipelined.IM.regfile[36] = 32'h2063FFFF;//loop: addi R3,R3,-1
pipelined.IM.regfile[40] = 32'h14E3FFFE;//startloop: bne R3,R7,-2
pipelined.IM.regfile[44] = 32'h01295818;//mult R11,R9,R9
pipelined.IM.regfile[48] = 32'h0166601A;//div R12,R11,R6
pipelined.IM.regfile[52] = 32'h34CE0002;//ori R14,R6,2
pipelined.IM.regfile[56] = 32'h11CC0000;//beq R14,R12, next
pipelined.IM.regfile[60] = 32'hADCE0006;//sw
// Data Memory intialization
pipelined.DM.regfile[0] = 32'd8;
pipelined.DM.regfile[1] = 32'd1;
pipelined.piperegs.regfile[0] = 0;
// Register File initialization
for (i = 0; i < 32; i = i + 1)
pipelined.piperegs.regfile[i] = 32'd0;
end
//Instantiate cpu
cpu pipelined(Clock);
endmodule
`endif

