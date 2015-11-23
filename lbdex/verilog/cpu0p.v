`define MEMSIZE   'h80000
`define MEMEMPTY   8'hFF
`define NULL       8'h00
`define IOADDR    'h80000  // IO mapping address

// 管線狀態：IDLE 或 WAIT
`define IDLE     2'b00       // 閒置中
`define WAIT_M   2'b01   // 等待回應
`define WAIT_ACK 2'b10   // 等待回應

// Operand width
`define INT32 2'b11     // 32 bits
`define INT24 2'b10     // 24 bits
`define INT16 2'b01     // 16 bits
`define BYTE  2'b00     // 8  bits

// C0 register name
`define PC   cpu.C0R[0]   // Program Counter
`define EPC  cpu.C0R[1]  // exception PC value

// register name
`define SP   cpu.R[13]   // Stack Pointer
`define LR   cpu.R[14]   // Link Register
`define SW   cpu.R[15]   // Status Word

`define IR   cpu.ir         // 指令暫存器

// SW Flage
`define I2   `SW[16] // Hardware Interrupt 1, IO1 interrupt, status, 
                    // 1: in interrupt
`define I1   `SW[15] // Hardware Interrupt 0, timer interrupt, status, 
                    // 1: in interrupt
`define I0   `SW[14] // Software interrupt, status, 1: in interrupt
`define I    `SW[13] // Interrupt, 1: in interrupt
`define I2E  `SW[12]  // Hardware Interrupt 1, IO1 interrupt, Enable
`define I1E  `SW[11]  // Hardware Interrupt 0, timer interrupt, Enable
`define I0E  `SW[10]  // Software Interrupt Enable
`define IE   `SW[9]  // Interrupt Enable
`define M    `SW[8:6]  // Mode bits, itype
`define D    `SW[5]  // Debug Trace
`define V    `SW[3]  // Overflow
`define C    `SW[2]  // Carry
`define Z    `SW[1]  // Zero
`define N    `SW[0]  // Negative flag
  
`define LE   cpu.CF[0]  // Endian bit, Big Endian:0, Little Endian:1
  
`define EXE 3'b000
`define RESET 3'b001
`define ABORT 3'b010
`define IRQ 3'b011
`define ERROR 3'b100

module iFetch(input clock, reset, iReady, output reg iGet, oReady, input oGet);
  parameter name="iFetch";
  reg [1:0] state;
  reg [31:0] pc, pc0;

  always @(posedge clock) begin
    if (reset) begin oReady=0; iGet=0; state=`IDLE; end 
    else case (state) 
      `IDLE: begin // 閒置中
        #1;
        if (iReady) begin // 輸入資料已準備好
          #1;
          iMemReadStart(`PC, `INT32);
          pc0 = `PC;
          `PC = `PC+4;
          pc = `PC;
          iGet = 1; // 處理輸入資料
          #1;
          state = `WAIT_ACK; // 進入等待狀態
          oReady = 1;
        end
      end
      `WAIT_ACK:begin // 等待回應 (資料被取走)
        #1;                
        if (oGet) begin // 資料被取走了
          oReady = 0; // 下一筆輸出資料尚未準備好。
          state = `IDLE; // 回到閒置狀態，準備取得下一筆輸入資料
        end
        #1;                
        iGet = 0;  // 下一筆輸入資料尚未準備好。
      end
    endcase
  end
endmodule

module iDecode(input clock, reset, iReady, output reg iGet, oReady, input oGet);
  parameter name="iDecode";
  reg [1:0] state;
  reg [31:0] ir, pc, pc0;
  reg [7:0] op;
  reg [3:0] a, b, c;
  reg [4:0] c5;
  reg signed [11:0] cx12;
  reg signed [15:0] cx16;
  reg signed [23:0] cx24;
  reg signed [31:0] c12, c16, c24, Ra, Rb, Rc; // ipc:instruction PC

  always @(posedge clock) begin
    if (reset) begin oReady=0; iGet=0; state=`IDLE; end 
    else case (state) 
      `IDLE: begin // 閒置中
        #1;                
        if (iReady && (!cpu.m_en1 || cpu.m_ack1)) begin // 輸入資料已準備好
          iMemReadEnd(ir); // IR = dbus = m[PC]
          pc0 = cpu.if1.pc0;
          pc = cpu.if1.pc;
          $display("%-4d:fetch , pc0=%x pc=%x ir=%x", $stime, pc0, pc, ir);
          iGet = 1;
          // 處理輸入資料
          {op, a, b, c, cx12} = ir;
          cx24 = ir[23:0];
          cx16 = ir[15:0];
          c5  = ir[4:0];
          c12 = cx12; // 取出 cx12 並轉為 32 位元有號數 c12
          c16 = cx16; // 取出 cx16 並轉為 32 位元有號數 c16
          c24 = cx24; // 取出 cx24 並轉為 32 位元有號數 c24
          Ra = cpu.R[a];
          Rb = cpu.R[b];
          Rc = cpu.R[c]; 
          #1;
          oReady = 1; // 輸出資料已準備好
          state = `WAIT_ACK; // 進入等待狀態
          $display("%-4d:decode, pc0=%x pc=%x ir=%x op=%x a=%x b=%x c=%x cx12=%x", $stime, pc0, pc, ir, op, a, b, c, cx12);
        end
        #1;                
      end
      `WAIT_ACK:begin // 等待回應 (資料被取走)
        #1;                
        if (oGet) begin // 資料被取走了
          oReady = 0; // 下一筆輸出資料尚未準備好。
          state = `IDLE; // 回到閒置狀態，準備取得下一筆輸入資料
        end
        #1;                
        iGet = 0;  // 下一筆輸入資料尚未準備好。
      end
    endcase
  end
endmodule

module iExec(input clock, reset, iReady, output reg iGet, oReady, input oGet);
  parameter name="iExec";
  reg [31:0] ir, pc, pc0;
  reg [7:0] op;
  reg [3:0] a, b, c;
  reg [4:0] c5;
  reg signed [31:0] c12, c16, c24, Ra, Rb, Rc; // ipc:instruction PC
  reg [31:0] uc16, URa, URb, URc;
  reg [1:0] state;
  reg [1:0] skip;
  integer i;

  always @(posedge clock) begin
    if (reset) begin oReady=0; iGet=0; state=`IDLE; skip = 0; end 
    else case (state) 
      `IDLE: begin // 閒置中
        #1;                
        if (iReady && (!cpu.m_en2 || cpu.m_ack2)) begin // 輸入資料已準備好
          if (skip > 0) skip = skip-1; else begin
          // 處理輸入資料
          pc0 = cpu.id1.pc0;
          pc = cpu.id1.pc;
          ir = cpu.id1.ir;
          op = cpu.id1.op;
          a = cpu.id1.a;
          b = cpu.id1.b;
          c = cpu.id1.c;
          c5 = cpu.id1.c5;
          c12 = cpu.id1.c12;
          c16 = cpu.id1.c16;
          c24 = cpu.id1.c24;
          Ra = cpu.id1.Ra;
          Rb = cpu.id1.Rb;
          Rc = cpu.id1.Rc;
          URa = cpu.id1.Ra;
          URb = cpu.id1.Rb;
          URc = cpu.id1.Rc;
          iGet = 1;
          // 準備輸出資料
          case (op)
          cpu.NOP:   ;
          // load and store instructions
          cpu.LD:    dMemReadStart(Rb+c16, `INT32);      // LD Ra,[Rb+Cx]; Ra<=[Rb+Cx]
          cpu.ST:    dMemWriteStart(Rb+c16, Ra, `INT32); // ST Ra,[Rb+Cx]; Ra=>[Rb+Cx]
          // LB Ra,[Rb+Cx]; Ra<=(byte)[Rb+Cx]
          cpu.LB:    dMemReadStart(Rb+c16, `BYTE);
          // LBu Ra,[Rb+Cx]; Ra<=(byte)[Rb+Cx]
          cpu.LBu:   dMemReadStart(Rb+c16, `BYTE);
          // SB Ra,[Rb+Cx]; Ra=>(byte)[Rb+Cx]
          cpu.SB:    dMemWriteStart(Rb+c16, Ra, `BYTE);
          cpu.LH:    dMemReadStart(Rb+c16, `INT16); // LH Ra,[Rb+Cx]; Ra<=(2bytes)[Rb+Cx]
          cpu.LHu:   dMemReadStart(Rb+c16, `INT16); // LHu Ra,[Rb+Cx]; Ra<=(2bytes)[Rb+Cx]
          // SH Ra,[Rb+Cx]; Ra=>(2bytes)[Rb+Cx]
          cpu.SH:    dMemWriteStart(Rb+c16, Ra, `INT16);
          // Conditional move
          cpu.MOVZ:  if (Rc==0) regSet(a, Rb);             // move if Rc equal to 0
          cpu.MOVN:  if (Rc!=0) regSet(a, Rb);             // move if Rc not equal to 0
          // Mathematic 
          cpu.ADDiu: regSet(a, Rb+c16);                   // ADDiu Ra, Rb+Cx; Ra<=Rb+Cx
          cpu.CMP:   begin `N=(Rb-Rc<0);`Z=(Rb-Rc==0); end // CMP Rb, Rc; SW=(Rb >=< Rc)
          cpu.ADDu:  regSet(a, Rb+Rc);               // ADDu Ra,Rb,Rc; Ra<=Rb+Rc
          cpu.ADD:   begin regSet(a, Rb+Rc); if (a < Rb) `V = 1; else `V = 0; 
            if (`V) begin `I0 = 1; `I = 1; end
          end
                                                 // ADD Ra,Rb,Rc; Ra<=Rb+Rc
          cpu.SUBu:  regSet(a, Rb-Rc);               // SUBu Ra,Rb,Rc; Ra<=Rb-Rc
          cpu.SUB:   begin regSet(a, Rb-Rc); if (Rb < 0 && Rc > 0 && a >= 0) 
                 `V = 1; else `V =0; 
            if (`V) begin `I0 = 1; `I = 1; end
          end         // SUB Ra,Rb,Rc; Ra<=Rb-Rc
          cpu.CLZ:   begin
            for (i=0; (i<32)&&((Rb&32'h80000000)==32'h00000000); i=i+1) begin
                Rb=Rb<<1;
            end
            regSet(a, i);
          end
          cpu.CLO:   begin
            for (i=0; (i<32)&&((Rb&32'h80000000)==32'h80000000); i=i+1) begin
                Rb=Rb<<1;
            end
            regSet(a, i);
          end
          cpu.MUL:   regSet(a, Rb*Rc);               // MUL Ra,Rb,Rc;     Ra<=Rb*Rc
          cpu.DIVu:  regHILOSet(URa%URb, URa/URb);   // DIVu URa,URb; HI<=URa%URb; 
                                                 // LO<=URa/URb
                                                 // without exception overflow
          cpu.DIV:   begin regHILOSet(Ra%Rb, Ra/Rb); 
                 if ((Ra < 0 && Rb < 0) || (Ra == 0)) `V = 1; 
                 else `V =0; end  // DIV Ra,Rb; HI<=Ra%Rb; LO<=Ra/Rb; With overflow
          cpu.AND:   regSet(a, Rb&Rc);               // AND Ra,Rb,Rc; Ra<=(Rb and Rc)
          cpu.ANDi:  regSet(a, Rb&uc16);             // ANDi Ra,Rb,c16; Ra<=(Rb and c16)
          cpu.OR:    regSet(a, Rb|Rc);               // OR Ra,Rb,Rc; Ra<=(Rb or Rc)
          cpu.ORi:   regSet(a, Rb|uc16);             // ORi Ra,Rb,c16; Ra<=(Rb or c16)
          cpu.XOR:   regSet(a, Rb^Rc);               // XOR Ra,Rb,Rc; Ra<=(Rb xor Rc)
          cpu.XORi:  regSet(a, Rb^uc16);             // XORi Ra,Rb,c16; Ra<=(Rb xor c16)
          cpu.LUi:   regSet(a, uc16<<16);
          cpu.SHL:   regSet(a, Rb<<c5);     // Shift Left; SHL Ra,Rb,Cx; Ra<=(Rb << Cx)
          cpu.SRA:   regSet(a, (Rb&'h80000000)|(Rb>>c5)); 
                                    // Shift Right with signed bit fill;
                                    // SHR Ra,Rb,Cx; Ra<=(Rb&0x80000000)|(Rb>>Cx)
          cpu.SHR:   regSet(a, Rb>>c5);     // Shift Right with 0 fill; 
                                        // SHR Ra,Rb,Cx; Ra<=(Rb >> Cx)
          cpu.SHLV:  regSet(a, Rb<<Rc);     // Shift Left; SHLV Ra,Rb,Rc; Ra<=(Rb << Rc)
          cpu.SRAV:  regSet(a, (Rb&'h80000000)|(Rb>>Rc)); 
                                    // Shift Right with signed bit fill;
                                    // SHRV Ra,Rb,Rc; Ra<=(Rb&0x80000000)|(Rb>>Rc)
          cpu.SHRV:  regSet(a, Rb>>Rc);     // Shift Right with 0 fill; 
                                        // SHRV Ra,Rb,Rc; Ra<=(Rb >> Rc)
          cpu.ROL:   regSet(a, (Rb<<c5)|(Rb>>(32-c5)));     // Rotate Left;
          cpu.ROR:   regSet(a, (Rb>>c5)|(Rb<<(32-c5)));     // Rotate Right;
          cpu.ROLV:  begin // Can set Rc to -32<=Rc<=32 more efficently.
            while (Rc < -32) Rc=Rc+32;
            while (Rc > 32) Rc=Rc-32;
            regSet(a, (Rb<<Rc)|(Rb>>(32-Rc)));     // Rotate Left;
          end
          cpu.RORV:  begin 
            while (Rc < -32) Rc=Rc+32;
            while (Rc > 32) Rc=Rc-32;
            regSet(a, (Rb>>Rc)|(Rb<<(32-Rc)));     // Rotate Right;
          end
          cpu.MFLO:  regSet(a, cpu.LO);         // MFLO Ra; Ra<=LO
          cpu.MFHI:  regSet(a, cpu.HI);         // MFHI Ra; Ra<=HI
          cpu.MTLO:  cpu.LO = Ra;               // MTLO Ra; LO<=Ra
          cpu.MTHI:  cpu.HI = Ra;               // MTHI Ra; HI<=Ra
          cpu.MULT:  {cpu.HI, cpu.LO}=Ra*Rb;        // MULT Ra,Rb; HI<=((Ra*Rb)>>32); 
                                        // LO<=((Ra*Rb) and 0x00000000ffffffff);
                                        // with exception overflow
          cpu.MULTu: {cpu.HI, cpu.LO}=URa*URb;      // MULT URa,URb; HI<=((URa*URb)>>32); 
                                        // LO<=((URa*URb) and 0x00000000ffffffff);
                                        // without exception overflow
          cpu.MFC0:  regSet(a, cpu.C0R[b]);     // MFC0 a, b; Ra<=C0R[Rb]
          cpu.MTC0:  C0regSet(a, Rb);       // MTC0 a, b; C0R[a]<=Rb
          cpu.C0MOV: C0regSet(a, cpu.C0R[b]);   // C0MOV a, b; C0R[a]<=C0R[b]
        `ifdef CPU0II
          // set
          cpu.SLT:   if (Rb < Rc) cpu.R[a]=1; else cpu.R[a]=0;
          cpu.SLTu:  if (Rb < Rc) cpu.R[a]=1; else cpu.R[a]=0;
          cpu.SLTi:  if (Rb < c16) cpu.R[a]=1; else cpu.R[a]=0;
          cpu.SLTiu: if (Rb < c16) cpu.R[a]=1; else cpu.R[a]=0;
          // Branch Instructions
          cpu.BEQ:   if (Ra==Rb) `PC=`PC+c16; 
          cpu.BNE:   if (Ra!=Rb) `PC=`PC+c16;
        `endif
          // Jump Instructions
          cpu.JEQ:   if (`Z) `PC=`PC+c24;            // JEQ Cx; if SW(=) PC  PC+Cx
          cpu.JNE:   if (!`Z) `PC=`PC+c24;           // JNE Cx; if SW(!=) PC PC+Cx
          cpu.JLT:   if (`N)`PC=`PC+c24;             // JLT Cx; if SW(<) PC  PC+Cx
          cpu.JGT:   if (!`N&&!`Z) `PC=`PC+c24;      // JGT Cx; if SW(>) PC  PC+Cx
          cpu.JLE:   if (`N || `Z) `PC=`PC+c24;      // JLE Cx; if SW(<=) PC PC+Cx    
          cpu.JGE:   if (!`N || `Z) `PC=`PC+c24;     // JGE Cx; if SW(>=) PC PC+Cx
          cpu.JMP:   `PC = `PC+c24;                  // JMP Cx; PC <= PC+Cx
          cpu.JALR:  begin regSet(a, `PC);`PC=Rb; end // JALR Ra,Rb; Ra<=PC; PC<=Rb
          cpu.BAL:   begin `LR=`PC;`PC=`PC + c24; end // BAL Cx; LR<=PC; PC<=PC+Cx
          cpu.JSUB:  begin `LR=`PC;`PC=`PC + c24; end // JSUB Cx; LR<=PC; PC<=PC+Cx
          cpu.RET:   begin `PC=Ra; end               // RET; PC <= Ra
          default : 
            $display("%4dns %8x : OP code %8x not support", $stime, pc0, op);
          endcase
          Ra = cpu.R[a];
          $display("%-4d:exec  , pc0=%x ir=%x Ra=%x=%-4d Rb=%x Rc=%x", $stime, pc0, ir, Ra, Ra, Rb, Rc);
          end
          #1;                
          oReady = 1; // 輸出資料已準備好
          state = `WAIT_ACK; // 進入等待狀態
        end
      end
      `WAIT_ACK:begin // 等待回應 (資料被取走)
        #1;                
        if (oGet) begin // 資料被取走了
            #1;                
            oReady = 0; // 下一筆輸出資料尚未準備好。
            state = `IDLE; // 回到閒置狀態，準備取得下一筆輸入資料
        end
        #1;                
        iGet = 0;  // 下一筆輸入資料尚未準備好。
      end
    endcase
  end
endmodule

module iMemAccess(input clock, reset, iReady, output reg iGet, oReady, input oGet);
  parameter name="iMemAccess";
  reg [1:0] state;
  reg [31:0] ir, pc, pc0;
  reg [7:0] op;
  reg [3:0] a, b, c;
  reg [4:0] c5;
  reg signed [11:0] cx12;
  reg signed [15:0] cx16;
  reg signed [23:0] cx24;
  reg signed [31:0] c12, c16, c24, Ra, Rb, Rc; // ipc:instruction PC

  always @(posedge clock) begin
    if (reset) begin oReady=0; iGet=0; state=`IDLE; end 
    else case (state) 
      `IDLE: begin // 閒置中
        #1;                
        if (iReady && (!cpu.m_en1 || cpu.m_ack1)) begin // 輸入資料已準備好
          pc0 = cpu.ie1.pc0;
          pc = cpu.ie1.pc;
          ir = cpu.ie1.ir;
          $display("%-4d:fetch , pc0=%x pc=%x ir=%x", $stime, pc0, pc, ir);
          iGet = 1;
          // 處理輸入資料
          {op, a, b, c, cx12} = ir;
          cx24 = ir[23:0];
          cx16 = ir[15:0];
          c5  = ir[4:0];
          c12 = cx12; // 取出 cx12 並轉為 32 位元有號數 c12
          c16 = cx16; // 取出 cx16 並轉為 32 位元有號數 c16
          c24 = cx24; // 取出 cx24 並轉為 32 位元有號數 c24
          Ra = cpu.R[a];
          Rb = cpu.R[b];
          Rc = cpu.R[c]; 
          case (op)
            cpu.ST, cpu.SB, cpu.SH  :  dMemWriteEnd(); // 寫入記憶體完成
          endcase
          #1;
          oReady = 1; // 輸出資料已準備好
          state = `WAIT_ACK; // 進入等待狀態
          $display("%-4d:decode, pc0=%x pc=%x ir=%x op=%x a=%x b=%x c=%x cx12=%x", $stime, pc0, pc, ir, op, a, b, c, cx12);
        end
        #1;                
      end
      `WAIT_ACK:begin // 等待回應 (資料被取走)
        #1;                
        if (oGet) begin // 資料被取走了
          oReady = 0; // 下一筆輸出資料尚未準備好。
          state = `IDLE; // 回到閒置狀態，準備取得下一筆輸入資料
        end
        #1;                
        iGet = 0;  // 下一筆輸入資料尚未準備好。
      end
    endcase
  end
endmodule

module iWriteBack(input clock, reset, iReady, output reg iGet, oReady, input oGet);
  parameter name="iWriteBack";
  reg [1:0] state;
  reg [31:0] ir, pc, pc0;
  reg [7:0] op;
  reg [3:0] a, b, c;
  reg [4:0] c5;
  reg signed [11:0] cx12;
  reg signed [15:0] cx16;
  reg signed [23:0] cx24;
  reg signed [31:0] c12, c16, c24, Ra, Rb, Rc; // ipc:instruction PC

  always @(posedge clock) begin
    if (reset) begin oReady=0; iGet=0; state=`IDLE; end 
    else case (state) 
      `IDLE: begin // 閒置中
        #1;                
        if (iReady && (!cpu.m_en1 || cpu.m_ack1)) begin // 輸入資料已準備好
          pc0 = cpu.im1.pc0;
          pc = cpu.im1.pc;
          ir = cpu.im1.ir;
          $display("%-4d:fetch , pc0=%x pc=%x ir=%x", $stime, pc0, pc, ir);
          iGet = 1;
          // 處理輸入資料
          {op, a, b, c, cx12} = ir;
          cx24 = ir[23:0];
          cx16 = ir[15:0];
          c5  = ir[4:0];
          c12 = cx12; // 取出 cx12 並轉為 32 位元有號數 c12
          c16 = cx16; // 取出 cx16 並轉為 32 位元有號數 c16
          c24 = cx24; // 取出 cx24 並轉為 32 位元有號數 c24
          Ra = cpu.R[a];
          Rb = cpu.R[b];
          Rc = cpu.R[c];
          case (op)
          cpu.LB, cpu.LBu  :
            dMemReadEnd(Ra);        //read memory complete
          cpu.LH, cpu.LHu  :
            dMemReadEnd(Ra);
          cpu.LD  : begin
            dMemReadEnd(Ra);
            if (`D)
              $display("%4dns %8x : %8x m[%-04x+%-04x]=%8x  SW=%8x", $stime, pc0, 
                       ir, Rb, c16, Ra, `SW);
          end
          endcase
          case (op)
          cpu.LB  : begin 
            if (Ra > 8'h7f) regSet(a, Ra|32'hffffff80);
          end
          cpu.LH  : begin 
            if (Ra > 16'h7fff) regSet(a, Ra|32'hffff8000);
          end
          endcase
          case (op)
          cpu.MULT, cpu.MULTu, cpu.DIV, cpu.DIVu, cpu.MTHI, cpu.MTLO :
            if (`D)
              $display("%4dns %8x : %8x HI=%8x LO=%8x SW=%8x", $stime, pc0, ir, cpu.HI, 
                       cpu.LO, `SW);
          cpu.ST : begin
            if (`D)
              $display("%4dns %8x : %8x m[%-04x+%-04x]=%8x  SW=%8x", $stime, pc0, 
                       ir, Rb, c16, Ra, `SW);
            if (Rb+c16 == `IOADDR) begin
              outw(Ra);
            end
          end
          cpu.SB : begin
            if (`D)
              $display("%4dns %8x : %8x m[%-04x+%-04x]=%c  SW=%8x, R[a]=%8x", 
                       $stime, pc0, ir, Rb, c16, Ra[7:0], `SW, Ra);
            if (Rb+c16 == `IOADDR) begin
              if (`LE)
                outc(Ra[7:0]);
              else
                outc(Ra[7:0]);
            end
          end
          cpu.MFC0, cpu.MTC0 :
            if (`D)
              $display("%4dns %8x : %8x R[%02d]=%-8x  C0R[%02d]=%-8x SW=%8x", 
                       $stime, pc0, ir, a, Ra, a, cpu.C0R[a], `SW);
          cpu.C0MOV :
            if (`D)
              $display("%4dns %8x : %8x C0R[%02d]=%-8x C0R[%02d]=%-8x SW=%8x", 
                       $stime, pc0, ir, a, cpu.C0R[a], b, cpu.C0R[b], `SW);
          default :
            if (`D) // Display the written register content
              $display("%4dns %8x : %8x R[%02d]=%-8x SW=%8x", $stime, pc0, ir, 
                       a, Ra, `SW);
          endcase
          if (`PC < 0) begin
            $display("total cpu cycles = %-d", cpu.cycles);
            $display("RET to PC < 0, finished!");
            $finish;
          end
          #1;
          oReady = 1; // 輸出資料已準備好
          state = `WAIT_ACK; // 進入等待狀態
          $display("%-4d:decode, pc0=%x pc=%x ir=%x op=%x a=%x b=%x c=%x cx12=%x", $stime, pc0, pc, ir, op, a, b, c, cx12);
        end
        #1;                
      end
      `WAIT_ACK:begin // 等待回應 (資料被取走)
        #1;                
        if (oGet) begin // 資料被取走了
            oReady = 0; // 下一筆輸出資料尚未準備好。
            state = `IDLE; // 回到閒置狀態，準備取得下一筆輸入資料
        end
        #1;                
        iGet = 0;  // 下一筆輸入資料尚未準備好。
      end
    endcase
  end
endmodule

// Reference web: http://ccckmit.wikidot.com/ocs:cpu0
module cpu0(input clock, reset, output reg [2:0] tick, 
            output reg [31:0] ir, pc, 
            output [31:0] mar1, mdr1, inout [31:0] dbus1, output reg m_en1, m_rw1, input m_ack1, output reg [1:0] m_size1, 
            output [31:0] mar2, mdr2, inout [31:0] dbus2, output reg m_en2, m_rw2, input m_ack2, output reg [1:0] m_size2, 
            input cfg);        
 // 管線相關參數
  wire ifiGet, idiGet, ieiGet, iwiGet; // pipe 輸入是否準備好了
  wire ifoReady, idoReady, ieoReady, iwoReady; // pipe 輸出是否準備好了
  parameter iReady = 1'b1, oGet=1'b1; // pipeline 的整體輸入輸出是否準備好了 (隨時都準備好，這樣才會不斷驅動)。
  // 暫存器與欄位
  reg [31:0] mar1, mdr1, mar2, mdr2;
  reg signed [31:0] R [0:15];
  reg signed [31:0] C0R [0:1]; // co-processor 0 register
  // High and Low part of 64 bit result
  reg [7:0] op;
  reg [3:0] a, b, c;
  reg [4:0] c5;
  reg signed [31:0] c12, c16, c24, Ra, Rb, Rc, pc0; // pc0: instruction pc
  reg [31:0] uc16, URa, URb, URc, HI, LO, CF, tmp;
  reg [63:0] cycles;

  // Instruction Opcode 
  parameter [7:0] NOP=8'h00,LD=8'h01,ST=8'h02,LB=8'h03,LBu=8'h04,SB=8'h05,
  LH=8'h06,LHu=8'h07,SH=8'h08,ADDiu=8'h09,MOVZ=8'h0A,MOVN=8'h0B,ANDi=8'h0C,
  ORi=8'h0D,XORi=8'h0E,LUi=8'h0F,
  CMP=8'h10,
  ADDu=8'h11,SUBu=8'h12,ADD=8'h13,SUB=8'h14,CLZ=8'h15,CLO=8'h16,MUL=8'h17,
  AND=8'h18,OR=8'h19,XOR=8'h1A,
  ROL=8'h1B,ROR=8'h1C,SRA=8'h1D,SHL=8'h1E,SHR=8'h1F,
  SRAV=8'h20,SHLV=8'h21,SHRV=8'h22,ROLV=8'h23,RORV=8'h24,
`ifdef CPU0II
  SLTi=8'h26,SLTiu=8'h27, SLT=8'h28,SLTu=8'h29,
  BEQ=8'h37,BNE=8'h38,
`endif
  JEQ=8'h30,JNE=8'h31,JLT=8'h32,JGT=8'h33,JLE=8'h34,JGE=8'h35,
  JMP=8'h36,
  JALR=8'h39,BAL=8'h3A,JSUB=8'h3B,RET=8'h3C,
  MULT=8'h41,MULTu=8'h42,DIV=8'h43,DIVu=8'h44,
  MFHI=8'h46,MFLO=8'h47,MTHI=8'h48,MTLO=8'h49,
  MFC0=8'h50,MTC0=8'h51,C0MOV=8'h52;

  reg [0:0] inExe = 0;
  reg [2:0] state, next_state; 
  reg [2:0] st_taskInt, ns_taskInt; 
  parameter Reset=3'h0, Fetch=3'h1, Decode=3'h2, Execute=3'h3, MemAccess=3'h4, 
            WriteBack=3'h5;
  integer i;

  //transform data from the memory to little-endian form
  task changeEndian(input [31:0] value, output [31:0] changeEndian); begin
    changeEndian = {value[7:0], value[15:8], value[23:16], value[31:24]};
  end endtask

  // Read Memory Word
  task iMemReadStart(input [31:0] addr, input [1:0] size); begin 
    mar1 = addr;     // read(m[addr])
    m_rw1 = 1;     // Access Mode: read 
    m_en1 = 1;     // Enable read
    m_size1 = size;
  end endtask

  // Read Memory Finish, get data
  task iMemReadEnd(output [31:0] data); begin
    mdr1 = dbus1; // get momory, dbus = m[addr]
    data = dbus1; // return to data
    m_en1 = 0; // read complete
  end endtask

  // Read Memory Word
  task dMemReadStart(input [31:0] addr, input [1:0] size); begin 
    mar2 = addr;     // read(m[addr])
    m_rw2 = 1;     // Access Mode: read 
    m_en2 = 1;     // Enable read
    m_size2 = size;
  end endtask

  // Read Memory Finish, get data
  task dMemReadEnd(output [31:0] data); begin
    mdr2 = dbus2; // get momory, dbus = m[addr]
    data = dbus2; // return to data
    m_en2 = 0; // read complete
  end endtask

  // Write memory -- addr: address to write, data: date to write
  task dMemWriteStart(input [31:0] addr, input [31:0] data, input [1:0] size); 
  begin 
    mar2 = addr;    // write(m[addr], data)
    mdr2 = data;
    m_rw2 = 0;    // access mode: write
    m_en2 = 1;     // Enable write
    m_size2  = size;
  end endtask

  task dMemWriteEnd; begin // Write Memory Finish
    m_en2 = 0; // write complete
  end endtask

  task regSet(input [3:0] i, input [31:0] data); begin
    if (i != 0) R[i] = data;
  end endtask

  task C0regSet(input [3:0] i, input [31:0] data); begin
    if (i < 2) C0R[i] = data;
  end endtask

  task regHILOSet(input [31:0] data1, input [31:0] data2); begin
    HI = data1;
    LO = data2;
  end endtask

  // output a word to Output port (equal to display the word to terminal)
  task outw(input [31:0] data); begin
    if (`LE) begin // Little Endian
      changeEndian(data, data);
    end 
    if (data[7:0] != 8'h00) begin
      $write("%c", data[7:0]);
      if (data[15:8] != 8'h00) 
        $write("%c", data[15:8]);
      if (data[23:16] != 8'h00) 
        $write("%c", data[23:16]);
      if (data[31:24] != 8'h00) 
        $write("%c", data[31:24]);
    end
  end endtask

  // output a character (a byte)
  task outc(input [7:0] data); begin
    $write("%c", data);
  end endtask

  iFetch      if1(clock, reset, iReady,   ifiGet, ifoReady, idiGet); // pipeline：
  iDecode     id1(clock, reset, ifoReady, idiGet, idoReady, ieiGet);
  iExec       ie1(clock, reset, idoReady, ieiGet, ieoReady, oGet);
  iMemAccess  im1(clock, reset, idoReady, ieiGet, ieoReady, oGet);
  iWriteBack  iw1(clock, reset, idoReady, ieiGet, ieoReady, oGet);

  always @(posedge clock) begin
    if (reset) begin
        `PC = 0; tick = 0; R[0] = 0; `SW = 0; `LR = -1;
        `IE = 0; `I0E = 1; `I1E = 1; `I2E = 1;
        `I = 0; `I0 = 0; `I1 = 0; `I2 = 0; inExe = 1;
        `LE = cfg;
        cycles = 0;
        `D = 1; // Trace register content at beginning
    end
  end
endmodule

module memory0(input clock, reset, en, rw, input [1:0] m_size, 
               input [31:0] abus, dbus_in, output [31:0] dbus_out, 
               output cfg);
  reg [31:0] mconfig [0:0];
  reg [7:0] m [0:`MEMSIZE-1];
`ifdef DLINKER
  reg [7:0] flash [0:`MEMSIZE-1];
  reg [7:0] dsym [0:192-1];
  reg [7:0] dstr [0:96-1];
  reg [7:0] so_func_offset[0:384-1];
  reg [7:0] globalAddr [0:3];
  reg [31:0] pltAddr [0:0];
  reg [31:0] gp;
  reg [31:0] gpPlt;
  reg [31:0] fabus;
  integer j;
  integer k;
  integer l;
  reg [31:0] j32;
  integer numDynEntry;
`endif
  reg [31:0] data;

  integer i;

  `define LE  mconfig[0][0:0]   // Endian bit, Big Endian:0, Little Endian:1

`ifdef DLINKER
`include "dynlinker.v"
`endif
  initial begin
  // erase memory
    for (i=0; i < `MEMSIZE; i=i+1) begin
       m[i] = `MEMEMPTY;
    end
  // load config from file to memory
    $readmemh("cpu0.config", mconfig);
  // load program from file to memory
    $readmemh("cpu0.hex", m);
  // display memory contents
    `ifdef TRACE
      for (i=0; i < `MEMSIZE && (m[i] != `MEMEMPTY || m[i+1] != `MEMEMPTY || 
         m[i+2] != `MEMEMPTY || m[i+3] != `MEMEMPTY); i=i+4) begin
        $display("%8x: %8x", i, {m[i], m[i+1], m[i+2], m[i+3]});
      end
    `endif
`ifdef DLINKER
  loadToFlash();
  createDynInfo();
`endif
  end

  always @(clock or abus or en or rw or dbus_in) 
  begin
    if (abus >= 0 && abus <= `MEMSIZE-4) begin
      if (en == 1 && rw == 0) begin // r_w==0:write
        data = dbus_in;
        if (`LE) begin // Little Endian
          case (m_size)
          `BYTE:  {m[abus]} = dbus_in[7:0];
          `INT16: {m[abus], m[abus+1] } = {dbus_in[7:0], dbus_in[15:8]};
          `INT24: {m[abus], m[abus+1], m[abus+2]} = 
                  {dbus_in[7:0], dbus_in[15:8], dbus_in[23:16]};
          `INT32: {m[abus], m[abus+1], m[abus+2], m[abus+3]} = 
                  {dbus_in[7:0], dbus_in[15:8], dbus_in[23:16], dbus_in[31:24]};
          endcase
        end else begin // Big Endian
          case (m_size)
          `BYTE:  {m[abus]} = dbus_in[7:0];
          `INT16: {m[abus], m[abus+1] } = dbus_in[15:0];
          `INT24: {m[abus], m[abus+1], m[abus+2]} = dbus_in[23:0];
          `INT32: {m[abus], m[abus+1], m[abus+2], m[abus+3]} = dbus_in;
          endcase
        end
      end else if (en == 1 && rw == 1) begin // r_w==1:read
        if (`LE) begin // Little Endian
          case (m_size)
          `BYTE:  data = {8'h00,     8'h00,     8'h00,     m[abus]};
          `INT16: data = {8'h00,     8'h00,     m[abus+1], m[abus]};
          `INT24: data = {8'h00,     m[abus+2], m[abus+1], m[abus]};
          `INT32: data = {m[abus+3], m[abus+2], m[abus+1], m[abus]};
          endcase
        end else begin // Big Endian
          case (m_size)
          `BYTE:  data = {8'h00  , 8'h00,     8'h00,     m[abus]  };
          `INT16: data = {8'h00  , 8'h00,     m[abus],   m[abus+1]};
          `INT24: data = {8'h00  , m[abus],   m[abus+1], m[abus+2]};
          `INT32: data = {m[abus], m[abus+1], m[abus+2], m[abus+3]};
          endcase
        end
      end else
        data = 32'hZZZZZZZZ;
      `ifdef DLINKER
      `include "flashio.v"
      `endif
    end else 
      data = 32'hZZZZZZZZ;
  end
  assign dbus_out = data;
  assign cfg = mconfig[0][0:0];
endmodule

module main;
  reg clock, reset;
  reg [2:0] itype;
  wire [2:0] tick;
  wire [31:0] pc, ir;
  wire [31:0] mar1, mar2, mdr1, mdr2, dbus1, dbus2;
  wire m_en1, m_en2, m_rw1, m_rw2, m_ack1, m_ack2;
  wire [1:0] m_size1, m_size2;
  wire cfg;

  cpu0 cpu(.clock(clock), .reset(reset), .pc(pc), .tick(tick), .ir(ir),
  .mar1(mar1), .mdr1(mdr1), .dbus1(dbus1), .m_en1(m_en1), .m_rw1(m_rw1), .m_size1(m_size1), .m_ack1(m_ack1),
  .mar2(mar2), .mdr2(mdr2), .dbus2(dbus2), .m_en2(m_en2), .m_rw2(m_rw2), .m_size2(m_size2), .m_ack2(m_ack2),
  .cfg(cfg));

  memory0 imem(.clock(clock), .reset(reset), .en(m_en1), .rw(m_rw1), 
  .m_size(m_size1), .abus(mar1), .dbus_in(mdr1), .dbus_out(dbus1), .cfg(cfg));

  memory0 dmem(.clock(clock), .reset(reset), .en(m_en2), .rw(m_rw2), 
  .m_size(m_size2), .abus(mar2), .dbus_in(mdr2), .dbus_out(dbus2), .cfg(cfg));
  
  initial
  begin
    clock = 0;
    itype = `RESET;
        reset = 1;
        #50 reset = 0;
    #300000000 $finish;
  end

  always #10 clock=clock+1;

endmodule
