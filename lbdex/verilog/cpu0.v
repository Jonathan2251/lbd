// https://www.francisz.cn/download/IEEE_Standard_1800-2012%20SystemVerilog.pdf

`define SIMULATE_DELAY_SLOT
// cpu032I memory limit, jsub:24-bit
`define MEMSIZE   'h1000000
`define MEMEMPTY   8'hFF
`define NULL       8'h00
`define IOADDR    'h1000000  // IO mapping address

// Operand width
`define INT32 2'b11     // 32 bits
`define INT24 2'b10     // 24 bits
`define INT16 2'b01     // 16 bits
`define BYTE  2'b00     // 8  bits

`define EXE 3'b000
`define RESET 3'b001
`define ABORT 3'b010
`define IRQ 3'b011
`define ERROR 3'b100

// Reference web: http://ccckmit.wikidot.com/ocs:cpu0
module cpu0(input clock, reset, input [2:0] itype, output reg [2:0] tick, 
            output reg [31:0] ir, pc, mar, mdr, inout [31:0] dbus, 
            output reg m_en, m_rw, output reg [1:0] m_size, 
            input cfg);
  reg signed [31:0] R [0:15];
  reg signed [31:0] C0R [0:1]; // co-processor 0 register
  // High and Low part of 64 bit result
  reg [7:0] op;
  reg [3:0] a, b, c;
  reg [4:0] c5;
  reg signed [31:0] c12, c16, c24, Ra, Rb, Rc, pc0; // pc0: instruction pc
  reg [31:0] uc16, URa, URb, URc, HI, LO, CF, tmp;
  reg [63:0] cycles;

  // register name
  `define SP   R[13]   // Stack Pointer
  `define LR   R[14]   // Link Register
  `define SW   R[15]   // Status Word

  // C0 register name
  `define PC   C0R[0]   // Program Counter
  `define EPC  C0R[1]  // exception PC value

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
  
  `define LE   CF[0]  // Endian bit, Big Endian:0, Little Endian:1
  // Instruction Opcode 
  parameter [7:0] NOP=8'h00,LD=8'h01,ST=8'h02,LB=8'h03,LBu=8'h04,SB=8'h05,
  LH=8'h06,LHu=8'h07,SH=8'h08,ADDiu=8'h09,MOVZ=8'h0A,MOVN=8'h0B,ANDi=8'h0C,
  ORi=8'h0D,XORi=8'h0E,LUi=8'h0F,
  ADDu=8'h11,SUBu=8'h12,ADD=8'h13,SUB=8'h14,CLZ=8'h15,CLO=8'h16,MUL=8'h17,
  AND=8'h18,OR=8'h19,XOR=8'h1A,NOR=8'h1B,
  ROL=8'h1C,ROR=8'h1D,SHL=8'h1E,SHR=8'h1F,
  SRA=8'h20,SRAV=8'h21,SHLV=8'h22,SHRV=8'h23,ROLV=8'h24,RORV=8'h25,
`ifdef CPU0II
  SLTi=8'h26,SLTiu=8'h27, SLT=8'h28,SLTu=8'h29,
`endif
  CMP=8'h2A,
  CMPu=8'h2B,
  JEQ=8'h30,JNE=8'h31,JLT=8'h32,JGT=8'h33,JLE=8'h34,JGE=8'h35,
  JMP=8'h36,
`ifdef CPU0II
  BEQ=8'h37,BNE=8'h38,
`endif
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
`ifdef SIMULATE_DELAY_SLOT
  reg [0:0] nextInstIsDelaySlot;
  reg [0:0] isDelaySlot;
  reg signed [31:0] delaySlotNextPC;
`endif

  //transform data from the memory to little-endian form
  task changeEndian(input [31:0] value, output [31:0] changeEndian); begin
    changeEndian = {value[7:0], value[15:8], value[23:16], value[31:24]};
  end endtask

  // Read Memory Word
  task memReadStart(input [31:0] addr, input [1:0] size); begin 
    mar = addr;     // read(m[addr])
    m_rw = 1;     // Access Mode: read 
    m_en = 1;     // Enable read
    m_size = size;
  end endtask

  // Read Memory Finish, get data
  task memReadEnd(output [31:0] data); begin
    mdr = dbus; // get momory, dbus = m[addr]
    data = mdr; // return to data
    m_en = 0; // read complete
  end endtask

  // Write memory -- addr: address to write, data: date to write
  task memWriteStart(input [31:0] addr, input [31:0] data, input [1:0] size); 
  begin 
    mar = addr;    // write(m[addr], data)
    mdr = data;
    m_rw = 0;    // access mode: write
    m_en = 1;     // Enable write
    m_size  = size;
  end endtask

  task memWriteEnd; begin // Write Memory Finish
    m_en = 0; // write complete
  end endtask

  task regSet(input [3:0] i, input [31:0] data); begin
    if (i != 0) R[i] = data;
  end endtask

  task C0regSet(input [3:0] i, input [31:0] data); begin
    if (i < 2) C0R[i] = data;
  end endtask

  task PCSet(input [31:0] data); begin
  `ifdef SIMULATE_DELAY_SLOT
    nextInstIsDelaySlot = 1;
    delaySlotNextPC = data;
  `else
    `PC = data;
  `endif
  end endtask

  task retValSet(input [3:0] i, input [31:0] data); begin
    if (i != 0)
    `ifdef SIMULATE_DELAY_SLOT
      R[i] = data + 4;
    `else
      R[i] = data;
    `endif
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

  task taskInterrupt(input [2:0] iMode); begin
  if (inExe == 0) begin
    case (iMode)
      `RESET: begin 
        `PC = 0; tick = 0; R[0] = 0; `SW = 0; `LR = -1;
        `IE = 0; `I0E = 1; `I1E = 1; `I2E = 1;
        `I = 0; `I0 = 0; `I1 = 0; `I2 = 0; inExe = 1;
        `LE = cfg;
        cycles = 0;
      end
      `ABORT: begin `PC = 4; end
      `IRQ:   begin `PC = 8; `IE = 0; inExe = 1; end
      `ERROR: begin `PC = 12; end
    endcase
  end
  $display("taskInterrupt(%3b)", iMode);
  end endtask

  task taskExecute; begin
    tick = tick+1;
    case (state)
    Fetch: begin  // Tick 1 : instruction fetch, throw PC to address bus, 
                  // memory.read(m[PC])
      memReadStart(`PC, `INT32);
      pc0  = `PC;
   `ifdef SIMULATE_DELAY_SLOT
     if (nextInstIsDelaySlot == 1) begin
       isDelaySlot = 1;
       nextInstIsDelaySlot = 0;
       `PC = delaySlotNextPC;
     end
     else begin
       if (isDelaySlot == 1) isDelaySlot = 0;
       `PC = `PC+4;
     end
   `else
     `PC = `PC+4;
   `endif
      next_state = Decode;
    end
    Decode: begin  // Tick 2 : instruction decode, ir = m[PC]
      memReadEnd(ir); // IR = dbus = m[PC]
      {op,a,b,c} = ir[31:12];
      c24 = $signed(ir[23:0]);
      c16 = $signed(ir[15:0]);
      uc16 = ir[15:0];
      c12 = $signed(ir[11:0]);
      c5  = ir[4:0];
      Ra = R[a];
      Rb = R[b];
      Rc = R[c];
      URa = R[a];
      URb = R[b];
      URc = R[c];
      next_state = Execute;
    end
    Execute: begin // Tick 3 : instruction execution
      case (op)
      NOP:   ;
      // load and store instructions
      LD:    memReadStart(Rb+c16, `INT32);      // LD Ra,[Rb+Cx]; Ra<=[Rb+Cx]
      ST:    memWriteStart(Rb+c16, Ra, `INT32); // ST Ra,[Rb+Cx]; Ra=>[Rb+Cx]
      // LB Ra,[Rb+Cx]; Ra<=(byte)[Rb+Cx]
      LB:    memReadStart(Rb+c16, `BYTE);
      // LBu Ra,[Rb+Cx]; Ra<=(byte)[Rb+Cx]
      LBu:   memReadStart(Rb+c16, `BYTE);
      // SB Ra,[Rb+Cx]; Ra=>(byte)[Rb+Cx]
      SB:    memWriteStart(Rb+c16, Ra, `BYTE);
      LH:    memReadStart(Rb+c16, `INT16); // LH Ra,[Rb+Cx]; Ra<=(2bytes)[Rb+Cx]
      LHu:   memReadStart(Rb+c16, `INT16); // LHu Ra,[Rb+Cx]; Ra<=(2bytes)[Rb+Cx]
      // SH Ra,[Rb+Cx]; Ra=>(2bytes)[Rb+Cx]
      SH:    memWriteStart(Rb+c16, Ra, `INT16);
      // Conditional move
      MOVZ:  if (Rc==0) regSet(a, Rb);             // move if Rc equal to 0
      MOVN:  if (Rc!=0) regSet(a, Rb);             // move if Rc not equal to 0
      // Mathematic 
      ADDiu: regSet(a, Rb+c16);                   // ADDiu Ra, Rb+Cx; Ra<=Rb+Cx
      CMP:  begin 
        if (Rb < Rc) `N=1; else `N=0; 
       `Z=(Rb-Rc==0); 
      end // CMP Rb, Rc; SW=(Rb >=< Rc)
      CMPu:  begin 
        if (URb < URc) `N=1; else `N=0; 
       `Z=(URb-URc==0); 
      end // CMPu URb, URc; SW=(URb >=< URc)
      ADDu:  regSet(a, Rb+Rc);               // ADDu Ra,Rb,Rc; Ra<=Rb+Rc
      ADD:   begin regSet(a, Rb+Rc); if (a < Rb) `V = 1; else `V = 0; 
        if (`V) begin `I0 = 1; `I = 1; end
      end
                                             // ADD Ra,Rb,Rc; Ra<=Rb+Rc
      SUBu:  regSet(a, Rb-Rc);               // SUBu Ra,Rb,Rc; Ra<=Rb-Rc
      SUB:   begin regSet(a, Rb-Rc); if (Rb < 0 && Rc > 0 && a >= 0) 
             `V = 1; else `V =0; 
        if (`V) begin `I0 = 1; `I = 1; end
      end         // SUB Ra,Rb,Rc; Ra<=Rb-Rc
      CLZ:   begin
        for (i=0; (i<32)&&((Rb&32'h80000000)==32'h00000000); i=i+1) begin
            Rb=Rb<<1;
        end
        regSet(a, i);
      end
      CLO:   begin
        for (i=0; (i<32)&&((Rb&32'h80000000)==32'h80000000); i=i+1) begin
            Rb=Rb<<1;
        end
        regSet(a, i);
      end
      MUL:   regSet(a, Rb*Rc);               // MUL Ra,Rb,Rc;     Ra<=Rb*Rc
      DIVu:  regHILOSet(URa%URb, URa/URb);   // DIVu URa,URb; HI<=URa%URb; 
                                             // LO<=URa/URb
                                             // without exception overflow
      DIV:   begin regHILOSet(Ra%Rb, Ra/Rb); 
             if ((Ra < 0 && Rb < 0) || (Ra == 0)) `V = 1; 
             else `V =0; end  // DIV Ra,Rb; HI<=Ra%Rb; LO<=Ra/Rb; With overflow
      AND:   regSet(a, Rb&Rc);               // AND Ra,Rb,Rc; Ra<=(Rb and Rc)
      ANDi:  regSet(a, Rb&uc16);             // ANDi Ra,Rb,c16; Ra<=(Rb and c16)
      OR:    regSet(a, Rb|Rc);               // OR Ra,Rb,Rc; Ra<=(Rb or Rc)
      ORi:   regSet(a, Rb|uc16);             // ORi Ra,Rb,c16; Ra<=(Rb or c16)
      XOR:   regSet(a, Rb^Rc);               // XOR Ra,Rb,Rc; Ra<=(Rb xor Rc)
      NOR:   regSet(a, ~(Rb|Rc));            // NOR Ra,Rb,Rc; Ra<=(Rb nor Rc)
      XORi:  regSet(a, Rb^uc16);             // XORi Ra,Rb,c16; Ra<=(Rb xor c16)
      LUi:   regSet(a, uc16<<16);
      SHL:   regSet(a, Rb<<c5);     // Shift Left; SHL Ra,Rb,Cx; Ra<=(Rb << Cx)
      SRA:   regSet(a, (Rb>>>c5));  // Shift Right with signed bit fill;
        // https://stackoverflow.com/questions/39911655/how-to-synthesize-hardware-for-sra-instruction
      SHR:   regSet(a, Rb>>c5);     // Shift Right with 0 fill; 
                                    // SHR Ra,Rb,Cx; Ra<=(Rb >> Cx)
      SHLV:  regSet(a, Rb<<Rc);     // Shift Left; SHLV Ra,Rb,Rc; Ra<=(Rb << Rc)
      SRAV:  regSet(a, (Rb>>>Rc));  // Shift Right with signed bit fill;
      SHRV:  regSet(a, Rb>>Rc);     // Shift Right with 0 fill; 
                                    // SHRV Ra,Rb,Rc; Ra<=(Rb >> Rc)
      ROL:   regSet(a, (Rb<<c5)|(Rb>>(32-c5)));     // Rotate Left;
      ROR:   regSet(a, (Rb>>c5)|(Rb<<(32-c5)));     // Rotate Right;
      ROLV:  begin // Can set Rc to -32<=Rc<=32 more efficently.
        while (Rc < -32) Rc=Rc+32;
        while (Rc > 32) Rc=Rc-32;
        regSet(a, (Rb<<Rc)|(Rb>>(32-Rc)));     // Rotate Left;
      end
      RORV:  begin 
        while (Rc < -32) Rc=Rc+32;
        while (Rc > 32) Rc=Rc-32;
        regSet(a, (Rb>>Rc)|(Rb<<(32-Rc)));     // Rotate Right;
      end
      MFLO:  regSet(a, LO);         // MFLO Ra; Ra<=LO
      MFHI:  regSet(a, HI);         // MFHI Ra; Ra<=HI
      MTLO:  LO = Ra;               // MTLO Ra; LO<=Ra
      MTHI:  HI = Ra;               // MTHI Ra; HI<=Ra
      MULT:  {HI, LO}=Ra*Rb;        // MULT Ra,Rb; HI<=((Ra*Rb)>>32); 
                                    // LO<=((Ra*Rb) and 0x00000000ffffffff);
                                    // with exception overflow
      MULTu: {HI, LO}=URa*URb;      // MULT URa,URb; HI<=((URa*URb)>>32); 
                                    // LO<=((URa*URb) and 0x00000000ffffffff);
                                    // without exception overflow
      MFC0:  regSet(a, C0R[b]);     // MFC0 a, b; Ra<=C0R[Rb]
      MTC0:  C0regSet(a, Rb);       // MTC0 a, b; C0R[a]<=Rb
      C0MOV: C0regSet(a, C0R[b]);   // C0MOV a, b; C0R[a]<=C0R[b]
   `ifdef CPU0II
      // set
      SLT:   if (Rb < Rc) R[a]=1; else R[a]=0;
      SLTu:  if (URb < URc) R[a]=1; else R[a]=0;
      SLTi:  if (Rb < c16) R[a]=1; else R[a]=0;
      SLTiu: if (URb < uc16) R[a]=1; else R[a]=0;
      // Branch Instructions
      BEQ:   if (Ra==Rb) PCSet(`PC+c16);
      BNE:   if (Ra!=Rb) PCSet(`PC+c16);
    `endif
      // Jump Instructions
      JEQ:   if (`Z) PCSet(`PC+c24);            // JEQ Cx; if SW(=) PC  PC+Cx
      JNE:   if (!`Z) PCSet(`PC+c24);           // JNE Cx; if SW(!=) PC PC+Cx
      JLT:   if (`N) PCSet(`PC+c24);            // JLT Cx; if SW(<) PC  PC+Cx
      JGT:   if (!`N&&!`Z) PCSet(`PC+c24);      // JGT Cx; if SW(>) PC  PC+Cx
      JLE:   if (`N || `Z) PCSet(`PC+c24);      // JLE Cx; if SW(<=) PC PC+Cx    
      JGE:   if (!`N || `Z) PCSet(`PC+c24);     // JGE Cx; if SW(>=) PC PC+Cx
      JMP:   `PC = `PC+c24;                     // JMP Cx; PC <= PC+Cx
      JALR:  begin retValSet(a, `PC); PCSet(Rb); end    // JALR Ra,Rb; Ra<=PC; PC<=Rb
      BAL:   begin `LR = `PC; `PC = `PC+c24; end // BAL Cx; LR<=PC; PC<=PC+Cx
      JSUB:  begin retValSet(14, `PC); PCSet(`PC+c24); end // JSUB Cx; LR<=PC; PC<=PC+Cx
      RET:   begin PCSet(Ra); end               // RET; PC <= Ra
      default : 
        $display("%4dns %8x : OP code %8x not support", $stime, pc0, op);
      endcase
      if (`IE && `I && (`I0E && `I0 || `I1E && `I1 || `I2E && `I2)) begin
        `EPC = `PC;
        next_state = Fetch;
        inExe = 0;
      end else
        next_state = MemAccess;
    end
    MemAccess: begin
      case (op)
      ST, SB, SH  :
        memWriteEnd();                // write memory complete
      endcase
      next_state = WriteBack;
    end
    WriteBack: begin // Read/Write finish, close memory
      case (op)
      LB, LBu  :
        memReadEnd(R[a]);        //read memory complete
      LH, LHu  :
        memReadEnd(R[a]);
      LD  : begin
        memReadEnd(R[a]);
        if (`D)
          $display("%4dns %8x : %8x m[%-04x+%-04x]=%8x  SW=%8x", $stime, pc0, 
                   ir, R[b], c16, R[a], `SW);
      end
      endcase
      case (op)
      LB  : begin 
        if (R[a] > 8'h7f) R[a]=R[a]|32'hffffff80;
      end
      LH  : begin 
        if (R[a] > 16'h7fff) R[a]=R[a]|32'hffff8000;
      end
      endcase
      case (op)
      MULT, MULTu, DIV, DIVu, MTHI, MTLO :
        if (`D)
          $display("%4dns %8x : %8x HI=%8x LO=%8x SW=%8x", $stime, pc0, ir, HI, 
                   LO, `SW);
      ST : begin
        if (`D)
          $display("%4dns %8x : %8x m[%-04x+%-04x]=%8x  SW=%8x", $stime, pc0, 
                   ir, R[b], c16, R[a], `SW);
        if (R[b]+c16 == `IOADDR) begin
          outw(R[a]);
        end
      end
      SB : begin
        if (`D)
          $display("%4dns %8x : %8x m[%-04x+%-04x]=%c  SW=%8x, R[a]=%8x", 
                   $stime, pc0, ir, R[b], c16, R[a][7:0], `SW, R[a]);
        if (R[b]+c16 == `IOADDR) begin
          if (`LE)
            outc(R[a][7:0]);
          else
            outc(R[a][7:0]);
        end
      end
      MFC0, MTC0 :
        if (`D)
          $display("%4dns %8x : %8x R[%02d]=%-8x  C0R[%02d]=%-8x SW=%8x", 
                   $stime, pc0, ir, a, R[a], a, C0R[a], `SW);
      C0MOV :
        if (`D)
          $display("%4dns %8x : %8x C0R[%02d]=%-8x C0R[%02d]=%-8x SW=%8x", 
                   $stime, pc0, ir, a, C0R[a], b, C0R[b], `SW);
      default :
        if (`D) // Display the written register content
          $display("%4dns %8x : %8x R[%02d]=%-8x SW=%8x", $stime, pc0, ir, 
                   a, R[a], `SW);
      endcase
      if (`PC < 0) begin
        $display("total cpu cycles = %-d", cycles);
        $display("RET to PC < 0, finished!");
        $finish;
      end
      next_state = Fetch;
    end
    endcase
  end endtask

  always @(posedge clock) begin
    if (inExe == 0 && (state == Fetch) && (`IE && `I) && (`I0E && `I0)) begin
    // software int
      `M = `IRQ;
      taskInterrupt(`IRQ);
      m_en = 0;
      state = Fetch;
    end else if (inExe == 0 && (state == Fetch) && (`IE && `I) && 
                 ((`I1E && `I1) || (`I2E && `I2)) ) begin
      `M = `IRQ;
      taskInterrupt(`IRQ);
      m_en = 0;
      state = Fetch;
    end else if (inExe == 0 && itype == `RESET) begin
    // Condition itype == `RESET must after the other `IE condition
      taskInterrupt(`RESET);
      `M = `RESET;
      state = Fetch;
    end else begin
    `ifdef TRACE
      `D = 1; // Trace register content at beginning
    `endif
      taskExecute();
      state = next_state;
    end
    pc = `PC;
    cycles = cycles + 1;
  end
endmodule

module memory0(input clock, reset, en, rw, input [1:0] m_size, 
               input [31:0] abus, dbus_in, output [31:0] dbus_out, 
               output cfg);
  reg [31:0] mconfig [0:0];
  reg [7:0] m [0:`MEMSIZE-1];
  reg [31:0] data;

  integer i;

  `define LE  mconfig[0][0:0]   // Endian bit, Big Endian:0, Little Endian:1

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
    end else 
      data = 32'hZZZZZZZZ;
  end
  assign dbus_out = data;
  assign cfg = mconfig[0][0:0];
endmodule

module main;
  reg clock;
  reg [2:0] itype;
  wire [2:0] tick;
  wire [31:0] pc, ir, mar, mdr, dbus;
  wire m_en, m_rw;
  wire [1:0] m_size;
  wire cfg;

  cpu0 cpu(.clock(clock), .itype(itype), .pc(pc), .tick(tick), .ir(ir),
  .mar(mar), .mdr(mdr), .dbus(dbus), .m_en(m_en), .m_rw(m_rw), .m_size(m_size),
  .cfg(cfg));

  memory0 mem(.clock(clock), .reset(reset), .en(m_en), .rw(m_rw), 
  .m_size(m_size), .abus(mar), .dbus_in(mdr), .dbus_out(dbus), .cfg(cfg));

  initial
  begin
    clock = 0;
    itype = `RESET;
    #300000000 $finish;
  end

  always #10 clock=clock+1;

endmodule
