// 管線狀態：IDLE 或 WAIT
`define IDLE     2'b00       // 閒置中
`define WAIT_M   2'b01   // 等待回應
`define WAIT_ACK 2'b10   // 等待回應

// 寬度形態常數
`define INT32 2'b11     // 寬度 32 位元
`define INT24 2'b10     // 寬度 24 位元
`define INT16 2'b01     // 寬度 16 位元
`define BYTE  2'b00     // 寬度  8 位元
// 暫存器簡稱
`define PC   cpu.R[15]   // 程式計數器
`define LR   cpu.R[14]   // 連結暫存器
`define SP   cpu.R[13]   // 堆疊暫存器
`define SW   cpu.R[12]   // 狀態暫存器
`define IR   cpu.ir         // 指令暫存器
// 狀態暫存器旗標位元
`define N    `SW[31] // 負號旗標
`define Z    `SW[30] // 零旗標
`define C    `SW[29] // 進位旗標
`define V    `SW[28] // 溢位旗標
`define I    `SW[7]  // 硬體中斷許可
`define T    `SW[6]  // 軟體中斷許可
`define M    `SW[0]  // 模式位元

module iFetch(input clock, reset, iReady, output reg iGet, oReady, input oGet);
reg [1:0] state;
reg [31:0] pc, pc0;

    always @(posedge clock) begin
        if (reset) begin oReady=0; iGet=0; state=`IDLE; end 
        else case (state) 
            `IDLE: begin // 閒置中
                #1;
                if (iReady) begin // 輸入資料已準備好
                    #1;
                    memReadStart1(`PC, `INT32);
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
                    memReadEnd1(ir); // IR = dbus = m[PC]
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
                    case (op)
                        // 載入儲存指令
                        cpu.LD:  memReadStart2(Rb+c16, `INT32);         // 載入word;    LD Ra, [Rb+Cx];     Ra<=[Rb+ Cx]
                        cpu.ST:  memWriteStart2(Rb+c16, Ra, `INT32); // 儲存word;    ST Ra, [Rb+ Cx];     Ra=>[Rb+ Cx]
                        cpu.LDB: memReadStart2(Rb+c16, `BYTE);        // 載入byte;    LDB Ra, [Rb+ Cx];     Ra<=(byte)[Rb+ Cx]
                        cpu.STB: memWriteStart2(Rb+c16, Ra, `BYTE);     // 儲存byte;    STB Ra, [Rb+ Cx];    Ra=>(byte)[Rb+ Cx]
                        cpu.LDR: memReadStart2(Rb+Rc, `INT32);        // LD的Rc版;     LDR Ra, [Rb+Rc];    Ra<=[Rb+ Rc]
                        cpu.STR: memWriteStart2(Rb+Rc, Ra, `INT32);    // ST的Rc版;    STR Ra, [Rb+Rc];    Ra=>[Rb+ Rc]
                        cpu.LBR: memReadStart2(Rb+Rc, `BYTE);        // LDB的Rc版;    LBR Ra, [Rb+Rc];    Ra<=(byte)[Rb+ Rc]
                        cpu.SBR: memWriteStart2(Rb+Rc, Ra, `BYTE);    // STB的Rc版;    SBR Ra, [Rb+Rc];    Ra=>(byte)[Rb+ Rc]
                        // 堆疊指令    
                        cpu.PUSH:begin `SP = `SP-4; memWriteStart2(`SP, Ra, `INT32); end // 推入 word;    PUSH Ra;    SP-=4;[SP]<=Ra;
                        cpu.POP: begin memReadStart2(`SP, `INT32); `SP = `SP + 4; end    // 彈出 word;    POP Ra;     Ra=[SP];SP+=4;
                        cpu.PUSHB:begin `SP = `SP-1; memWriteStart2(`SP, Ra, `BYTE); end    // 推入 byte;    PUSHB Ra;     SP--;[SP]<=Ra;(byte)
                        cpu.POPB:begin memReadStart2(`SP, `BYTE); `SP = `SP+1; end        // 彈出 byte;    POPB Ra;     Ra<=[SP];SP++;(byte)
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

module iExec(input clock, reset, iReady, output reg iGet, oReady, input oGet);
    parameter name="iExec";
    reg [31:0] ir, pc, pc0;
    reg [7:0] op;
    reg [3:0] a, b, c;
    reg [4:0] c5;
    reg signed [31:0] c12, c16, c24, Ra, Rb, Rc; // ipc:instruction PC
    reg [1:0] state;
    reg [1:0] skip;

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
                    iGet = 1;
                    // 準備輸出資料
                    case (op)
                        cpu.LD, cpu.LDB, cpu.LDR, cpu.LBR, cpu.POP, cpu.POPB  : memReadEnd2(cpu.R[a]); // 讀取記憶體完成
                        cpu.ST, cpu.STB, cpu.STR, cpu.SBR, cpu.PUSH, cpu.PUSHB: memWriteEnd2(); // 寫入記憶體完成
                        cpu.LDI: cpu.R[a] = Rb+c16;                     // 立即載入;    LDI Ra, Rb+Cx;        Ra<=Rb + Cx
                        // 運算指令
                        cpu.CMP: begin `N=(Ra-Rb<0);`Z=(Ra-Rb==0); end // 比較;        CMP Ra, Rb;         SW=(Ra >=< Rb)
                        cpu.MOV: regSet(a, Rb);                 // 移動;            MOV Ra, Rb;         Ra<=Rb
                        cpu.ADD: regSet(a, Rb+Rc);                // 加法;            ADD Ra, Rb, Rc;     Ra<=Rb+Rc
                        cpu.SUB: regSet(a, Rb-Rc);                // 減法;            SUB Ra, Rb, Rc;     Ra<=Rb-Rc
                        cpu.MUL: regSet(a, Rb*Rc);                // 乘法;             MUL Ra, Rb, Rc;     Ra<=Rb*Rc
                        cpu.DIV: regSet(a, Rb/Rc);                // 除法;             DIV Ra, Rb, Rc;     Ra<=Rb/Rc
                        cpu.AND: regSet(a, Rb&Rc);                // 位元 AND;        AND Ra, Rb, Rc;     Ra<=Rb and Rc
                        cpu.OR:  regSet(a, Rb|Rc);                // 位元 OR;            OR Ra, Rb, Rc;         Ra<=Rb or Rc
                        cpu.XOR: regSet(a, Rb^Rc);                // 位元 XOR;        XOR Ra, Rb, Rc;     Ra<=Rb xor Rc
                        cpu.SHL: regSet(a, Rb<<c5);                // 向左移位;        SHL Ra, Rb, Cx;     Ra<=Rb << Cx
                        cpu.SHR: regSet(a, Rb>>c5);                // 向右移位;        SHR Ra, Rb, Cx;     Ra<=Rb >> Cx
                        // 跳躍指令
                        cpu.JEQ: if (`Z) begin `PC=pc+c24; skip=2; end // 跳躍 (相等);        JEQ Cx;        if SW(=) PC  PC+Cx
                        cpu.JNE: if (!`Z) begin `PC=pc+c24; skip=2; end  // 跳躍 (不相等);    JNE Cx;     if SW(!=) PC  PC+Cx
                        cpu.JLT: if (`N) begin `PC=pc+c24; skip=2; end        // 跳躍 ( < );        JLT Cx;     if SW(<) PC  PC+Cx
                        cpu.JGT: if (!`N&&!`Z) begin `PC=pc+c24; skip=2; end        // 跳躍 ( > );        JGT Cx;     if SW(>) PC  PC+Cx
                        cpu.JLE: if (`N || `Z) begin `PC=pc+c24; skip=2; end        // 跳躍 ( <= );        JLE Cx;     if SW(<=) PC  PC+Cx    
                        cpu.JGE: if (!`N || `Z) begin `PC=pc+c24; skip=2; end    // 跳躍 ( >= );        JGE Cx;     if SW(>=) PC  PC+Cx
                        cpu.JMP: begin `PC = pc+c24; skip=2; end                     // 跳躍 (無條件);    JMP Cx;     PC <= PC+Cx
                        cpu.SWI: begin `LR=pc;`PC= c24; `I = 1'b1; skip=2; end // 軟中斷;    SWI Cx;         LR <= PC; PC <= Cx; INT<=1
                        cpu.CALL:begin `LR=pc;`PC=pc + c24; skip=2; end // 跳到副程式;    CALL Cx;     LR<=PC; PC<=PC+Cx
                        cpu.RET: begin `PC=`LR; skip=2;                 // 返回;            RET;         PC <= LR
                            if (`PC < 0) begin
                                $display("RET to PC < 0, finished!");
                                $finish;
                            end                        
                        end
                        cpu.IRET:begin `PC=`LR;`I = 1'b0; skip=2; end    // 中斷返回;        IRET;         PC <= LR; INT<=0
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

module cpu(input clock, reset, output [31:0] ir, pc,
           output [31:0] mar1, mdr1, inout [31:0] dbus1, output reg m_en1, m_rw1, input m_ack1, output reg [1:0] m_size1,
           output [31:0] mar2, mdr2, inout [31:0] dbus2, output reg m_en2, m_rw2, input m_ack2, output reg [1:0] m_size2
           ); // cpu0 是由 if1, id1, ie1, iw1 四根管子 (pipe) 連接後形成的管線           
   // 管線相關參數
    wire ifiGet, idiGet, ieiGet, iwiGet; // pipe 輸入是否準備好了
    wire ifoReady, idoReady, ieoReady, iwoReady; // pipe 輸出是否準備好了
    parameter iReady = 1'b1, oGet=1'b1; // pipeline 的整體輸入輸出是否準備好了 (隨時都準備好，這樣才會不斷驅動)。
    // 暫存器與欄位
    reg [31:0] mar1, mdr1, mar2, mdr2;
    reg signed [31:0] R [0:15];

    // 指令編碼表
    parameter [7:0] LD=8'h00,ST=8'h01,LDB=8'h02,STB=8'h03,LDR=8'h04,STR=8'h05,
    LBR=8'h06,SBR=8'h07,LDI=8'h08,CMP=8'h10,MOV=8'h12,ADD=8'h13,SUB=8'h14,
    MUL=8'h15,DIV=8'h16,AND=8'h18,OR=8'h19,XOR=8'h1A,ROL=8'h1C,ROR=8'h1D,
    SHL=8'h1E,SHR=8'h1F,JEQ=8'h20,JNE=8'h21,JLT=8'h22,JGT=8'h23,JLE=8'h24,
    JGE=8'h25,JMP=8'h26,SWI=8'h2A,CALL=8'h2B,RET=8'h2C,IRET=8'h2D,
    PUSH=8'h30,POP=8'h31,PUSHB=8'h32,POPB=8'h33;

    task memReadStart1(input [31:0] addr, input [1:0] size); begin // 讀取記憶體 Word
       mar1 = addr;     // read(m[addr])
       m_rw1 = 1;     // 讀取模式：read 
       m_en1 = 1;     // 啟動讀取
       m_size1 = size;
    end    endtask

    task memReadEnd1(output [31:0] data); begin // 讀取記憶體完成，取得資料
       mdr1 = dbus1; // 取得記憶體傳回的 dbus = m[addr]
       data = dbus1; // 傳回資料
       m_en1 = 0; // 讀取完畢
    end    endtask

    // 寫入記憶體 -- addr:寫入位址, data:寫入資料
    task memWriteStart1(input [31:0] addr, input [31:0] data, input [1:0] size); begin 
       mar1 = addr;    // write(m[addr], data)
       mdr1 = data;
       m_rw1 = 0;    // 寫入模式：write
       m_en1 = 1;     // 啟動寫入
       m_size1  = size;
    end    endtask

    task memWriteEnd1; begin // 寫入記憶體完成
       m_en1 = 0; // 寫入完畢
    end endtask

    task memReadStart2(input [31:0] addr, input [1:0] size); begin // 讀取記憶體 Word
       mar2 = addr;     // read(m[addr])
       m_rw2 = 1;     // 讀取模式：read 
       m_en2 = 1;     // 啟動讀取
       m_size2 = size;
    end    endtask

    task memReadEnd2(output [31:0] data); begin // 讀取記憶體完成，取得資料
       mdr2 = dbus2; // 取得記憶體傳回的 dbus = m[addr]
       data = dbus2; // 傳回資料
       m_en2 = 0; // 讀取完畢
    end    endtask

    // 寫入記憶體 -- addr:寫入位址, data:寫入資料
    task memWriteStart2(input [31:0] addr, input [31:0] data, input [1:0] size); begin 
       mar2 = addr;    // write(m[addr], data)
       mdr2 = data;
       m_rw2 = 0;    // 寫入模式：write
       m_en2 = 1;     // 啟動寫入
       m_size2  = size;
    end    endtask

    task memWriteEnd2; begin // 寫入記憶體完成
       m_en2 = 0; // 寫入完畢
    end endtask

    task regSet(input [3:0] i, input [31:0] data); begin
        if (i!=0) R[i] = data;
    end endtask

    iFetch      if1(clock, reset, iReady,   ifiGet, ifoReady, idiGet); // 管子：
    iDecode     id1(clock, reset, ifoReady, idiGet, idoReady, ieiGet); // 管子：
    iExec       ie1(clock, reset, idoReady, ieiGet, ieoReady, oGet);   // 管子：

    always @(posedge clock) begin
        if (reset) begin `PC = 0; R[0] = 0; `SW = 0; `LR = -1;  end
    end
endmodule

module memory(input clock, reset, en1, en2, rw1, rw2, output reg ack1, ack2, input [1:0] size1, size2, 
              input [31:0] abus1, abus2, dbus_in1, dbus_in2, output [31:0] dbus_out1, dbus_out2);
reg [7:0] m [0:258];
reg [31:0] data1, data2;

integer i;
initial begin
    $readmemh("cpu0p.hex", m);
    for (i=0; i < 255; i=i+4) begin
       $display("%8x: %8x", i, {m[i], m[i+1], m[i+2], m[i+3]});
    end
end

    always @(*) 
    begin
        if (en1) begin
            ack1 = 0;
            #30;
            if (abus1 >=0 && abus1 <= 255) begin
                if (rw1 == 0) begin // r_w==0:write
                    data1 = dbus_in1;
                    case (size1)
                        `BYTE:  {m[abus1]} = dbus_in1[7:0];
                        `INT16: {m[abus1], m[abus1+1] } = dbus_in1[15:0];
                        `INT24: {m[abus1], m[abus1+1], m[abus1+2]} = dbus_in1[24:0];
                        `INT32: {m[abus1], m[abus1+1], m[abus1+2], m[abus1+3]} = dbus_in1;
                    endcase
                end else begin// rw == 1:read
                    case (size1)
                        `BYTE:  data1 = {8'h00  , 8'h00,   8'h00,   m[abus1]      };
                        `INT16: data1 = {8'h00  , 8'h00,   m[abus1], m[abus1+1]    };
                        `INT24: data1 = {8'h00  , m[abus1], m[abus1+1], m[abus1+2]  };
                        `INT32: data1 = {m[abus1], m[abus1+1], m[abus1+2], m[abus1+3]};
                    endcase
                end
            end
            ack1 = 1;
        end else begin
            data1 = 32'hZZZZZZZZ;
            ack1 = 0;
        end
    end
    assign dbus_out1 = data1;

    always @(*) 
    begin
        if (en2) begin
            ack2 = 0;
            #30;
            if (abus2 >=0 && abus2 <= 255) begin
                if (rw2 == 0) begin // r_w==0:write
                    data2 = dbus_in2;
                    case (size2)
                        `BYTE:  {m[abus2]} = dbus_in2[7:0];
                        `INT16: {m[abus2], m[abus2+1] } = dbus_in2[15:0];
                        `INT24: {m[abus2], m[abus2+1], m[abus2+2]} = dbus_in2[24:0];
                        `INT32: {m[abus2], m[abus2+1], m[abus2+2], m[abus2+3]} = dbus_in2;
                    endcase
                end else begin// rw == 1:read
                    case (size2)
                        `BYTE:  data2 = {8'h00  , 8'h00,   8'h00,   m[abus2]      };
                        `INT16: data2 = {8'h00  , 8'h00,   m[abus2], m[abus2+1]    };
                        `INT24: data2 = {8'h00  , m[abus2], m[abus2+1], m[abus2+2]  };
                        `INT32: data2 = {m[abus2], m[abus2+1], m[abus2+2], m[abus2+3]};
                    endcase
                end
            end
            ack2 = 1;
        end else begin
            data2 = 32'hZZZZZZZZ;
            ack2 = 0;
        end
    end
    assign dbus_out2 = data2;
endmodule

module main;
    reg clock, reset;
    wire [31:0] pc, ir;
    wire [31:0] mar1, mar2, mdr1, mdr2, dbus1, dbus2;
    wire m_en1, m_en2, m_rw1, m_rw2, m_ack1, m_ack2;
    wire [1:0] m_size1, m_size2;

    cpu cpu0(.clock(clock), .reset(reset), .pc(pc), .ir(ir),
        .mar1(mar1), .mdr1(mdr1), .dbus1(dbus1), .m_en1(m_en1), .m_rw1(m_rw1), .m_size1(m_size1), .m_ack1(m_ack1),
        .mar2(mar2), .mdr2(mdr2), .dbus2(dbus2), .m_en2(m_en2), .m_rw2(m_rw2), .m_size2(m_size2), .m_ack2(m_ack2));

    memory memory0(.clock(clock), .reset(reset), 
      .en1(m_en1), .rw1(m_rw1), .ack1(m_ack1), .size1(m_size1), .abus1(mar1), .dbus_in1(mdr1), .dbus_out1(dbus1),
      .en2(m_en2), .rw2(m_rw2), .ack2(m_ack2), .size2(m_size2), .abus2(mar2), .dbus_in2(mdr2), .dbus_out2(dbus2));

    initial
    begin
        clock = 0;
        reset = 1;
        #50 reset = 0;
        #10000 $finish;
    end

    always #10 clock=clock+1;

endmodule
