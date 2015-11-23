`define IDLE 1'b0     // 閒置中
`define WAIT 2'b1     // 等待回應

// pipe : 單一根管子 (iReady, iMsg, iGet) 為輸入部分，(oReady, oMsg, oGet) 為輸出部分, id 為管線代號
module pipe(input clock, reset, iReady, input [7:0] iMsg, output iGet, 
            output oReady, output [7:0] oMsg, input oGet, input [3:0] id);
reg oReady, iGet, state;
reg [7:0] iMsgReg, oMsg;
    always @(posedge clock) begin
        if (reset) begin oReady=0; iGet=0; state=`IDLE; end 
        else case (state) 
            `IDLE: begin // 閒置中
                if (iReady) begin // 輸入資料已準備好
                    iMsgReg <= iMsg; // 儲存輸入資料
                    iGet <= 1;
                    state <= `WAIT; // 進入等帶狀態
                    $display("%-8d:p%x iMsg=%d, iGet", $stime, id, iMsg);
                    #1 oMsg <= iMsg + 1; // 設定輸出資料
                    #2;
                    $display("%-8d:p%x oMsg=%d, oReady", $stime, id, oMsg);
                end else 
                    $display("%-8d:p%x IDLE, not iReady", $stime, id);
            end
            `WAIT:begin // 等待回應 (資料被取走)
                if (oGet) begin // 資料被取走了
                    oReady <= 0; // 下一筆輸出資料尚未準備好。
                    state <= `IDLE; // 回到閒置狀態，準備取得下一筆輸入資料
                    #2;
                    $display("%-8d:p%x oGet", $stime, id); // 顯示資料已經取走
                end else begin
                    oReady <= 1; // 這裏有修改 *****
                     $display("%-8d:p%x WAIT, oReady, not oGet", $stime, id);
                end
                iGet <= 0;  // 下一筆輸入資料尚未準備好。
            end
        endcase
    end
endmodule

module pipeline; // pipeline : 多根管子連接後形成的管線
reg clock, reset; // 時脈
wire [7:0] p1Msg, p2Msg, p3Msg; // pipe 傳遞的訊息
wire p1iGet, p2iGet, p3iGet; // pipe 輸入是否準備好了
wire p1oReady, p2oReady, p3oReady; // pipe 輸出是否準備好了
reg [7:0] iMsg=0; // pipeline 的整體輸入訊息
parameter iReady = 1'b1, oGet=1'b1; // pipeline 的整體輸入輸出是否準備好了 (隨時都準備好，這樣才會不斷驅動)。

pipe p1(clock, reset, iReady,   iMsg, p1iGet,  p1oReady, p1Msg, p2iGet,1); // 第一根管子
pipe p2(clock, reset, p1oReady, p1Msg, p2iGet, p2oReady, p2Msg, p3iGet,2); // 第二根管子
pipe p3(clock, reset, p2oReady, p2Msg, p3iGet, p3oReady, p3Msg, oGet,     3); // 第三根管子

initial begin
  reset = 1;
  clock = 0;
  #100 reset = 0;
  #100 iMsg = 10; // pipelie 的輸入資料改為 10
  #100 iMsg = 20; // pipelie 的輸入資料改為 20
  #200 $finish;
end

always #10 begin
  clock=clock+1;
end

endmodule

