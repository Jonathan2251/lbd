`define FLASHADDR 'hA0000

`ifdef DLINKER
    end else if (abus >= `FLASHADDR && abus <= `FLASHADDR+`MEMSIZE-4) begin
      fabus = abus-`FLASHADDR;
      if (en == 1 && rw == 0) begin // r_w==0:write
        data = dbus_in;
        case (m_size)
        `BYTE:  {flash[fabus]} = dbus_in[7:0];
        `INT16: {flash[fabus], flash[fabus+1] } = dbus_in[15:0];
        `INT24: {flash[fabus], flash[fabus+1], flash[fabus+2]} = dbus_in[24:0];
        `INT32: {flash[fabus], flash[fabus+1], flash[fabus+2], flash[fabus+3]} 
                = dbus_in;
        endcase
      end else if (en == 1 && rw == 1) begin// r_w==1:read
        case (m_size)
        `BYTE:  data = {8'h00  , 8'h00,   8'h00,   flash[fabus]};
        `INT16: data = {8'h00  , 8'h00,   flash[fabus], flash[fabus+1]};
        `INT24: data = {8'h00  , flash[fabus], flash[fabus+1], flash[fabus+2]};
        `INT32: data = {flash[fabus], flash[fabus+1], flash[fabus+2], 
                       flash[fabus+3]};
        endcase
      end else
        data = 32'hZZZZZZZZ;
`endif

