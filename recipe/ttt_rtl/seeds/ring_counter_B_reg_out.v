module ring_counter (
    input  wire       clk,
    input  wire       reset,
    output reg  [7:0] out
);
    // Direct register-on-port variant: eliminate the intermediate state reg
    // so yosys sees one 8-bit FF cluster instead of state+driver.
    always @(posedge clk or posedge reset) begin
        if (reset) out <= 8'b0000_0001;
        else       out <= {out[6:0], out[7]};
    end
endmodule
