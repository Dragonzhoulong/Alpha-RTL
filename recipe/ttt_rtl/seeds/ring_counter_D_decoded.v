module ring_counter (
    input  wire       clk,
    input  wire       reset,
    output reg  [7:0] out
);
    // Pointer-decoded variant: store a 3-bit index (~3 FFs instead of 8) and
    // one-hot-decode it combinationally. Trades FFs for a small decoder.
    reg [2:0] idx;

    always @(posedge clk or posedge reset) begin
        if (reset) idx <= 3'd0;
        else       idx <= idx + 3'd1;
    end

    always @* begin
        out = 8'b0000_0001 << idx;
    end
endmodule
