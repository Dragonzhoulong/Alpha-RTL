module ring_counter (
    input  wire       clk,
    input  wire       reset,
    output wire [7:0] out
);
    // Rotate-left expressed as (state<<1) | (state>>7) — synthesizer usually
    // reduces this to the same 1-bit-shift LUT, but it exercises a different
    // source form than the concat-based reference.
    reg [7:0] state;

    always @(posedge clk or posedge reset) begin
        if (reset) state <= 8'h01;
        else       state <= (state << 1) | (state >> 7);
    end

    assign out = state;
endmodule
