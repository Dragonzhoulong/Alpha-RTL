`timescale 1ns/1ps
module radix2_div(
    input  wire        clk,
    input  wire        rst,
    input  wire [7:0]  dividend,
    input  wire [7:0]  divisor,
    input  wire        sign,
    input  wire        opn_valid,
    output reg         res_valid,
    input  wire        res_ready,
    output reg  [15:0] result
);
    // Fully unrolled 8-stage combinational divider. Trades area for latency:
    // one clock from opn_valid to res_valid (vs. 8 clocks for iterative styles).
    wire [7:0] abs_dividend = (sign & dividend[7]) ? (~dividend + 8'd1) : dividend;
    wire [7:0] abs_divisor  = (sign & divisor[7])  ? (~divisor  + 8'd1) : divisor;
    wire       qsgn         = sign & (dividend[7] ^ divisor[7]);
    wire       rsgn         = sign & dividend[7];

    // 9-iteration chain: lvl[0] = {R=0, Q=|dividend|}; lvl[8] = final {R,Q}.
    wire [15:0] lvl [0:8];
    assign lvl[0] = {8'd0, abs_dividend};

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : stage
            wire [15:0] sh   = {lvl[i][14:0], 1'b0};
            wire [8:0]  sub  = {1'b0, sh[15:8]} - {1'b0, abs_divisor};
            wire        ge   = ~sub[8];
            assign lvl[i+1] = {ge ? sub[7:0] : sh[15:8], sh[7:1], ge};
        end
    endgenerate

    wire [7:0] rem_u  = lvl[8][15:8];
    wire [7:0] quot_u = lvl[8][ 7:0];
    wire [7:0] rem_c  = rsgn ? (~rem_u  + 8'd1) : rem_u;
    wire [7:0] quot_c = qsgn ? (~quot_u + 8'd1) : quot_u;

    always @(posedge clk) begin
        if (rst) begin
            res_valid <= 1'b0;
            result    <= 16'd0;
        end else begin
            if (res_valid && res_ready) res_valid <= 1'b0;
            if (!res_valid && opn_valid) begin
                result    <= {rem_c, quot_c};
                res_valid <= 1'b1;
            end
        end
    end
endmodule
