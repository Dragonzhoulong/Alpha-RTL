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
    output wire [15:0] result
);
    // Non-restoring division with 10-bit signed partial remainder.
    // 10-bit range [-512, 511] safely holds 2*R + 1 for |R| < D <= 255.
    reg signed [9:0] R;
    reg        [7:0] Q;
    reg        [7:0] D;
    reg        [3:0] step;   // counts 0..8 (step 8 = finalize)
    reg              busy;
    reg              sq, sr;

    wire signed [9:0] D_ext   = {2'b00, D};
    wire signed [9:0] R_shift = {R[8:0], Q[7]};
    wire signed [9:0] R_next  = R[9] ? (R_shift + D_ext) : (R_shift - D_ext);
    wire              qbit    = ~R_next[9];
    wire signed [9:0] R_fix   = R[9] ? (R + D_ext) : R;  // final restore when R<0

    wire [7:0] rem_u  = R_fix[7:0];
    wire [7:0] quot_u = Q;
    wire [7:0] rem_o  = sr ? (~rem_u  + 8'd1) : rem_u;
    wire [7:0] quot_o = sq ? (~quot_u + 8'd1) : quot_u;
    assign result = {rem_o, quot_o};

    always @(posedge clk) begin
        if (rst) begin
            R         <= 10'sd0;
            Q         <= 8'd0;
            D         <= 8'd0;
            step      <= 4'd0;
            busy      <= 1'b0;
            sq        <= 1'b0;
            sr        <= 1'b0;
            res_valid <= 1'b0;
        end else begin
            if (res_valid && res_ready) res_valid <= 1'b0;

            if (!busy && !res_valid && opn_valid) begin
                R    <= 10'sd0;
                Q    <= (sign & dividend[7]) ? (~dividend + 8'd1) : dividend;
                D    <= (sign & divisor[7])  ? (~divisor  + 8'd1) : divisor;
                sq   <= sign & (dividend[7] ^ divisor[7]);
                sr   <= sign & dividend[7];
                step <= 4'd0;
                busy <= 1'b1;
            end else if (busy) begin
                if (step == 4'd8) begin
                    busy      <= 1'b0;
                    res_valid <= 1'b1;
                end else begin
                    R    <= R_next;
                    Q    <= {Q[6:0], qbit};
                    step <= step + 4'd1;
                end
            end
        end
    end
endmodule
