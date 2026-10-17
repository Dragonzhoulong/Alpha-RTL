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
    // Single 16-bit shift register holds {R[15:8], Q[7:0]}.
    reg [15:0] sr;
    reg [7:0]  divisor_r;
    reg [3:0]  step;
    reg        busy;
    reg        q_sign;
    reg        r_sign;

    // Shift SR left by 1, then trial-subtract upper byte against |divisor|.
    wire [15:0] shifted = {sr[14:0], 1'b0};
    wire [8:0]  trial   = {1'b0, shifted[15:8]} - {1'b0, divisor_r};
    wire        take    = ~trial[8];
    wire [15:0] sr_next = {take ? trial[7:0] : shifted[15:8],
                           shifted[7:1],
                           take};

    wire [7:0] rem_u  = sr[15:8];
    wire [7:0] quot_u = sr[7:0];
    wire [7:0] rem_s  = r_sign ? (~rem_u  + 8'd1) : rem_u;
    wire [7:0] quot_s = q_sign ? (~quot_u + 8'd1) : quot_u;
    assign result = {rem_s, quot_s};

    always @(posedge clk) begin
        if (rst) begin
            sr        <= 16'd0;
            divisor_r <= 8'd0;
            step      <= 4'd0;
            busy      <= 1'b0;
            q_sign    <= 1'b0;
            r_sign    <= 1'b0;
            res_valid <= 1'b0;
        end else begin
            if (res_valid && res_ready) begin
                res_valid <= 1'b0;
            end
            if (!busy && !res_valid && opn_valid) begin
                sr        <= {8'd0, (sign & dividend[7]) ? (~dividend + 8'd1) : dividend};
                divisor_r <= (sign & divisor[7]) ? (~divisor + 8'd1) : divisor;
                q_sign    <= sign & (dividend[7] ^ divisor[7]);
                r_sign    <= sign & dividend[7];
                step      <= 4'd0;
                busy      <= 1'b1;
            end else if (busy) begin
                sr   <= sr_next;
                step <= step + 4'd1;
                if (step == 4'd7) begin
                    busy      <= 1'b0;
                    res_valid <= 1'b1;
                end
            end
        end
    end
endmodule
