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
    localparam IDLE = 2'd0, CALC = 2'd1, DONE = 2'd2;

    reg [1:0] state;
    reg [3:0] cnt;
    reg [8:0] R;
    reg [7:0] Q;
    reg [7:0] div_r;
    reg       q_sign;
    reg       r_sign;

    wire [8:0] R_shift = {R[7:0], Q[7]};
    wire [8:0] sub     = R_shift - {1'b0, div_r};
    wire       ge      = ~sub[8];

    wire [7:0] rem_u  = R[7:0];
    wire [7:0] quot_u = Q;
    wire [7:0] rem_out  = r_sign ? (~rem_u  + 8'd1) : rem_u;
    wire [7:0] quot_out = q_sign ? (~quot_u + 8'd1) : quot_u;
    assign result = {rem_out, quot_out};

    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            res_valid <= 1'b0;
            cnt       <= 4'd0;
            R         <= 9'd0;
            Q         <= 8'd0;
            div_r     <= 8'd0;
            q_sign    <= 1'b0;
            r_sign    <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (opn_valid && !res_valid) begin
                        R      <= 9'd0;
                        Q      <= (sign & dividend[7]) ? (~dividend + 8'd1) : dividend;
                        div_r  <= (sign & divisor[7])  ? (~divisor  + 8'd1) : divisor;
                        q_sign <= sign & (dividend[7] ^ divisor[7]);
                        r_sign <= sign & dividend[7];
                        cnt    <= 4'd0;
                        state  <= CALC;
                    end
                end
                CALC: begin
                    R   <= ge ? sub : R_shift;
                    Q   <= {Q[6:0], ge};
                    cnt <= cnt + 4'd1;
                    if (cnt == 4'd7) state <= DONE;
                end
                DONE: begin
                    res_valid <= 1'b1;
                    if (res_valid && res_ready) begin
                        res_valid <= 1'b0;
                        state     <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
