// Seed C — 16/16 carry-select adder.
// Computes both possible high halves speculatively (cin=0 and cin=1) and
// muxes based on the low half's carry-out. Shorter critical path at the cost
// of a duplicated upper adder.
module adder_32bit(A, B, S, C32);
    input  [31:0] A;
    input  [31:0] B;
    output [31:0] S;
    output        C32;

    wire [16:0] sum_lo  = {1'b0, A[15:0]}  + {1'b0, B[15:0]};
    wire [16:0] sum_hi0 = {1'b0, A[31:16]} + {1'b0, B[31:16]};
    wire [16:0] sum_hi1 = {1'b0, A[31:16]} + {1'b0, B[31:16]} + 17'd1;

    wire        c16    = sum_lo[16];
    wire [16:0] sum_hi = c16 ? sum_hi1 : sum_hi0;

    assign S   = {sum_hi[15:0], sum_lo[15:0]};
    assign C32 = sum_hi[16];
endmodule
