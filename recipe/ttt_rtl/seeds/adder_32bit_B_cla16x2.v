// Seed B — Hierarchical 16-bit carry-lookahead blocks.
// Matches the "adder_32bit consists of several 16-bit CLA blocks" spec.
// Uses standard OR-based carry equations (vs. the reference's non-standard
// XOR form) and a clean parameterizable carry chain inside each block.
module adder_32bit(A, B, S, C32);
    input  [31:0] A;
    input  [31:0] B;
    output [31:0] S;
    output        C32;

    wire c16;

    cla16 cla_lo (.A(A[15:0]),  .B(B[15:0]),  .cin(1'b0), .S(S[15:0]),  .cout(c16));
    cla16 cla_hi (.A(A[31:16]), .B(B[31:16]), .cin(c16),  .S(S[31:16]), .cout(C32));
endmodule

module cla16(A, B, cin, S, cout);
    input  [15:0] A;
    input  [15:0] B;
    input         cin;
    output [15:0] S;
    output        cout;

    wire [15:0] p = A ^ B;
    wire [15:0] g = A & B;
    wire [16:0] c;
    assign c[0] = cin;

    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : carry_chain
            assign c[i+1] = g[i] | (p[i] & c[i]);
        end
    endgenerate

    assign S    = p ^ c[15:0];
    assign cout = c[16];
endmodule
