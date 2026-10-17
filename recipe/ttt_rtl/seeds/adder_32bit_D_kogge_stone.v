// Seed D — 32-bit Kogge-Stone parallel-prefix adder.
// log2(32)=5 prefix stages; all positions get their prefix (G,P) in parallel.
// Shortest logical depth of the four seeds.
module adder_32bit(A, B, S, C32);
    input  [31:0] A;
    input  [31:0] B;
    output [31:0] S;
    output        C32;

    wire [31:0] p0 = A ^ B;
    wire [31:0] g0 = A & B;

    wire [31:0] g1, p1, g2, p2, g3, p3, g4, p4, g5, p5;

    genvar i;
    // Stage 1: combine with neighbour 1 away
    generate for (i = 0; i < 32; i = i + 1) begin : st1
        if (i < 1) begin
            assign g1[i] = g0[i];
            assign p1[i] = p0[i];
        end else begin
            assign g1[i] = g0[i] | (p0[i] & g0[i-1]);
            assign p1[i] = p0[i] & p0[i-1];
        end
    end endgenerate

    // Stage 2: span 2
    generate for (i = 0; i < 32; i = i + 1) begin : st2
        if (i < 2) begin
            assign g2[i] = g1[i];
            assign p2[i] = p1[i];
        end else begin
            assign g2[i] = g1[i] | (p1[i] & g1[i-2]);
            assign p2[i] = p1[i] & p1[i-2];
        end
    end endgenerate

    // Stage 3: span 4
    generate for (i = 0; i < 32; i = i + 1) begin : st3
        if (i < 4) begin
            assign g3[i] = g2[i];
            assign p3[i] = p2[i];
        end else begin
            assign g3[i] = g2[i] | (p2[i] & g2[i-4]);
            assign p3[i] = p2[i] & p2[i-4];
        end
    end endgenerate

    // Stage 4: span 8
    generate for (i = 0; i < 32; i = i + 1) begin : st4
        if (i < 8) begin
            assign g4[i] = g3[i];
            assign p4[i] = p3[i];
        end else begin
            assign g4[i] = g3[i] | (p3[i] & g3[i-8]);
            assign p4[i] = p3[i] & p3[i-8];
        end
    end endgenerate

    // Stage 5: span 16
    generate for (i = 0; i < 32; i = i + 1) begin : st5
        if (i < 16) begin
            assign g5[i] = g4[i];
            assign p5[i] = p4[i];
        end else begin
            assign g5[i] = g4[i] | (p4[i] & g4[i-16]);
            assign p5[i] = p4[i] & p4[i-16];
        end
    end endgenerate

    // Carry into bit i is the prefix-G of bits 0..i-1; cin assumed 0.
    wire [31:0] c;
    assign c[0] = 1'b0;
    generate for (i = 1; i < 32; i = i + 1) begin : carries
        assign c[i] = g5[i-1];
    end endgenerate

    assign S   = p0 ^ c;
    assign C32 = g5[31];
endmodule
