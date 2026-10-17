module multi_8bit (
    input  wire [ 7:0] A,
    input  wire [ 7:0] B,
    output wire [15:0] product
);
    // Wallace-tree style: 8 AND-rows of partial products, then column-wise
    // 3:2 compression to two rows, finished by a single carry-propagate add.
    // Written as bit-level expressions to surface the carry-save structure to
    // the synthesizer (vs. a chain of `+` operators).
    wire [15:0] pp [0:7];
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : g_pp
            assign pp[i] = {8'b0, (A & {8{B[i]}})} << i;
        end
    endgenerate

    // First reduction layer: combine rows {0,1,2}, {3,4,5}, {6,7,carry slot}
    // into 6 rows (sum/carry pairs from 3:2 compressors per column).
    wire [15:0] s012, c012, s345, c345;
    assign s012 = pp[0] ^ pp[1] ^ pp[2];
    assign c012 = ((pp[0] & pp[1]) | (pp[1] & pp[2]) | (pp[0] & pp[2])) << 1;
    assign s345 = pp[3] ^ pp[4] ^ pp[5];
    assign c345 = ((pp[3] & pp[4]) | (pp[4] & pp[5]) | (pp[3] & pp[5])) << 1;

    // Second layer: 4 rows {s012, c012, s345, c345} + remaining 2 rows {pp6, pp7}
    // Compress {s012, c012, s345} → (s_a, c_a) and {c345, pp[6], pp[7]} → (s_b, c_b)
    wire [15:0] s_a, c_a, s_b, c_b;
    assign s_a = s012 ^ c012 ^ s345;
    assign c_a = ((s012 & c012) | (c012 & s345) | (s012 & s345)) << 1;
    assign s_b = c345 ^ pp[6] ^ pp[7];
    assign c_b = ((c345 & pp[6]) | (pp[6] & pp[7]) | (c345 & pp[7])) << 1;

    // Third layer: 4 → 2 with one more 3:2 + the leftover row
    wire [15:0] s_c, c_c;
    assign s_c = s_a ^ c_a ^ s_b;
    assign c_c = ((s_a & c_a) | (c_a & s_b) | (s_a & s_b)) << 1;

    // Final CPA: s_c + c_c + c_b
    assign product = s_c + c_c + c_b;
endmodule
