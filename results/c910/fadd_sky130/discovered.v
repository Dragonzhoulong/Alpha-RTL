module ct_fadd_close_s0_d(
  input   [52:0]  close_adder0,
  input   [52:0]  close_adder1,
  output          close_eq,
  output          close_op_chg,
  output  [52:0]  close_sum_a_b,
  output  [52:0]  close_sum_b_a,
  output  [5 :0]  ff1_pred,
  output  [52:0]  ff1_pred_onehot
);

wire [53:0] close_sum0, close_sum1;
assign close_sum0[53:0]  = {1'b0, close_adder0[52:0]} - {1'b0, close_adder1[52:0]};
assign close_sum1[53:0]  = {1'b0, close_adder1[52:0]} - {1'b0, close_adder0[52:0]};
assign close_sum_a_b     = close_sum0[52:0];
assign close_sum_b_a     = close_sum1[52:0];
assign close_op_chg      = close_sum0[53];              // borrow: B > A
assign close_eq          = !close_sum0[53] && !close_sum1[53];  // A == B

wire [52:0] ff1_c, ff1_t, ff1_g, ff1_z, ff1_f;

assign ff1_c[52:0]    = ~close_adder1[52:0];              // complement B
assign ff1_t[52:0]    = close_adder0[52:0] ^ ff1_c[52:0]; // propagate
assign ff1_g[52:0]    = close_adder0[52:0] & ff1_c[52:0]; // generate
assign ff1_z[52:0]    = (~close_adder0[52:0]) & (~ff1_c[52:0]); // delete

// F[52]: MSB boundary (no upper neighbor)
assign ff1_f[52]   = (ff1_g[52] & ~ff1_z[51]) | (ff1_z[52] & ~ff1_g[51]);

// F[0]: LSB special case — simplifies to G[0] | Z[0]
assign ff1_f[0]    = (ff1_t[1]  & (ff1_g[0] | ff1_z[0])) |
                     (~ff1_t[1] & (ff1_z[0] | ff1_g[0]));

// F[51:1]: general case (vectorized)
assign ff1_f[51:1] = ( ff1_t[52:2] & ( (ff1_g[51:1] & ~ff1_z[50:0]) |
                                        (ff1_z[51:1] & ~ff1_g[50:0]) )) |
                     (~ff1_t[52:2] & ( (ff1_g[51:1] & ~ff1_g[50:0]) |
                                        (ff1_z[51:1] & ~ff1_z[50:0]) ));

// Fast priority encoder using a hierarchical tree structure

// Extract all bits
wire ff1_f52 = ff1_f[52];
wire ff1_f51 = ff1_f[51];
wire ff1_f50 = ff1_f[50];
wire ff1_f49 = ff1_f[49];
wire ff1_f48 = ff1_f[48];
wire ff1_f47 = ff1_f[47];
wire ff1_f46 = ff1_f[46];
wire ff1_f45 = ff1_f[45];
wire ff1_f44 = ff1_f[44];
wire ff1_f43 = ff1_f[43];
wire ff1_f42 = ff1_f[42];
wire ff1_f41 = ff1_f[41];
wire ff1_f40 = ff1_f[40];
wire ff1_f39 = ff1_f[39];
wire ff1_f38 = ff1_f[38];
wire ff1_f37 = ff1_f[37];
wire ff1_f36 = ff1_f[36];
wire ff1_f35 = ff1_f[35];
wire ff1_f34 = ff1_f[34];
wire ff1_f33 = ff1_f[33];
wire ff1_f32 = ff1_f[32];
wire ff1_f31 = ff1_f[31];
wire ff1_f30 = ff1_f[30];
wire ff1_f29 = ff1_f[29];
wire ff1_f28 = ff1_f[28];
wire ff1_f27 = ff1_f[27];
wire ff1_f26 = ff1_f[26];
wire ff1_f25 = ff1_f[25];
wire ff1_f24 = ff1_f[24];
wire ff1_f23 = ff1_f[23];
wire ff1_f22 = ff1_f[22];
wire ff1_f21 = ff1_f[21];
wire ff1_f20 = ff1_f[20];
wire ff1_f19 = ff1_f[19];
wire ff1_f18 = ff1_f[18];
wire ff1_f17 = ff1_f[17];
wire ff1_f16 = ff1_f[16];
wire ff1_f15 = ff1_f[15];
wire ff1_f14 = ff1_f[14];
wire ff1_f13 = ff1_f[13];
wire ff1_f12 = ff1_f[12];
wire ff1_f11 = ff1_f[11];
wire ff1_f10 = ff1_f[10];
wire ff1_f9 = ff1_f[9];
wire ff1_f8 = ff1_f[8];
wire ff1_f7 = ff1_f[7];
wire ff1_f6 = ff1_f[6];
wire ff1_f5 = ff1_f[5];
wire ff1_f4 = ff1_f[4];
wire ff1_f3 = ff1_f[3];
wire ff1_f2 = ff1_f[2];
wire ff1_f1 = ff1_f[1];
wire ff1_f0 = ff1_f[0];

// First level grouping: 16-bit groups
wire group0 = ff1_f52 | ff1_f51 | ff1_f50 | ff1_f49 | ff1_f48 | ff1_f47 | ff1_f46 | ff1_f45 |
              ff1_f44 | ff1_f43 | ff1_f42 | ff1_f41 | ff1_f40 | ff1_f39 | ff1_f38 | ff1_f37;
wire group1 = ff1_f36 | ff1_f35 | ff1_f34 | ff1_f33 | ff1_f32 | ff1_f31 | ff1_f30 | ff1_f29 |
              ff1_f28 | ff1_f27 | ff1_f26 | ff1_f25 | ff1_f24 | ff1_f23 | ff1_f22 | ff1_f21;
wire group2 = ff1_f20 | ff1_f19 | ff1_f18 | ff1_f17 | ff1_f16 | ff1_f15 | ff1_f14 | ff1_f13 |
              ff1_f12 | ff1_f11 | ff1_f10 | ff1_f9  | ff1_f8  | ff1_f7  | ff1_f6  | ff1_f5;
wire group3 = ff1_f4  | ff1_f3  | ff1_f2  | ff1_f1  | ff1_f0  | 1'b0 | 1'b0 | 1'b0 |
              1'b0 | 1'b0 | 1'b0 | 1'b0 | 1'b0 | 1'b0 | 1'b0 | 1'b0;

// Determine group index
wire [5:0] group_idx;
assign group_idx = (group0) ? 6'd0 :
                   (group1) ? 6'd1 :
                   (group2) ? 6'd2 :
                   (group3) ? 6'd3 : 6'd14;

// Determine exact bit position within group
wire [5:0] exact_bit;
assign exact_bit = (group0) ? (
    ff1_f52 ? 52 :
    ff1_f51 ? 51 :
    ff1_f50 ? 50 :
    ff1_f49 ? 49 :
    ff1_f48 ? 48 :
    ff1_f47 ? 47 :
    ff1_f46 ? 46 :
    ff1_f45 ? 45 :
    ff1_f44 ? 44 :
    ff1_f43 ? 43 :
    ff1_f42 ? 42 :
    ff1_f41 ? 41 :
    ff1_f40 ? 40 :
    ff1_f39 ? 39 :
    ff1_f38 ? 38 :
    ff1_f37 ? 37 :
    53
) : (group1) ? (
    ff1_f36 ? 36 :
    ff1_f35 ? 35 :
    ff1_f34 ? 34 :
    ff1_f33 ? 33 :
    ff1_f32 ? 32 :
    ff1_f31 ? 31 :
    ff1_f30 ? 30 :
    ff1_f29 ? 29 :
    ff1_f28 ? 28 :
    ff1_f27 ? 27 :
    ff1_f26 ? 26 :
    ff1_f25 ? 25 :
    ff1_f24 ? 24 :
    ff1_f23 ? 23 :
    ff1_f22 ? 22 :
    ff1_f21 ? 21 :
    53
) : (group2) ? (
    ff1_f20 ? 20 :
    ff1_f19 ? 19 :
    ff1_f18 ? 18 :
    ff1_f17 ? 17 :
    ff1_f16 ? 16 :
    ff1_f15 ? 15 :
    ff1_f14 ? 14 :
    ff1_f13 ? 13 :
    ff1_f12 ? 12 :
    ff1_f11 ? 11 :
    ff1_f10 ? 10 :
    ff1_f9  ? 9 :
    ff1_f8  ? 8 :
    ff1_f7  ? 7 :
    ff1_f6  ? 6 :
    ff1_f5  ? 5 :
    53
) : (group3) ? (
    ff1_f4  ? 4 :
    ff1_f3  ? 3 :
    ff1_f2  ? 2 :
    ff1_f1  ? 1 :
    ff1_f0  ? 0 :
    53
) : 53;

// Final prediction
assign ff1_pred = (close_eq) ? 6'd53 : (52 - exact_bit);

// Generate one-hot encoding
wire [52:0] onehot_mask;
assign onehot_mask = (close_eq) ? {53{1'b0}} : (53'd1 << (52 - ff1_pred));

assign ff1_pred_onehot = onehot_mask;
endmodule