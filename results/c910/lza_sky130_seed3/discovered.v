module ct_vfmau_lza_simd_half(
  addend,
  lza_result,
  lza_result_zero,
  sub_vld,
  summand
);

input   [23:0]  addend;
input           sub_vld;
input   [23:0]  summand;
output  [4 :0]  lza_result;
output          lza_result_zero;

wire    [23:0]  carry_d;
wire    [23:0]  carry_g;
wire    [23:0]  carry_p;
wire    [23:0]  lza_precod;

// Pre-encoding stage
assign carry_p = summand ^ addend;
assign carry_g = summand & addend;
assign carry_d = ~(summand | addend);

assign lza_precod[0] =
     carry_p[1] && (carry_g[0] && sub_vld || carry_d[0])
 || !carry_p[1] && (carry_d[0] && sub_vld || carry_g[0]);

assign lza_precod[23] =
     sub_vld && (carry_g[23] && !carry_d[22] || carry_d[23] && !carry_g[22])
 || !sub_vld && (carry_d[23] && !carry_d[22] || !carry_d[23]);

assign lza_precod[22:1] =
    carry_p[23:2] & (carry_g[22:1] & ~carry_d[21:0] | carry_d[22:1] & ~carry_g[21:0])
 | ~carry_p[23:2] & (carry_g[22:1] & ~carry_g[21:0] | carry_d[22:1] & ~carry_d[21:0]);

// Priority encoding stage

// Step 1: Compress bits 23:20 using 4:2 compressor
wire lza_vld_1;
wire lza_p0_1, lza_p1_1;
ct_vfmau_lza_42 lza_42_1 (
  .lza_p0(lza_p0_1),
  .lza_p1(lza_p1_1),
  .lza_precod(lza_precod[23:20]),
  .lza_vld(lza_vld_1)
);

// Step 2: Compress bits 19:16 using 4:2 compressor
wire lza_vld_2;
wire lza_p0_2, lza_p1_2;
ct_vfmau_lza_42 lza_42_2 (
  .lza_p0(lza_p0_2),
  .lza_p1(lza_p1_2),
  .lza_precod(lza_precod[19:16]),
  .lza_vld(lza_vld_2)
);

// Step 3: Compress bits 15:12 using 4:2 compressor
wire lza_vld_3;
wire lza_p0_3, lza_p1_3;
ct_vfmau_lza_42 lza_42_3 (
  .lza_p0(lza_p0_3),
  .lza_p1(lza_p1_3),
  .lza_precod(lza_precod[15:12]),
  .lza_vld(lza_vld_3)
);

// Step 4: Compress bits 11:8 using 4:2 compressor
wire lza_vld_4;
wire lza_p0_4, lza_p1_4;
ct_vfmau_lza_42 lza_42_4 (
  .lza_p0(lza_p0_4),
  .lza_p1(lza_p1_4),
  .lza_precod(lza_precod[11:8]),
  .lza_vld(lza_vld_4)
);

// Step 5: Compress bits 7:4 using 4:2 compressor
wire lza_vld_5;
wire lza_p0_5, lza_p1_5;
ct_vfmau_lza_42 lza_42_5 (
  .lza_p0(lza_p0_5),
  .lza_p1(lza_p1_5),
  .lza_precod(lza_precod[7:4]),
  .lza_vld(lza_vld_5)
);

// Step 6: Compress bits 3:0 using 4:2 compressor
wire lza_vld_6;
wire lza_p0_6, lza_p1_6;
ct_vfmau_lza_42 lza_42_6 (
  .lza_p0(lza_p0_6),
  .lza_p1(lza_p1_6),
  .lza_precod(lza_precod[3:0]),
  .lza_vld(lza_vld_6)
);

// Step 7: Combine results from the first 4 compressors (bits 23:20)
wire [2:0] lza_enc_1 = {lza_p1_1, lza_p0_1};
wire [2:0] lza_enc_2 = {lza_p1_2, lza_p0_2};
wire [2:0] lza_enc_3 = {lza_p1_3, lza_p0_3};
wire [2:0] lza_enc_4 = {lza_p1_4, lza_p0_4};

// Step 8: Combine results from bits 19:16 with lower bits
wire [4:0] lza_result_1;
wire [4:0] lza_result_2;
wire [4:0] lza_result_3;
wire [4:0] lza_result_4;
wire [4:0] lza_result_5;
wire [4:0] lza_result_6;

// Generate possible results from first compression
assign lza_result_1 = (lza_enc_1 == 3'b000) ? 5'd0 : 
                      (lza_enc_1 == 3'b001) ? 5'd1 : 
                      (lza_enc_1 == 3'b010) ? 5'd2 : 
                      (lza_enc_1 == 3'b011) ? 5'd3 : 5'd24;

assign lza_result_2 = (lza_enc_2 == 3'b000) ? 5'd4 : 
                      (lza_enc_2 == 3'b001) ? 5'd5 : 
                      (lza_enc_2 == 3'b010) ? 5'd6 : 
                      (lza_enc_2 == 3'b011) ? 5'd7 : 5'd24;

assign lza_result_3 = (lza_enc_3 == 3'b000) ? 5'd8 : 
                      (lza_enc_3 == 3'b001) ? 5'd9 : 
                      (lza_enc_3 == 3'b010) ? 5'd10 : 
                      (lza_enc_3 == 3'b011) ? 5'd11 : 5'd24;

assign lza_result_4 = (lza_enc_4 == 3'b000) ? 5'd12 : 
                      (lza_enc_4 == 3'b001) ? 5'd13 : 
                      (lza_enc_4 == 3'b010) ? 5'd14 : 
                      (lza_enc_4 == 3'b011) ? 5'd15 : 5'd24;

// Fill in results for other chunks
assign lza_result_5 = (lza_p1_5 == 1'b0 && lza_p0_5 == 1'b0) ? 5'd16 :
                      (lza_p1_5 == 1'b0 && lza_p0_5 == 1'b1) ? 5'd17 : 
                      (lza_p1_5 == 1'b1 && lza_p0_5 == 1'b0) ? 5'd18 : 
                      (lza_p1_5 == 1'b1 && lza_p0_5 == 1'b1) ? 5'd19 : 5'd24;

assign lza_result_6 = (lza_p1_6 == 1'b0 && lza_p0_6 == 1'b0) ? 5'd20 :
                      (lza_p1_6 == 1'b0 && lza_p0_6 == 1'b1) ? 5'd21 : 
                      (lza_p1_6 == 1'b1 && lza_p0_6 == 1'b0) ? 5'd22 : 
                      (lza_p1_6 == 1'b1 && lza_p0_6 == 1'b1) ? 5'd23 : 5'd24;

// Determine which result is valid
wire vld_1 = lza_vld_1;
wire vld_2 = lza_vld_2;
wire vld_3 = lza_vld_3;
wire vld_4 = lza_vld_4;
wire vld_5 = lza_vld_5;
wire vld_6 = lza_vld_6;

// Final result
wire [4:0] lza_result;
assign lza_result = (vld_1) ? lza_result_1 :
                    (vld_2) ? lza_result_2 :
                    (vld_3) ? lza_result_3 :
                    (vld_4) ? lza_result_4 :
                    (vld_5) ? lza_result_5 :
                    (vld_6) ? lza_result_6 : 5'd24;

wire lza_result_zero;
assign lza_result_zero = ~(vld_1 || vld_2 || vld_3 || vld_4 || vld_5 || vld_6);

endmodule