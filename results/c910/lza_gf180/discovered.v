module ct_vfmau_lza_simd_half(
  addend,                           // input  [23:0]
  lza_result,                       // output [4 :0]
  lza_result_zero,                  // output
  sub_vld,                          // input
  summand                           // input  [23:0]
);

input   [23:0]  addend;
input           sub_vld;
input   [23:0]  summand;
output  [4 :0]  lza_result;
output          lza_result_zero;

// Carry signals
wire    [23:0]  carry_p;
wire    [23:0]  carry_g;
wire    [23:0]  carry_d;
wire    [23:0]  lza_precod;

// Pre-code bit 0 (special case — uses carry_p[1])
assign lza_precod[0] =
     carry_p[1] && (carry_g[0] && sub_vld || carry_d[0])
 || !carry_p[1] && (carry_d[0] && sub_vld || carry_g[0]);

// Pre-code bit 23 (special case — MSB boundary)
assign lza_precod[23] =
     sub_vld && (carry_g[23] && !carry_d[22] || carry_d[23] && !carry_g[22])
 || !sub_vld && (carry_d[23] && !carry_d[22] || !carry_d[23]);

// Pre-code bits [22:1] (general case — uses carry_p[i+1], carry_g/d[i], carry_g/d[i-1])
assign lza_precod[22:1] =
    carry_p[23:2] & (carry_g[22:1] & ~carry_d[21:0] | carry_d[22:1] & ~carry_g[21:0])
 | ~carry_p[23:2] & (carry_g[22:1] & ~carry_g[21:0] | carry_d[22:1] & ~carry_d[21:0]);

// Carry signals
assign carry_p[23:0] = summand[23:0] ^ addend[23:0];   // propagate
assign carry_g[23:0] = summand[23:0] & addend[23:0];   // generate
assign carry_d[23:0] = ~(summand[23:0] | addend[23:0]); // delete

// Extract lza_precod bits for compression
wire bit0 = lza_precod[0];
wire bit1 = lza_precod[1];
wire bit2 = lza_precod[2];
wire bit3 = lza_precod[3];
wire bit4 = lza_precod[4];
wire bit5 = lza_precod[5];
wire bit6 = lza_precod[6];
wire bit7 = lza_precod[7];
wire bit8 = lza_precod[8];
wire bit9 = lza_precod[9];
wire bit10 = lza_precod[10];
wire bit11 = lza_precod[11];
wire bit12 = lza_precod[12];
wire bit13 = lza_precod[13];
wire bit14 = lza_precod[14];
wire bit15 = lza_precod[15];
wire bit16 = lza_precod[16];
wire bit17 = lza_precod[17];
wire bit18 = lza_precod[18];
wire bit19 = lza_precod[19];
wire bit20 = lza_precod[20];
wire bit21 = lza_precod[21];
wire bit22 = lza_precod[22];
wire bit23 = lza_precod[23];

// Compressor stages
wire lza_vld_0_2 = bit0 | bit1 | bit2;
wire lza_p0_0_2 = !bit2 && bit1;
wire lza_p1_0_2 = !bit2 && !bit1;

wire lza_vld_3_5 = bit3 | bit4 | bit5;
wire lza_p0_3_5 = !bit5 && bit4;
wire lza_p1_3_5 = !bit5 && !bit4;

wire lza_vld_6_8 = bit6 | bit7 | bit8;
wire lza_p0_6_8 = !bit8 && bit7;
wire lza_p1_6_8 = !bit8 && !bit7;

wire lza_vld_9_11 = bit9 | bit10 | bit11;
wire lza_p0_9_11 = !bit11 && bit10;
wire lza_p1_9_11 = !bit11 && !bit10;

wire lza_vld_12_14 = bit12 | bit13 | bit14;
wire lza_p0_12_14 = !bit14 && bit13;
wire lza_p1_12_14 = !bit14 && !bit13;

wire lza_vld_15_17 = bit15 | bit16 | bit17;
wire lza_p0_15_17 = !bit17 && bit16;
wire lza_p1_15_17 = !bit17 && !bit16;

wire lza_vld_18_20 = bit18 | bit19 | bit20;
wire lza_p0_18_20 = !bit20 && bit19;
wire lza_p1_18_20 = !bit20 && !bit19;

wire lza_vld_21_23 = bit21 | bit22 | bit23;
wire lza_p0_21_23 = !bit23 && bit22;
wire lza_p1_21_23 = !bit23 && !bit22;

// Merging 2-bit results from 3-bit blocks
wire lza_vld_0_5 = lza_vld_0_2 | lza_vld_3_5;
wire lza_p0_0_5 = lza_vld_0_2 ? lza_p0_0_2 : lza_p0_3_5;
wire lza_p1_0_5 = lza_vld_0_2 ? lza_p1_0_2 : lza_p1_3_5;

wire lza_vld_6_11 = lza_vld_6_8 | lza_vld_9_11;
wire lza_p0_6_11 = lza_vld_6_8 ? lza_p0_6_8 : lza_p0_9_11;
wire lza_p1_6_11 = lza_vld_6_8 ? lza_p1_6_8 : lza_p1_9_11;

wire lza_vld_12_17 = lza_vld_12_14 | lza_vld_15_17;
wire lza_p0_12_17 = lza_vld_12_14 ? lza_p0_12_14 : lza_p0_15_17;
wire lza_p1_12_17 = lza_vld_12_14 ? lza_p1_12_14 : lza_p1_15_17;

wire lza_vld_18_23 = lza_vld_18_20 | lza_vld_21_23;
wire lza_p0_18_23 = lza_vld_18_20 ? lza_p0_18_20 : lza_p0_21_23;
wire lza_p1_18_23 = lza_vld_18_20 ? lza_p1_18_20 : lza_p1_21_23;

// Merging 2-bit results from 6-bit blocks
wire lza_vld_0_11 = lza_vld_0_5 | lza_vld_6_11;
wire lza_p0_0_11 = lza_vld_0_5 ? lza_p0_0_5 : lza_p0_6_11;
wire lza_p1_0_11 = lza_vld_0_5 ? lza_p1_0_5 : lza_p1_6_11;

wire lza_vld_12_23 = lza_vld_12_17 | lza_vld_18_23;
wire lza_p0_12_23 = lza_vld_12_17 ? lza_p0_12_17 : lza_p0_18_23;
wire lza_p1_12_23 = lza_vld_12_17 ? lza_p1_12_17 : lza_p1_18_23;

// Final compression stage
wire lza_vld_final = lza_vld_0_11 | lza_vld_12_23;
wire lza_p0 = lza_vld_0_11 ? lza_p0_0_11 : lza_p0_12_23;
wire lza_p1 = lza_vld_0_11 ? lza_p1_0_11 : lza_p1_12_23;

// Convert final result to 5-bit lza_result
wire [4:0] lza_result_val;

// Position mapping: 0→23, 1→22, 2→21, ..., 23→0
assign lza_result_val = 
     (bit23 | bit22) ? 
     (bit23 ? 5'd0 : 5'd1) :
     (bit21 | bit20) ? 
     (bit21 ? 5'd2 : 5'd3) :
     (bit19 | bit18) ? 
     (bit19 ? 5'd4 : 5'd5) :
     (bit17 | bit16) ? 
     (bit17 ? 5'd6 : 5'd7) :
     (bit15 | bit14) ? 
     (bit15 ? 5'd8 : 5'd9) :
     (bit13 | bit12) ? 
     (bit13 ? 5'd10 : 5'd11) :
     (bit11 | bit10) ? 
     (bit11 ? 5'd12 : 5'd13) :
     (bit9 | bit8) ? 
     (bit9 ? 5'd14 : 5'd15) :
     (bit7 | bit6) ? 
     (bit7 ? 5'd16 : 5'd17) :
     (bit5 | bit4) ? 
     (bit5 ? 5'd18 : 5'd19) :
     (bit3 | bit2) ? 
     (bit3 ? 5'd20 : 5'd21) :
     (bit1 | bit0) ? 
     (bit1 ? 5'd22 : 5'd23) :
     5'd24;

// lza_result_zero is 1 if no bits are set
assign lza_result_zero = ~|lza_precod[23:0];

// Assign final result
assign lza_result = lza_result_val;

endmodule