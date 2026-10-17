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

assign carry_p[23:0] = summand[23:0] ^ addend[23:0];   // propagate
assign carry_g[23:0] = summand[23:0] & addend[23:0];   // generate
assign carry_d[23:0] = ~(summand[23:0] | addend[23:0]); // delete

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

// Group bits into 6 groups of 4 bits each
wire [3:0] lza_group0 = {lza_precod[23], lza_precod[22], lza_precod[21], lza_precod[20]};
wire [3:0] lza_group1 = {lza_precod[19], lza_precod[18], lza_precod[17], lza_precod[16]};
wire [3:0] lza_group2 = {lza_precod[15], lza_precod[14], lza_precod[13], lza_precod[12]};
wire [3:0] lza_group3 = {lza_precod[11], lza_precod[10], lza_precod[9], lza_precod[8]};
wire [3:0] lza_group4 = {lza_precod[7], lza_precod[6], lza_precod[5], lza_precod[4]};
wire [3:0] lza_group5 = {lza_precod[3], lza_precod[2], lza_precod[1], lza_precod[0]};

// Use 4:2 compressors for each group
wire lza0_p0, lza0_p1, lza0_vld;
wire lza1_p0, lza1_p1, lza1_vld;
wire lza2_p0, lza2_p1, lza2_vld;
wire lza3_p0, lza3_p1, lza3_vld;
wire lza4_p0, lza4_p1, lza4_vld;
wire lza5_p0, lza5_p1, lza5_vld;

ct_vfmau_lza_42 lza0_inst (
  .lza_precod(lza_group0),
  .lza_p0(lza0_p0),
  .lza_p1(lza0_p1),
  .lza_vld(lza0_vld)
);

ct_vfmau_lza_42 lza1_inst (
  .lza_precod(lza_group1),
  .lza_p0(lza1_p0),
  .lza_p1(lza1_p1),
  .lza_vld(lza1_vld)
);

ct_vfmau_lza_42 lza2_inst (
  .lza_precod(lza_group2),
  .lza_p0(lza2_p0),
  .lza_p1(lza2_p1),
  .lza_vld(lza2_vld)
);

ct_vfmau_lza_42 lza3_inst (
  .lza_precod(lza_group3),
  .lza_p0(lza3_p0),
  .lza_p1(lza3_p1),
  .lza_vld(lza3_vld)
);

ct_vfmau_lza_42 lza4_inst (
  .lza_precod(lza_group4),
  .lza_p0(lza4_p0),
  .lza_p1(lza4_p1),
  .lza_vld(lza4_vld)
);

ct_vfmau_lza_42 lza5_inst (
  .lza_precod(lza_group5),
  .lza_p0(lza5_p0),
  .lza_p1(lza5_p1),
  .lza_vld(lza5_vld)
);

// Convert 2-bit position code to 5-bit result for each group
wire [4:0] group_result0, group_result1, group_result2, group_result3, group_result4, group_result5;

assign group_result0 = (lza0_vld) ? ({lza0_p1, lza0_p0} + 0) : 5'd24;
assign group_result1 = (lza1_vld) ? ({lza1_p1, lza1_p0} + 4) : 5'd24;
assign group_result2 = (lza2_vld) ? ({lza2_p1, lza2_p0} + 8) : 5'd24;
assign group_result3 = (lza3_vld) ? ({lza3_p1, lza3_p0} + 12) : 5'd24;
assign group_result4 = (lza4_vld) ? ({lza4_p1, lza4_p0} + 16) : 5'd24;
assign group_result5 = (lza5_vld) ? ({lza5_p1, lza5_p0} + 20) : 5'd24;

// Determine the highest set bit
wire [4:0] result;
assign result = 
    (lza0_vld && group_result0 < 24) ? group_result0 : 
    (lza1_vld && group_result1 < 24) ? group_result1 : 
    (lza2_vld && group_result2 < 24) ? group_result2 : 
    (lza3_vld && group_result3 < 24) ? group_result3 : 
    (lza4_vld && group_result4 < 24) ? group_result4 : 
    (lza5_vld && group_result5 < 24) ? group_result5 : 
    5'd24;

// Check if all bits are zero
assign lza_result = result;
assign lza_result_zero = (lza0_vld == 0) && 
                         (lza1_vld == 0) && 
                         (lza2_vld == 0) && 
                         (lza3_vld == 0) && 
                         (lza4_vld == 0) && 
                         (lza5_vld == 0);

endmodule