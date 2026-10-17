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

wire [23:0] carry_p;
wire [23:0] carry_g;
wire [23:0] carry_d;
wire [23:0] lza_precod;

assign carry_p = summand ^ addend;   // propagate
assign carry_g = summand & addend;   // generate
assign carry_d = ~(summand | addend); // delete

assign lza_precod[0] =
     carry_p[1] && (carry_g[0] && sub_vld || carry_d[0])
 || !carry_p[1] && (carry_d[0] && sub_vld || carry_g[0]);

assign lza_precod[23] =
     sub_vld && (carry_g[23] && !carry_d[22] || carry_d[23] && !carry_g[22])
 || !sub_vld && (carry_d[23] && !carry_d[22] || !carry_d[23]);

assign lza_precod[22:1] =
    carry_p[23:2] & (carry_g[22:1] & ~carry_d[21:0] | carry_d[22:1] & ~carry_g[21:0])
 | ~carry_p[23:2] & (carry_g[22:1] & ~carry_g[21:0] | carry_d[22:1] & ~carry_d[21:0]);

// Priority encoding logic using a compact binary tree approach
wire lza_vld;
assign lza_vld = |lza_precod;

wire lza_result_zero;
assign lza_result_zero = ~lza_vld;

// Determine position of highest set bit using a compact binary tree approach
wire [2:0] lza_p0, lza_p1, lza_p2;
wire [4:0] lza_result;

// Create a binary tree of lza_precod bits
wire [2:0] lza_bit0, lza_bit1, lza_bit2, lza_bit3, lza_bit4, lza_bit5, lza_bit6, lza_bit7;
wire [2:0] lza_bit8, lza_bit9, lza_bit10, lza_bit11, lza_bit12, lza_bit13, lza_bit14, lza_bit15;
wire [2:0] lza_bit16, lza_bit17, lza_bit18, lza_bit19, lza_bit20, lza_bit21, lza_bit22, lza_bit23;

// First level: group bits 0-2
assign lza_bit0 = {lza_precod[0], lza_precod[1], lza_precod[2]};
assign lza_p0 = lza_precod[0];
assign lza_p1 = lza_precod[1] || (lza_precod[2] && !lza_precod[1]);
assign lza_p2 = lza_precod[2];

// Group bits 3-5
assign lza_bit1 = {lza_precod[3], lza_precod[4], lza_precod[5]};
assign lza_p3 = lza_precod[3];
assign lza_p4 = lza_precod[4] || (lza_precod[5] && !lza_precod[4]);
assign lza_p5 = lza_precod[5];

// Group bits 6-8
assign lza_bit2 = {lza_precod[6], lza_precod[7], lza_precod[8]};
assign lza_p6 = lza_precod[6];
assign lza_p7 = lza_precod[7] || (lza_precod[8] && !lza_precod[7]);
assign lza_p8 = lza_precod[8];

// Group bits 9-11
assign lza_bit3 = {lza_precod[9], lza_precod[10], lza_precod[11]};
assign lza_p9 = lza_precod[9];
assign lza_p10 = lza_precod[10] || (lza_precod[11] && !lza_precod[10]);
assign lza_p11 = lza_precod[11];

// Group bits 12-14
assign lza_bit4 = {lza_precod[12], lza_precod[13], lza_precod[14]};
assign lza_p12 = lza_precod[12];
assign lza_p13 = lza_precod[13] || (lza_precod[14] && !lza_precod[13]);
assign lza_p14 = lza_precod[14];

// Group bits 15-17
assign lza_bit5 = {lza_precod[15], lza_precod[16], lza_precod[17]};
assign lza_p15 = lza_precod[15];
assign lza_p16 = lza_precod[16] || (lza_precod[17] && !lza_precod[16]);
assign lza_p17 = lza_precod[17];

// Group bits 18-20
assign lza_bit6 = {lza_precod[18], lza_precod[19], lza_precod[20]};
assign lza_p18 = lza_precod[18];
assign lza_p19 = lza_precod[19] || (lza_precod[20] && !lza_precod[19]);
assign lza_p20 = lza_precod[20];

// Group bits 21-23
assign lza_bit7 = {lza_precod[21], lza_precod[22], lza_precod[23]};
assign lza_p21 = lza_precod[21];
assign lza_p22 = lza_precod[22] || (lza_precod[23] && !lza_precod[22]);
assign lza_p23 = lza_precod[23];

// Combine all bits for final result
assign lza_result = 
    (lza_p23 ? 5'd0 :
    lza_p22 ? 5'd1 :
    lza_p21 ? 5'd2 :
    lza_p20 ? 5'd3 :
    lza_p19 ? 5'd4 :
    lza_p18 ? 5'd5 :
    lza_p17 ? 5'd6 :
    lza_p16 ? 5'd7 :
    lza_p15 ? 5'd8 :
    lza_p14 ? 5'd9 :
    lza_p13 ? 5'd10 :
    lza_p12 ? 5'd11 :
    lza_p11 ? 5'd12 :
    lza_p10 ? 5'd13 :
    lza_p9 ? 5'd14 :
    lza_p8 ? 5'd15 :
    lza_p7 ? 5'd16 :
    lza_p6 ? 5'd17 :
    lza_p5 ? 5'd18 :
    lza_p4 ? 5'd19 :
    lza_p3 ? 5'd20 :
    lza_p2 ? 5'd21 :
    lza_p1 ? 5'd22 :
    lza_p0 ? 5'd23 : 5'd24);

endmodule