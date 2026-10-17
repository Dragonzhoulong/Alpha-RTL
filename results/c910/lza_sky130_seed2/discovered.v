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

// Carry signals
wire    [23:0]  carry_p;
wire    [23:0]  carry_g;
wire    [23:0]  carry_d;

// Pre-code signals
wire    [23:0]  lza_precod;

// Carry signals assignment
assign carry_p = summand ^ addend;
assign carry_g = summand & addend;
assign carry_d = ~(summand | addend);

// Pre-code bit 0 (special case)
assign lza_precod[0] =
     carry_p[1] && (carry_g[0] && sub_vld || carry_d[0]) 
 || !carry_p[1] && (carry_d[0] && sub_vld || carry_g[0]);

// Pre-code bit 23 (special case)
assign lza_precod[23] =
     sub_vld && (carry_g[23] && !carry_d[22] || carry_d[23] && !carry_g[22])
 || !sub_vld && (carry_d[23] && !carry_d[22] || !carry_d[23]);

// Pre-code bits [22:1] (general case)
assign lza_precod[22:1] =
    carry_p[23:2] & (carry_g[22:1] & ~carry_d[21:0] | carry_d[22:1] & ~carry_g[21:0])
 | ~carry_p[23:2] & (carry_g[22:1] & ~carry_g[21:0] | carry_d[22:1] & ~carry_d[21:0]);

// Priority encoding logic optimized for lower area and delay
// Split into 4-bit segments for efficient encoding
wire [3:0] upper_4 = lza_precod[23:20];
wire [3:0] middle_4_1 = lza_precod[19:16];
wire [3:0] middle_4_2 = lza_precod[15:12];
wire [3:0] middle_4_3 = lza_precod[11:8];
wire [3:0] middle_4_4 = lza_precod[7:4];
wire [3:0] lower_4 = lza_precod[3:0];

// Encode 4-bit segments to 2-bit position
wire [1:0] encoded_upper_pos;
wire [1:0] encoded_middle_1_pos;
wire [1:0] encoded_middle_2_pos;
wire [1:0] encoded_middle_3_pos;
wire [1:0] encoded_middle_4_pos;
wire [1:0] encoded_lower_pos;

// Encode upper 4 bits (bits 23-20)
assign encoded_upper_pos = 
  upper_4[3] ? 2'b00 : 
  upper_4[2] ? 2'b01 : 
  upper_4[1] ? 2'b10 : 
  upper_4[0] ? 2'b11 : 
  2'b00;

// Encode middle 4 bits (bits 19-16)
assign encoded_middle_1_pos = 
  middle_4_1[3] ? 2'b00 : 
  middle_4_1[2] ? 2'b01 : 
  middle_4_1[1] ? 2'b10 : 
  middle_4_1[0] ? 2'b11 : 
  2'b00;

// Encode middle 4 bits (bits 15-12)
assign encoded_middle_2_pos = 
  middle_4_2[3] ? 2'b00 : 
  middle_4_2[2] ? 2'b01 : 
  middle_4_2[1] ? 2'b10 : 
  middle_4_2[0] ? 2'b11 : 
  2'b00;

// Encode middle 4 bits (bits 11-8)
assign encoded_middle_3_pos = 
  middle_4_3[3] ? 2'b00 : 
  middle_4_3[2] ? 2'b01 : 
  middle_4_3[1] ? 2'b10 : 
  middle_4_3[0] ? 2'b11 : 
  2'b00;

// Encode middle 4 bits (bits 7-4)
assign encoded_middle_4_pos = 
  middle_4_4[3] ? 2'b00 : 
  middle_4_4[2] ? 2'b01 : 
  middle_4_4[1] ? 2'b10 : 
  middle_4_4[0] ? 2'b11 : 
  2'b00;

// Encode lower 4 bits (bits 3-0)
assign encoded_lower_pos = 
  lower_4[3] ? 2'b00 : 
  lower_4[2] ? 2'b01 : 
  lower_4[1] ? 2'b10 : 
  lower_4[0] ? 2'b11 : 
  2'b00;

// Determine which segment has the first '1'
wire leading_upper = (upper_4 != 4'b0);
wire leading_middle_1 = (middle_4_1 != 4'b0);
wire leading_middle_2 = (middle_4_2 != 4'b0);
wire leading_middle_3 = (middle_4_3 != 4'b0);
wire leading_middle_4 = (middle_4_4 != 4'b0);
wire leading_lower = (lower_4 != 4'b0);

// Convert 2-bit position to 5-bit result
wire [4:0] result_upper;
wire [4:0] result_middle_1;
wire [4:0] result_middle_2;
wire [4:0] result_middle_3;
wire [4:0] result_middle_4;
wire [4:0] result_lower;

assign result_upper = 
  upper_4[3] ? 5'd0 : 
  upper_4[2] ? 5'd1 : 
  upper_4[1] ? 5'd2 : 
  upper_4[0] ? 5'd3 : 
  5'd4;

assign result_middle_1 = 
  middle_4_1[3] ? 5'd4 : 
  middle_4_1[2] ? 5'd5 : 
  middle_4_1[1] ? 5'd6 : 
  middle_4_1[0] ? 5'd7 : 
  5'd8;

assign result_middle_2 = 
  middle_4_2[3] ? 5'd8 : 
  middle_4_2[2] ? 5'd9 : 
  middle_4_2[1] ? 5'd10 : 
  middle_4_2[0] ? 5'd11 : 
  5'd12;

assign result_middle_3 = 
  middle_4_3[3] ? 5'd12 : 
  middle_4_3[2] ? 5'd13 : 
  middle_4_3[1] ? 5'd14 : 
  middle_4_3[0] ? 5'd15 : 
  5'd16;

assign result_middle_4 = 
  middle_4_4[3] ? 5'd16 : 
  middle_4_4[2] ? 5'd17 : 
  middle_4_4[1] ? 5'd18 : 
  middle_4_4[0] ? 5'd19 : 
  5'd20;

assign result_lower = 
  lower_4[3] ? 5'd20 : 
  lower_4[2] ? 5'd21 : 
  lower_4[1] ? 5'd22 : 
  lower_4[0] ? 5'd23 : 
  5'd24;

// Determine which segment has the first '1'
assign lza_result = 
  (leading_upper) ? result_upper : 
  (leading_middle_1) ? result_middle_1 : 
  (leading_middle_2) ? result_middle_2 : 
  (leading_middle_3) ? result_middle_3 : 
  (leading_middle_4) ? result_middle_4 : 
  (leading_lower) ? result_lower : 
  5'd24;

// lza_result_zero is 1 if all bits in lza_precod are 0
assign lza_result_zero = ~(|lza_precod);

endmodule