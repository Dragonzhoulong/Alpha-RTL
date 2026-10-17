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

// Pre-encoding signals
wire    [23:0]  carry_d;
wire    [23:0]  carry_g;
wire    [23:0]  carry_p;
wire    [23:0]  lza_precod;

// Carry signals
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

//==========================================================
//                     LZA coding (optimized)
//==========================================================

// Partition into 4-bit chunks for efficient encoding
wire [3:0] lza_upper   = lza_precod[23:20]; // Bits 23-20
wire [3:0] lza_upper_lower = lza_precod[19:16]; // Bits 19-16
wire [3:0] lza_middle2 = lza_precod[15:12]; // Bits 15-12
wire [3:0] lza_middle1 = lza_precod[11:8];  // Bits 11-8
wire [3:0] lza_lower2  = lza_precod[7:4];   // Bits 7-4
wire [3:0] lza_lower3  = lza_precod[3:0];   // Bits 3-0

// Direct encoding of lza_precod to find the position of the highest set bit
// This version uses a compact and efficient encoding strategy with a hierarchical approach

// Check for bits in groups from MSB to LSB
wire has_upper = |lza_upper;
wire has_upper_lower = |lza_upper_lower;
wire has_middle2 = |lza_middle2;
wire has_middle1 = |lza_middle1;
wire has_lower2 = |lza_lower2;
wire has_lower3 = |lza_lower3;

// Compute lza_result using direct encoding with hierarchical checking
assign lza_result = 
    has_upper ? 
        (lza_upper[3] ? 5'd0 : 
         lza_upper[2] ? 5'd1 : 
         lza_upper[1] ? 5'd2 : 
         lza_upper[0] ? 5'd3 : 5'd4) :
    has_upper_lower ? 
        (lza_upper_lower[3] ? 5'd4 : 
         lza_upper_lower[2] ? 5'd5 : 
         lza_upper_lower[1] ? 5'd6 : 
         lza_upper_lower[0] ? 5'd7 : 5'd8) :
    has_middle2 ? 
        (lza_middle2[3] ? 5'd8 : 
         lza_middle2[2] ? 5'd9 : 
         lza_middle2[1] ? 5'd10 : 
         lza_middle2[0] ? 5'd11 : 5'd12) :
    has_middle1 ? 
        (lza_middle1[3] ? 5'd12 : 
         lza_middle1[2] ? 5'd13 : 
         lza_middle1[1] ? 5'd14 : 
         lza_middle1[0] ? 5'd15 : 5'd16) :
    has_lower2 ? 
        (lza_lower2[3] ? 5'd16 : 
         lza_lower2[2] ? 5'd17 : 
         lza_lower2[1] ? 5'd18 : 
         lza_lower2[0] ? 5'd19 : 5'd20) :
    has_lower3 ? 
        (lza_lower3[3] ? 5'd20 : 
         lza_lower3[2] ? 5'd21 : 
         lza_lower3[1] ? 5'd22 : 
         lza_lower3[0] ? 5'd23 : 5'd24) :
    5'd24;

// Compute lza_result_zero
assign lza_result_zero = ~|lza_precod;

endmodule