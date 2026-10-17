module multi_8bit (
    input  wire [ 7:0] A,
    input  wire [ 7:0] B,
    output wire [15:0] product
);
    // Urdhva Tiryagbhyam (Vedic) decomposition: split each operand into two
    // 4-bit halves and sum the 4 cross-products. Lets the synthesizer optimise
    // each 4x4 sub-multiply independently — typically a small but real ADP
    // win at this width.
    wire [3:0] a_l = A[3:0];
    wire [3:0] a_h = A[7:4];
    wire [3:0] b_l = B[3:0];
    wire [3:0] b_h = B[7:4];

    wire [7:0] ll = a_l * b_l;     // bits  [7:0]
    wire [7:0] lh = a_l * b_h;     // bits [11:4]
    wire [7:0] hl = a_h * b_l;     // bits [11:4]
    wire [7:0] hh = a_h * b_h;     // bits [15:8]

    wire [8:0] mid = lh + hl;      // 9 bits to hold carry
    wire [15:0] cross_term = {3'b0, mid, 4'b0};
    wire [15:0] hi    = {hh, 8'b0};
    wire [15:0] lo    = {8'b0, ll};

    assign product = lo + cross_term + hi;
endmodule
