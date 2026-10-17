module multi_8bit (
    input  wire [ 7:0] A,
    input  wire [ 7:0] B,
    output wire [15:0] product
);
    // Radix-4 Booth, unsigned 8x8: zero-extend B to 10 bits to make it a
    // signed-positive number, giving 5 Booth digits (column 0,2,4,6,8) each
    // in {-2A,-A,0,+A,+2A}. The final sum still equals A*B mod 2^16.
    wire signed [10:0] Ap  = {{3{1'b0}}, A};   // +A
    wire signed [10:0] An  = -Ap;               // -A
    wire signed [10:0] A2p = Ap <<< 1;          // +2A
    wire signed [10:0] A2n = -A2p;              // -2A

    function automatic [10:0] booth_sel;
        input [2:0] g;
        input signed [10:0] ap, an, a2p, a2n;
        begin
            case (g)
                3'b001, 3'b010: booth_sel = ap;
                3'b011:         booth_sel = a2p;
                3'b100:         booth_sel = a2n;
                3'b101, 3'b110: booth_sel = an;
                default:        booth_sel = 11'sd0;
            endcase
        end
    endfunction

    wire [9:0] B_ext = {2'b00, B};   // zero-extend; B_ext[-1] is implicit 0

    wire [2:0] g0 = {B_ext[1], B_ext[0], 1'b0};
    wire [2:0] g1 = {B_ext[3], B_ext[2], B_ext[1]};
    wire [2:0] g2 = {B_ext[5], B_ext[4], B_ext[3]};
    wire [2:0] g3 = {B_ext[7], B_ext[6], B_ext[5]};
    wire [2:0] g4 = {B_ext[9], B_ext[8], B_ext[7]};

    wire signed [10:0] pp0r = booth_sel(g0, Ap, An, A2p, A2n);
    wire signed [10:0] pp1r = booth_sel(g1, Ap, An, A2p, A2n);
    wire signed [10:0] pp2r = booth_sel(g2, Ap, An, A2p, A2n);
    wire signed [10:0] pp3r = booth_sel(g3, Ap, An, A2p, A2n);
    wire signed [10:0] pp4r = booth_sel(g4, Ap, An, A2p, A2n);

    // Sign-extend each PP to 16 bits, then arithmetic-shift to its column
    wire signed [15:0] pp0 = {{5{pp0r[10]}}, pp0r};
    wire signed [15:0] pp1 = {{5{pp1r[10]}}, pp1r} <<< 2;
    wire signed [15:0] pp2 = {{5{pp2r[10]}}, pp2r} <<< 4;
    wire signed [15:0] pp3 = {{5{pp3r[10]}}, pp3r} <<< 6;
    wire signed [15:0] pp4 = {{5{pp4r[10]}}, pp4r} <<< 8;

    assign product = pp0 + pp1 + pp2 + pp3 + pp4;
endmodule
