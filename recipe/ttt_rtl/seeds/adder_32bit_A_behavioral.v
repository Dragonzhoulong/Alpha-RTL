// Seed A — Behavioral adder.
// Lets the synthesizer pick the best gate-level adder topology for the
// target library; typically yields the smallest area and good delay.
module adder_32bit(A, B, S, C32);
    input  [31:0] A;
    input  [31:0] B;
    output [31:0] S;
    output        C32;

    assign {C32, S} = A + B;
endmodule
