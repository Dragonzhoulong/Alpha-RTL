// Seed C — Gated-capture variant.
// Rather than muxing dataout every clk_b cycle, this version only clocks the
// output register when the synchronized enable is high. Functionally
// equivalent to the MUX synchronizer because an unclocked register holds its
// value. Often slightly lower switching activity (hence power).
module synchronizer(
    input            clk_a,
    input            clk_b,
    input            arstn,
    input            brstn,
    input      [3:0] data_in,
    input            data_en,
    output reg [3:0] dataout
);
    reg [3:0] data_reg;
    reg       en_data_reg;
    reg       en_clap_one;
    reg       en_clap_two;

    always @(posedge clk_a or negedge arstn) begin
        if (!arstn) begin
            data_reg    <= 4'd0;
            en_data_reg <= 1'b0;
        end else begin
            data_reg    <= data_in;
            en_data_reg <= data_en;
        end
    end

    always @(posedge clk_b or negedge brstn) begin
        if (!brstn) begin
            en_clap_one <= 1'b0;
            en_clap_two <= 1'b0;
        end else begin
            en_clap_one <= en_data_reg;
            en_clap_two <= en_clap_one;
        end
    end

    // Separate output register — hold when en_clap_two is low.
    always @(posedge clk_b or negedge brstn) begin
        if (!brstn)             dataout <= 4'd0;
        else if (en_clap_two)   dataout <= data_reg;
    end
endmodule
