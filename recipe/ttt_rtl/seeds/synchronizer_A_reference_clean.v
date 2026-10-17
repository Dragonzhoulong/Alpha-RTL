// Seed A — Clean reference-style MUX-based synchronizer.
// Same architecture as RTLLM's verified implementation but fixes a latent
// bug there: the en_data_reg register (clocked by clk_a) used `!brstn`
// instead of `!arstn` for its reset. This version resets it with arstn.
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

    // Source (clk_a) domain
    always @(posedge clk_a or negedge arstn) begin
        if (!arstn) data_reg <= 4'd0;
        else        data_reg <= data_in;
    end

    always @(posedge clk_a or negedge arstn) begin
        if (!arstn) en_data_reg <= 1'b0;
        else        en_data_reg <= data_en;
    end

    // Destination (clk_b) domain: two-flop synchronizer on the enable.
    always @(posedge clk_b or negedge brstn) begin
        if (!brstn) en_clap_one <= 1'b0;
        else        en_clap_one <= en_data_reg;
    end

    always @(posedge clk_b or negedge brstn) begin
        if (!brstn) en_clap_two <= 1'b0;
        else        en_clap_two <= en_clap_one;
    end

    // Output mux: capture data_reg when the synchronized enable is high,
    // otherwise hold the previous value.
    always @(posedge clk_b or negedge brstn) begin
        if (!brstn)             dataout <= 4'd0;
        else if (en_clap_two)   dataout <= data_reg;
        else                    dataout <= dataout;
    end
endmodule
