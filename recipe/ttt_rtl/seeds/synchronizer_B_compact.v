// Seed B — Compact single-always-per-domain style.
// Same behaviour as the MUX synchronizer, but each clock domain is handled
// in exactly one always block; the two-flop sync chain is packed into a
// 2-bit shift register.
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
    reg [1:0] en_sync;   // {en_clap_two, en_clap_one}

    // Source domain: data path and enable capture share one always block.
    always @(posedge clk_a or negedge arstn) begin
        if (!arstn) begin
            data_reg    <= 4'd0;
            en_data_reg <= 1'b0;
        end else begin
            data_reg    <= data_in;
            en_data_reg <= data_en;
        end
    end

    // Destination domain: 2-flop sync of the enable and the output mux.
    always @(posedge clk_b or negedge brstn) begin
        if (!brstn) begin
            en_sync <= 2'b00;
            dataout <= 4'd0;
        end else begin
            en_sync <= {en_sync[0], en_data_reg};
            if (en_sync[1]) dataout <= data_reg;
        end
    end
endmodule
