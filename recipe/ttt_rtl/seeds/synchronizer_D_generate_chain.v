// Seed D — Parameterised sync-chain variant.
// Uses a localparam-sized shift register so the synchronizer depth can be
// tuned; defaults to depth 2 to match the spec. The output stage explicitly
// muxes between data_reg and the held dataout to mirror the RTLLM wording.
module synchronizer(
    input            clk_a,
    input            clk_b,
    input            arstn,
    input            brstn,
    input      [3:0] data_in,
    input            data_en,
    output reg [3:0] dataout
);
    localparam integer SYNC_DEPTH = 2;

    reg [3:0]            data_reg;
    reg                  en_data_reg;
    reg [SYNC_DEPTH-1:0] en_sync;

    always @(posedge clk_a or negedge arstn) begin
        if (!arstn) data_reg <= 4'd0;
        else        data_reg <= data_in;
    end

    always @(posedge clk_a or negedge arstn) begin
        if (!arstn) en_data_reg <= 1'b0;
        else        en_data_reg <= data_en;
    end

    always @(posedge clk_b or negedge brstn) begin
        if (!brstn) en_sync <= {SYNC_DEPTH{1'b0}};
        else        en_sync <= {en_sync[SYNC_DEPTH-2:0], en_data_reg};
    end

    wire en_sync_final = en_sync[SYNC_DEPTH-1];

    always @(posedge clk_b or negedge brstn) begin
        if (!brstn)              dataout <= 4'd0;
        else                     dataout <= en_sync_final ? data_reg : dataout;
    end
endmodule
