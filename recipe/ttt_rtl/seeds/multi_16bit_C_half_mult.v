module multi_16bit (
    input            clk,
    input            rst_n,
    input            start,
    input     [15:0] ain,
    input     [15:0] bin,
    output    [31:0] yout,
    output           done
);
    // Two-cycle half-multiplier. A single 8x16 sub-multiplier is reused
    // across two cycles (one for bin[7:0], one for bin[15:8]) and the
    // partial products are accumulated with the right shift. Far fewer
    // cycles than the bit-serial seed, with a different area/delay corner
    // (bigger per-cycle datapath, fewer cycles, no FSM-style mux tree).
    reg [31:0] acc;
    reg [ 1:0] cnt;
    reg        done_r;

    wire [ 7:0] bin_byte = (cnt == 2'd0) ? bin[7:0] : bin[15:8];
    wire [23:0] partial  = ain * bin_byte;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc    <= 32'd0;
            cnt    <= 2'd0;
            done_r <= 1'b0;
        end else if (start) begin
            case (cnt)
                2'd0: begin
                    acc <= {8'd0, partial};
                    cnt <= 2'd1;
                end
                2'd1: begin
                    acc    <= acc + {partial, 8'd0};
                    cnt    <= 2'd2;
                    done_r <= 1'b1;
                end
                default: ; // hold done asserted
            endcase
        end else begin
            cnt    <= 2'd0;
            done_r <= 1'b0;
        end
    end

    assign yout = acc;
    assign done = done_r;
endmodule
