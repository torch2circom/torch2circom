pragma circom 2.0.0;

include "../circomlib-ml/circuits/linear.circom";
include "../circomlib-ml/circuits/SingleheadAttn.circom";
include "../circomlib-ml/circuits/PositionalEncoding.circom";
include "../circomlib-ml/circuits/ReLU.circom";

template Model() {
signal input pe_b[3][4];
signal input in[2][3][4];
signal  encoderattn_q[2][3][4];
signal  encoderattn_k[2][3][4];
signal  encoderattn_v[2][3][4];
signal input encoderattn_sq[4][4];
signal input encoderattn_sk[4][4];
signal input encoderattn_sv[4][4];
signal input encoderattn_bq[4];
signal input encoderattn_bk[4];
signal input encoderattn_bv[4];
signal input linear_1_b[4][4];
signal input linear_1_c[4];
signal input linear_2_b[4][4];
signal input linear_2_c[4];
signal input pe2_b[3][4];
signal input pe2_a[2][3][4];
signal  decoderselfattn_q[2][1][4];
signal  decoderselfattn_k[2][3][4];
signal  decoderselfattn_v[2][3][4];
signal input decoderselfattn_sq[4][4];
signal input decoderselfattn_sk[4][4];
signal input decoderselfattn_sv[4][4];
signal input decoderselfattn_bq[4];
signal input decoderselfattn_bk[4];
signal input decoderselfattn_bv[4];
signal input linear_ds1_b[4][4];
signal input linear_ds1_c[4];
signal input linear_ds2_b[4][4];
signal input linear_ds2_c[4];
signal  decodercrossattn_q[2][1][4];
signal  decodercrossattn_k[2][3][4];
signal  decodercrossattn_v[2][3][4];
signal input decodercrossattn_sq[4][4];
signal input decodercrossattn_sk[4][4];
signal input decodercrossattn_sv[4][4];
signal input decodercrossattn_bq[4];
signal input decodercrossattn_bk[4];
signal input decodercrossattn_bv[4];
signal input linear_dc1_b[4][4];
signal input linear_dc1_c[4];
signal input linear_dc2_b[4][4];
signal input linear_dc2_c[4];
signal input linear_vocab_logits_b[4][3];
signal input linear_vocab_logits_c[3];
signal output out[2][1][3];

component pe = PositionalEncoding(2, 3, 4);
component encoderattn = SingleheadAttn(2, 3, 3, 4);
component linear_1 = linear(2, 3, 4, 4);
component RELU[2][3][4];
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            RELU[i0][i1][i2] = ReLU();
}}}
component linear_2 = linear(2, 3, 4, 4);
component pe2 = PositionalEncoding(2, 3, 4);
component decoderselfattn = SingleheadAttn(2, 1, 3, 4);
component linear_ds1 = linear(2, 1, 4, 4);
component RELU2[2][1][4];
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            RELU2[i0][i1][i2] = ReLU();
}}}
component linear_ds2 = linear(2, 1, 4, 4);
component decodercrossattn = SingleheadAttn(2, 1, 3, 4);
component linear_dc1 = linear(2, 1, 4, 4);
component RELU3[2][1][4];
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            RELU3[i0][i1][i2] = ReLU();
}}}
component linear_dc2 = linear(2, 1, 4, 4);
component linear_vocab_logits = linear(2, 1, 4, 3);

for (var i0 = 0; i0 < 3; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        pe.b[i0][i1] <== pe_b[i0][i1];
}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            pe.a[i0][i1][i2] <== in[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
        for (var i1 = 0; i1 < 3; i1++) {
            for (var i2 = 0; i2 < 4; i2++) {
                encoderattn_q[i0][i1][i2] <== pe.out[i0][i1][i2];
    }}}
    for (var i0 = 0; i0 < 2; i0++) {
        for (var i1 = 0; i1 < 3; i1++) {
            for (var i2 = 0; i2 < 4; i2++) {
                encoderattn_k[i0][i1][i2] <== pe.out[i0][i1][i2];
    }}}

    // manual v
    for (var i0 = 0; i0 < 2; i0++) {
        for (var i1 = 0; i1 < 3; i1++) {
            for (var i2 = 0; i2 < 4; i2++) {
                encoderattn_v[i0][i1][i2] <== pe.out[i0][i1][i2];
    }}}
    
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            encoderattn.q[i0][i1][i2] <== encoderattn_q[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            encoderattn.k[i0][i1][i2] <== encoderattn_k[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            encoderattn.v[i0][i1][i2] <== encoderattn_v[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        encoderattn.sq[i0][i1] <== encoderattn_sq[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        encoderattn.sk[i0][i1] <== encoderattn_sk[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        encoderattn.sv[i0][i1] <== encoderattn_sv[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    encoderattn.bq[i0] <== encoderattn_bq[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    encoderattn.bk[i0] <== encoderattn_bk[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    encoderattn.bv[i0] <== encoderattn_bv[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            linear_1.a[i0][i1][i2] <== encoderattn.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        linear_1.b[i0][i1] <== linear_1_b[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    linear_1.c[i0] <== linear_1_c[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            RELU[i0][i1][i2].in <== linear_1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            linear_2.a[i0][i1][i2] <== RELU[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        linear_2.b[i0][i1] <== linear_2_b[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    linear_2.c[i0] <== linear_2_c[i0];
}
for (var i0 = 0; i0 < 3; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        pe2.b[i0][i1] <== pe2_b[i0][i1];
}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            pe2.a[i0][i1][i2] <== pe2_a[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
            for (var i2 = 0; i2 < 4; i2++) {
                decoderselfattn_q[i0][0][i2] <== pe2.out[i0][2][i2]; //B, 1, d / B, N, d
    }}
    // manual decoder k
    for (var i0 = 0; i0 < 2; i0++) {
        for (var i1 = 0; i1 < 3; i1++) {
            for (var i2 = 0; i2 < 4; i2++) {
                decoderselfattn_k[i0][i1][i2] <== pe2.out[i0][i1][i2];
    }}}
    // manual decoder v
    for (var i0 = 0; i0 < 2; i0++) {
        for (var i1 = 0; i1 < 3; i1++) {
            for (var i2 = 0; i2 < 4; i2++) {
                decoderselfattn_v[i0][i1][i2] <== pe2.out[i0][i1][i2];
    }}}
    
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            decoderselfattn.q[i0][i1][i2] <== decoderselfattn_q[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            decoderselfattn.k[i0][i1][i2] <== decoderselfattn_k[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            decoderselfattn.v[i0][i1][i2] <== decoderselfattn_v[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        decoderselfattn.sq[i0][i1] <== decoderselfattn_sq[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        decoderselfattn.sk[i0][i1] <== decoderselfattn_sk[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        decoderselfattn.sv[i0][i1] <== decoderselfattn_sv[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    decoderselfattn.bq[i0] <== decoderselfattn_bq[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    decoderselfattn.bk[i0] <== decoderselfattn_bk[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    decoderselfattn.bv[i0] <== decoderselfattn_bv[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            linear_ds1.a[i0][i1][i2] <== decoderselfattn.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        linear_ds1.b[i0][i1] <== linear_ds1_b[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    linear_ds1.c[i0] <== linear_ds1_c[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            RELU2[i0][i1][i2].in <== linear_ds1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            linear_ds2.a[i0][i1][i2] <== RELU2[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        linear_ds2.b[i0][i1] <== linear_ds2_b[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    linear_ds2.c[i0] <== linear_ds2_c[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
            for (var i2 = 0; i2 < 4; i2++) {
                decodercrossattn_q[i0][0][i2] <== linear_ds2.out[i0][0][i2];
    }}
    // manual decoder cross k
    for (var i0 = 0; i0 < 2; i0++) {
        for (var i1 = 0; i1 < 3; i1++) {
            for (var i2 = 0; i2 < 4; i2++) {
                decodercrossattn_k[i0][i1][i2] <== linear_2.out[i0][i1][i2];
    }}}
    // manual decoder cross v
    for (var i0 = 0; i0 < 2; i0++) {
        for (var i1 = 0; i1 < 3; i1++) {
            for (var i2 = 0; i2 < 4; i2++) {
                decodercrossattn_v[i0][i1][i2] <== linear_2.out[i0][i1][i2];
    }}}
    
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            decodercrossattn.q[i0][i1][i2] <== decodercrossattn_q[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            decodercrossattn.k[i0][i1][i2] <== decodercrossattn_k[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            decodercrossattn.v[i0][i1][i2] <== decodercrossattn_v[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        decodercrossattn.sq[i0][i1] <== decodercrossattn_sq[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        decodercrossattn.sk[i0][i1] <== decodercrossattn_sk[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        decodercrossattn.sv[i0][i1] <== decodercrossattn_sv[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    decodercrossattn.bq[i0] <== decodercrossattn_bq[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    decodercrossattn.bk[i0] <== decodercrossattn_bk[i0];
}
for (var i0 = 0; i0 < 4; i0++) {
    decodercrossattn.bv[i0] <== decodercrossattn_bv[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            linear_dc1.a[i0][i1][i2] <== decodercrossattn.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        linear_dc1.b[i0][i1] <== linear_dc1_b[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    linear_dc1.c[i0] <== linear_dc1_c[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            RELU3[i0][i1][i2].in <== linear_dc1.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            linear_dc2.a[i0][i1][i2] <== RELU3[i0][i1][i2].out;
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 4; i1++) {
        linear_dc2.b[i0][i1] <== linear_dc2_b[i0][i1];
}}
for (var i0 = 0; i0 < 4; i0++) {
    linear_dc2.c[i0] <== linear_dc2_c[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 4; i2++) {
            linear_vocab_logits.a[i0][i1][i2] <== linear_dc2.out[i0][i1][i2];
}}}
for (var i0 = 0; i0 < 4; i0++) {
    for (var i1 = 0; i1 < 3; i1++) {
        linear_vocab_logits.b[i0][i1] <== linear_vocab_logits_b[i0][i1];
}}
for (var i0 = 0; i0 < 3; i0++) {
    linear_vocab_logits.c[i0] <== linear_vocab_logits_c[i0];
}
for (var i0 = 0; i0 < 2; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        for (var i2 = 0; i2 < 3; i2++) {
            out[i0][i1][i2] <== linear_vocab_logits.out[i0][i1][i2];
}}}

}

component main = Model();
