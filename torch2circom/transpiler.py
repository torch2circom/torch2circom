from .circom import *
from .model import *
from .transformer import TransformerTranslator


import os

from difflib import SequenceMatcher

poly_activation = "4wEAAAAAAAAAAAAAAAEAAAACAAAAQwAAAHMMAAAAfABkARMAfAAXAFMAKQJO6QIAAACpACkB2gF4\ncgIAAAByAgAAAHpOL3Zhci9mb2xkZXJzL2d0L3NnM3Y4cmQxM2w1Mmp4OTFtZmJnemJmYzAwMDBn\nbi9UL2lweWtlcm5lbF8xNTU3MS8yMTc2NzAzOTE5LnB52gg8bGFtYmRhPggAAADzAAAAAA==\n"


def transpile(output_dir: str = "output") -> Circuit:
    """Transpile a Keras model to a CIRCOM circuit."""

    batch_size = 2
    embed_dim = 4
    num_blocks = 1
    num_heads = 1  # Must be factor of token size
    max_context_length = 1000
    CUDA = True
    num_epochs = 1000
    learning_rate = 1e-3
    # device = torch.device("cuda:0" if CUDA else "cpu")
    use_teacher_forcing = False

    # torch.set_default_tensor_type(torch.cuda.FloatTensor if CUDA else torch.FloatTensor)
    encoder_vocab_size = 3
    output_vocab_size = 3
    model = TransformerTranslator(embed_dim, num_blocks, num_heads, encoder_vocab_size, output_vocab_size, CUDA=CUDA)

    modules = []
    from collections import Counter

    module_counter = Counter()
    for n_, m in model.named_modules():
        if n_:
            # dealing with embeddings
            if "emb_" in n_:
                modules.append([n_, m])
            if "pe" == n_ or "pe2" == n_:
                modules.append([n_, m])
            if "attn" in n_ and "." not in n_:
                modules.append([n_, m])
            # if "addandnorm" in n_ and "." not in n_:
            #     modules.append([n_, m])
            if "RELU" in n_:
                modules.append([n_, m])
            if "linear_" in n_:
                modules.append([n_, m])

    circuit = Circuit()

    for layer in modules:
        if "linear" in layer[0]:
            circuit.add_components(transpile_layer(layer))
        # elif "emb" in layer[0]:
        #     circuit.add_components(transpile_layer(layer))
        elif "pe" == layer[0] or "pe2" == layer[0]:
            circuit.add_components(transpile_layer(layer))
        elif "attn" in layer[0]:
            circuit.add_components(transpile_layer(layer))
        elif "RELU" in layer[0]:
            circuit.add_components(transpile_layer(layer))
        elif "addandnorm" in layer[0]:
            circuit.add_components(transpile_layer(layer))

    # circuit.add_components(transpile_layer(model.layers[-1], True))

    # if raw:
    #     if circuit.components[-1].template.op_name == "ArgMax":
    #         circuit.components.pop()
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + "/circuit.json", "w") as f:
        f.write(circuit.to_json())

    with open(output_dir + "/circuit.circom", "w") as f:
        f.write(post_process(circuit.to_circom()))

    return circuit


def transpile_layer(layer, last: bool = False) -> typing.List[Component]:
    """Transpile a Keras layer to CIRCOM component(s)."""

    # if layer == "ReLU":
    #     return transpile_ReLU(layer)
    if "linear" in layer[0]:
        return transpile_Dense(layer, last)
    if "emb_" in layer[0]:
        return transpile_emb(layer)
    if "pe" == layer[0] or "pe2" == layer[0]:
        return transpile_pe(layer)
    if "attn" in layer[0]:
        return transpile_attn(layer)
    if "RELU" in layer[0]:
        return transpile_relu(layer)
    if "addandnorm" in layer[0]:
        return transpile_addandnorm(layer)

    raise NotImplementedError(f"Layer {layer} is not supported yet.")


# TODO: handle scaling
def transpile_addandnorm(layer) -> typing.List[Component]:
    # import ipdb

    # ipdb.set_trace()
    import torch

    emb = Component(
        layer[0],
        templates["3d_add2Mat"],
        [
            Signal("a", (2, 3, 4)),  # vocab, d
            Signal("b", (2, 3, 4), torch.zeros(2, 3, 4)),  # B, N
        ],
        [Signal("out", (2, 3, 4))],  # B, N, d
        {"B": 2, "N": 3, "d": 4},  #
    )

    return [emb]


def transpile_relu(layer: Layer) -> typing.List[Component]:
    if layer[0] in ["RELU2", "RELU3"]:
        inout_size = (2, 1, 4)
    else:
        inout_size = (2, 3, 4)
    return [Component(layer[0], templates["ReLU"], [Signal("in", inout_size)], [Signal("out", inout_size)])]


def transpile_attn(layer) -> typing.List[Component]:
    # import ipdb

    # ipdb.set_trace()
    import torch

    # q_size:
    # 128, 256, 512 for encoder
    # 128, 1, 512 for decoder self attn
    # 128, 1, 512 for decoder cross attn
    if "encoder" in layer[0]:
        q_size = (2, 3, 4)
    else:
        q_size = (2, 1, 4)

    emb = Component(
        layer[0],
        templates["SingleheadAttn"],
        [
            Signal("q", q_size, torch.zeros(q_size)),  # B, Nq, d
            Signal("k", (2, 3, 4), torch.zeros(2, 3, 4)),  # B, Nk, d
            Signal("v", (2, 3, 4), torch.zeros(2, 3, 4)),  # B, Nk, d
            Signal("sq", layer[1]._modules["qw"]._parameters["weight"].size(), layer[1]._modules["qw"]._parameters["weight"]),  # B, Nk, d
            Signal("sk", layer[1]._modules["kw"]._parameters["weight"].size(), layer[1]._modules["kw"]._parameters["weight"]),  # B, Nk, d
            Signal("sv", layer[1]._modules["vw"]._parameters["weight"].size(), layer[1]._modules["vw"]._parameters["weight"]),  # B, Nk, d
            Signal("bq", layer[1]._modules["qw"]._parameters["bias"].size(), layer[1]._modules["qw"]._parameters["bias"]),  # B, Nk, d
            Signal("bk", layer[1]._modules["kw"]._parameters["bias"].size(), layer[1]._modules["kw"]._parameters["bias"]),  # B, Nk, d
            Signal("bv", layer[1]._modules["vw"]._parameters["bias"].size(), layer[1]._modules["vw"]._parameters["bias"]),  # B, Nk, d
        ],
        [Signal("out", q_size)],  # B, N, d
        {"B": 2, "N_q": q_size[1], "N_k": 3, "d": 4},  #
    )

    return [emb]


def transpile_pe(layer) -> typing.List[Component]:
    # import ipdb

    # ipdb.set_trace()
    import torch

    if layer[0] == "pe2":
        emb = Component(
            layer[0],
            templates["PositionalEncoding"],
            [
                Signal("b", layer[1]._buffers["emb_pe"].squeeze().size(), layer[1]._buffers["emb_pe"].squeeze()),  # vocab, d
                Signal("a", (2, 3, 4), torch.zeros(2, 3, 4)),  # B, N
            ],
            [Signal("out", (2, 3, 4))],  # B, N, d
            {"B": 2, "N": 3, "d": 4},  #
        )
    else:
        emb = Component(
            layer[0],
            templates["PositionalEncoding"],
            [
                Signal("b", layer[1]._buffers["emb_pe"].squeeze().size(), layer[1]._buffers["emb_pe"].squeeze()),  # vocab, d
                Signal("a", (2, 3, 4)),  # B, N
            ],
            [Signal("out", (2, 3, 4))],  # B, N, d
            {"B": 2, "N": 3, "d": 4},  #
        )

    return [emb]


def transpile_emb(layer) -> typing.List[Component]:
    # import ipdb

    # ipdb.set_trace()
    import torch

    emb = Component(
        layer[0],
        templates["3d_EmbeddingLookup"],
        [
            Signal("a", layer[1]._parameters["weight"].size(), layer[1]._parameters["weight"]),  # vocab, d
            Signal("b", (2, 3), torch.zeros(2, 3)),  # B, N
        ],
        [Signal("out", (2, 3, 4))],  # B, N, d
        {"N": 5, "d": 4, "B": 2, "m": 3},  #
    )

    return [emb]


def transpile_Dense(layer, last: bool = False) -> typing.List[Component]:
    # import ipdb

    # ipdb.set_trace()
    if "linear_ds" in layer[0] or "linear_dc" in layer[0]:
        in_size = (2, 1, 4)
        out_size = (2, 1, 4)
    elif "linear_vocab_logits" == layer[0]:
        in_size = (2, 1, 4)
        out_size = (2, 1, 3)
    else:
        in_size = (2, 3, 4)
        out_size = (2, 3, 4)
    dense = Component(
        layer[0],
        templates["linear"],
        [
            Signal("a", in_size),
            Signal("b", layer[1].weight.size(), layer[1].weight),
            Signal("c", layer[1].bias.size(), layer[1].bias),
        ],
        [Signal("out", out_size)],
        {"B": in_size[0], "N": in_size[1], "d": in_size[2], "d2": out_size[2]},
    )

    return [dense]


def post_process(raw):
    injection1 = """for (var i0 = 0; i0 < 2; i0++) {
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
    """

    injection2 = """for (var i0 = 0; i0 < 2; i0++) {
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
    """

    injection3 = """for (var i0 = 0; i0 < 2; i0++) {
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
    """

    splitlines = raw.split("\n")
    filtered = []
    for e in splitlines:
        if not (e.startswith("signal input") and ("attn_q" in e or "attn_k" in e or "attn_v" in e)):
            filtered.append(e)
        else:
            filtered.append("".join(e.split("input")))

    filtered_ = []
    for e in filtered:
        if e.startswith("signal input linear_vocab_logits_b"):
            # print(e)
            e = e.split("[")
            a = e[1].split("]")[0]
            b = e[2].split("]")[0]
            e_ = f"signal input linear_vocab_logits_b[{b}][{a}];"
            # print(e_)
            filtered_.append(e_)
        else:
            filtered_.append(e)

    filtered_2 = []
    for e in filtered_:
        if e.strip().startswith("RELU.in") or e.strip().startswith("RELU2.in") or e.strip().startswith("RELU3.in"):
            tmp = e.split("[")
            tmp[0] = tmp[0][:-3]
            tmp[3] = tmp[3].split("<")[0].strip() + ".in" + " <" + tmp[3].split("<")[1]
            # tmp[2] = tmp[2].split('<')[0].strip()
            tmp = "[".join(tmp)
            filtered_2.append(tmp)
        else:
            filtered_2.append(e)

    for idx, e in enumerate(filtered_2):
        if "linear_vocab_logits.b" in e.strip():
            line1 = filtered_2[idx - 2]
            line2 = filtered_2[idx - 1]
            chunk1 = line1.split("<")
            num1 = chunk1[1].strip()[0]
            chunk2 = line2.split("<")
            num2 = chunk2[1].strip()[0]
            newline1 = chunk1[0] + "< " + num2 + chunk1[1].strip()[1:]
            newline2 = chunk2[0] + "< " + num1 + chunk2[1].strip()[1:]
            break
    filtered_2[idx - 2] = newline1
    filtered_2[idx - 1] = newline2

    filtered_3 = []
    flag1, flag2, flag3 = False, False, False
    for idx, e in enumerate(filtered_2):
        if "== encoderattn_q" in e and not flag1:
            tmp1 = filtered_2[idx - 3]
            tmp2 = filtered_2[idx - 2]
            tmp3 = filtered_2[idx - 1]
            filtered_3 = filtered_3[:-3]
            for inject in injection1.split("\n"):
                filtered_3.append(inject)
            filtered_3.append(tmp1)
            filtered_3.append(tmp2)
            filtered_3.append(tmp3)
            flag1 = True
        elif "== decoderselfattn_q" in e and not flag2:
            tmp1 = filtered_2[idx - 3]
            tmp2 = filtered_2[idx - 2]
            tmp3 = filtered_2[idx - 1]
            filtered_3 = filtered_3[:-3]
            for inject in injection2.split("\n"):
                filtered_3.append(inject)
            filtered_3.append(tmp1)
            filtered_3.append(tmp2)
            filtered_3.append(tmp3)
            flag2 = True
        elif "== decodercrossattn_q" in e and not flag3:
            tmp1 = filtered_2[idx - 3]
            tmp2 = filtered_2[idx - 2]
            tmp3 = filtered_2[idx - 1]
            filtered_3 = filtered_3[:-3]
            for inject in injection3.split("\n"):
                filtered_3.append(inject)
            filtered_3.append(tmp1)
            filtered_3.append(tmp2)
            filtered_3.append(tmp3)
            flag3 = True
        filtered_3.append(e)

    filtered_4 = []
    for e in filtered_3:
        if "RELU.out" in e.strip():
            tmp = e.split("RELU.out")
            tmp = tmp[0] + "RELU[i0][i1][i2].out;"
            filtered_4.append(tmp)
        elif "RELU2.out" in e.strip():
            tmp = e.split("RELU2.out")
            tmp = tmp[0] + "RELU2[i0][i1][i2].out;"
            filtered_4.append(tmp)
        elif "RELU3.out" in e.strip():
            tmp = e.split("RELU3.out")
            tmp = tmp[0] + "RELU3[i0][i1][i2].out;"
            filtered_4.append(tmp)
        else:
            filtered_4.append(e)

    return "\n".join(filtered_4)
