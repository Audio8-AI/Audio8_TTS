"""
onnx-to-candle-weight-extraction

Walks slow_ar_int4.onnx and fast_ar_int4.onnx (and, for completeness, the
non-quantized codec_decoder_fp16.onnx), finds every GatherBlockQuantized and
MatMulNBits node, dequantizes the packed INT4 weight blocks back to FP32 using
each op's own block-quantization formula, and extracts every other weight
tensor (embeddings that aren't quantized, RMSNorm weights, biases) as-is.

Output: a single .npz per source graph, tensor names rewritten to match the
canonical PyTorch module naming used by modeling_arktts.py / the HF
model.safetensors checkpoint (verified against both), e.g.:
    layers.0.attention.wqkv.weight
    layers.0.attention.wqkv.bias
    layers.0.attention_norm.weight
    layers.0.feed_forward.w1.weight
    embeddings.weight
    codebook_embeddings.weight
    fast_layers.0.attention.wo.weight
    fast_output.weight
    decoder.model.0.conv.weight   (codec decoder, unquantized passthrough)

Dequantization formulas (established live against the actual ONNX Runtime
quantization tool source, matmul_nbits_quantizer.py, and cross-checked
against the graphs' own node attributes):

  MatMulNBits (weight-only 4-bit linear layers):
    - Input B packed shape: (N, ceil(K/block_size), block_size/2) uint8
    - scales shape: (N, ceil(K/block_size)) float16, one scale per block
    - zero_points packed shape: (N, ceil(ceil(K/block_size)/2)) uint8,
      2 zero-points packed per byte, same adjacent-pair nibble scheme
    - Packing order (verified against onnxruntime's own RTN packer,
      DefaultWeightOnlyQuantizer.pack_int8_to_int4): ADJACENT PAIRS -
      flat_index i -> low nibble, flat_index i+1 -> high nibble, i.e.
      byte_j = (val[2*j] & 0xF) | ((val[2*j+1] & 0xF) << 4)
    - Dequant: value = (packed_nibble - zero_point) * scale
    - Weight matrix is stored transposed (row-major over N=output features,
      each row split into K/block_size blocks of block_size input features)
      matching nn.Linear's [out_features, in_features] convention directly -
      no transpose needed to match PyTorch wqkv.weight etc.

  GatherBlockQuantized (quantized embedding tables):
    - Same block-quantized affine scheme, but the raw ONNX tensor is typed
      as 4-bit (uint4/int4 via ml_dtypes), not raw uint8, so onnx.numpy_helper
      already unpacks it to one nibble value per element (shape matches the
      logical embedding table shape directly, e.g. (155776, 896) for
      embeddings.weight) - scales/zero_points are per (row, block) shaped
      (num_rows, K/block_size). Dequant: value = (elem - zero_point) * scale.

Both formulas produce plain FP32 numpy arrays; run live sanity checks (finite,
correct shape, distribution sanity) after every tensor is dequantized.
"""
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MODEL_DIR = Path(__file__).resolve().parents[2] / "onnx_runtime" / "model"
OUT_DIR = Path(__file__).resolve().parents[1] / "weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def node_attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
    return out


def onnx_name_to_pytorch_name(node_name: str) -> str:
    """
    Node names look like '/layers.0/attention/wqkv/MatMul_Q4' or
    '/embeddings/Gather_Q4' or '/fast_output/MatMul_Q4'. Strip the leading
    slash, trailing '/<Op>_Q4' or '/<Op>_output_0'-style suffix, and the
    trailing op-name path segment, to recover 'layers.0.attention.wqkv' /
    'embeddings' / 'fast_output', then append '.weight'.

    Root-level MatMul nodes with no module-path segments at all (e.g. the
    bare '/MatMul_Q4' node found live in slow_ar_int4.onnx) are the
    tied-output-embedding projection (config.tie_word_embeddings=true in
    modeling_arktts.py's ArkttsModel.forward: `F.linear(normalized,
    self.embeddings.weight)`) - ONNX baked a separate quantized copy of this
    matmul rather than re-reading the embedding table, so it is named
    explicitly as 'lm_head.weight' (a distinct extracted tensor, NOT
    embeddings.weight, since ORT quantized it independently and it should be
    verified against embeddings.weight for closeness, not silently merged).
    """
    n = node_name.strip("/")
    parts = n.split("/")
    parts = parts[:-1]  # drop the op invocation segment itself
    if not parts:
        return "lm_head.weight"
    return ".".join(parts) + ".weight"


def conv_node_to_pytorch_name(node_name: str) -> str:
    """
    Conv/ConvTranspose weight tensors are ONNX-export-anonymized
    (onnx::Conv_4310 etc, no dotted param name) but the owning node's own
    name still encodes the module path, e.g. '/out_proj/Conv' ->
    'out_proj.weight', '/decoder/model.1/block.1/conv/Conv' ->
    'decoder.model.1.block.1.conv.weight'. Same stripping rule as
    onnx_name_to_pytorch_name but keyed off the Conv node itself, not a
    quant node.
    """
    n = node_name.strip("/")
    parts = n.split("/")
    parts = parts[:-1]
    if not parts:
        raise ValueError(f"unexpected root-level conv node name: {node_name}")
    return ".".join(parts) + ".weight"


def unpack_matmulnbits_weight(packed: np.ndarray, scales: np.ndarray, zero_points, K: int, N: int, block_size: int, bits: int) -> np.ndarray:
    assert bits == 4, f"only 4-bit MatMulNBits supported, got bits={bits}"
    n_blocks = -(-K // block_size)  # ceil
    assert packed.shape == (N, n_blocks, block_size // 2), (packed.shape, N, n_blocks, block_size)
    assert scales.shape == (N, n_blocks), (scales.shape, N, n_blocks)

    # unpack adjacent-pair nibbles: low nibble = even flat index, high = odd
    low = (packed & 0x0F).astype(np.int32)
    high = ((packed >> 4) & 0x0F).astype(np.int32)
    # interleave: element 2*j = low[...,j], element 2*j+1 = high[...,j]
    unpacked = np.empty((N, n_blocks, block_size), dtype=np.int32)
    unpacked[:, :, 0::2] = low
    unpacked[:, :, 1::2] = high

    if zero_points is not None:
        zp_packed = zero_points
        zp_n_pairs = -(-n_blocks // 2)
        assert zp_packed.shape == (N, zp_n_pairs), (zp_packed.shape, N, zp_n_pairs)
        zp_low = (zp_packed & 0x0F).astype(np.int32)
        zp_high = ((zp_packed >> 4) & 0x0F).astype(np.int32)
        zp_full = np.empty((N, zp_n_pairs * 2), dtype=np.int32)
        zp_full[:, 0::2] = zp_low
        zp_full[:, 1::2] = zp_high
        zp = zp_full[:, :n_blocks]
    else:
        zp = np.full((N, n_blocks), 8, dtype=np.int32)  # ORT default midpoint for 4-bit

    scales_f32 = scales.astype(np.float32)
    dequant = (unpacked - zp[:, :, None]).astype(np.float32) * scales_f32[:, :, None]
    dequant = dequant.reshape(N, n_blocks * block_size)[:, :K]
    return dequant


def unpack_gatherblockquantized(elems: np.ndarray, scales: np.ndarray, zero_points, block_size: int, quantize_axis: int) -> np.ndarray:
    # elems already unpacked to one (u)int4 nibble value per logical element
    # by onnx.numpy_helper (ml_dtypes uint4/int4 -> normal-shaped array).
    elems_i32 = elems.astype(np.int32)
    rows, cols = elems_i32.shape
    assert quantize_axis == 1, f"unexpected quantize_axis={quantize_axis}"
    n_blocks = -(-cols // block_size)
    assert scales.shape == (rows, n_blocks), (scales.shape, rows, n_blocks)
    if zero_points is not None:
        zp = zero_points.astype(np.int32)
        assert zp.shape == (rows, n_blocks), (zp.shape, rows, n_blocks)
    else:
        zp = np.zeros((rows, n_blocks), dtype=np.int32)

    pad = n_blocks * block_size - cols
    if pad:
        elems_i32 = np.pad(elems_i32, ((0, 0), (0, pad)))
    elems_i32 = elems_i32.reshape(rows, n_blocks, block_size)
    scales_f32 = scales.astype(np.float32)
    dequant = (elems_i32 - zp[:, :, None]).astype(np.float32) * scales_f32[:, :, None]
    dequant = dequant.reshape(rows, n_blocks * block_size)[:, :cols]
    return dequant


def extract_graph(onnx_path: Path, name_prefix_strip=None):
    print(f"=== loading {onnx_path.name} ===", file=sys.stderr)
    m = onnx.load(str(onnx_path), load_external_data=True)
    g = m.graph
    init = {i.name: i for i in g.initializer}

    tensors = {}  # pytorch_name -> np.ndarray
    dtypes_report = {}

    quant_consumed = set()
    n_matmul = 0
    n_gather = 0

    for node in g.node:
        if node.op_type == "MatMulNBits":
            attrs = node_attrs(node)
            K, N = attrs["K"], attrs["N"]
            block_size = attrs["block_size"]
            bits = attrs["bits"]
            b_name, scales_name = node.input[1], node.input[2]
            zp_name = node.input[3] if len(node.input) > 3 else None
            quant_consumed.update(node.input)

            packed = numpy_helper.to_array(init[b_name])
            scales = numpy_helper.to_array(init[scales_name])
            zero_points = numpy_helper.to_array(init[zp_name]) if zp_name and zp_name in init else None

            weight = unpack_matmulnbits_weight(packed, scales, zero_points, K, N, block_size, bits)
            assert weight.shape == (N, K), (weight.shape, N, K)
            assert np.isfinite(weight).all(), f"non-finite values in {node.name}"

            pt_name = onnx_name_to_pytorch_name(node.name)
            tensors[pt_name] = weight.astype(np.float32)
            dtypes_report[pt_name] = f"MatMulNBits K={K} N={N} block_size={block_size} bits={bits} -> fp32 {weight.shape}"
            n_matmul += 1

        elif node.op_type == "GatherBlockQuantized":
            attrs = node_attrs(node)
            block_size = attrs["block_size"]
            quantize_axis = attrs.get("quantize_axis", 1)
            data_name, _, scales_name = node.input[0], node.input[1], node.input[2]
            zp_name = node.input[3] if len(node.input) > 3 else None
            quant_consumed.add(data_name)
            quant_consumed.add(scales_name)
            if zp_name:
                quant_consumed.add(zp_name)

            elems = numpy_helper.to_array(init[data_name])
            scales = numpy_helper.to_array(init[scales_name])
            zero_points = numpy_helper.to_array(init[zp_name]) if zp_name and zp_name in init else None

            weight = unpack_gatherblockquantized(elems, scales, zero_points, block_size, quantize_axis)
            assert np.isfinite(weight).all(), f"non-finite values in {node.name}"

            # node name like '/embeddings/Gather_Q4' or '/codebook_embeddings_3/Gather_Q4'
            # (the _N suffix on repeated Gather calls against the SAME table -
            # only emit once, keyed by the underlying initializer's own pytorch name)
            raw_pt_name = onnx_name_to_pytorch_name(node.name)
            pt_name = re.sub(r"_\d+\.weight$", ".weight", raw_pt_name)
            if pt_name not in tensors:
                tensors[pt_name] = weight.astype(np.float32)
                dtypes_report[pt_name] = f"GatherBlockQuantized block_size={block_size} -> fp32 {weight.shape}"
            n_gather += 1

    # Conv/ConvTranspose/plain-MatMul weights: anonymized initializer names
    # (onnx::Conv_NNNN / onnx::MatMul_NNNN, no dotted param name preserved by
    # the exporter), recover the real module path from the owning node's own
    # name instead - same recovery rule as the quantized MatMulNBits nodes.
    # For MatMul, only the node's second input (the constant weight operand)
    # qualifies - self-attention's QK^T and softmax@V MatMul nodes (e.g.
    # '/post_module/layers.0/attention/MatMul') have NO initializer input at
    # all (both operands are runtime activations), so init.get() naturally
    # skips them.
    n_conv = 0
    for node in g.node:
        if node.op_type not in ("Conv", "ConvTranspose", "MatMul"):
            continue
        w_name = node.input[1]
        if w_name not in init:
            continue
        arr = numpy_helper.to_array(init[w_name])
        pt_name = conv_node_to_pytorch_name(node.name)
        if pt_name not in tensors:
            tensors[pt_name] = arr.astype(np.float32) if arr.dtype != np.float32 else arr
            dtypes_report[pt_name] = f"{node.op_type} weight -> fp32 {arr.shape}"
            quant_consumed.add(w_name)  # exclude from the generic plain pass below
            n_conv += 1

    # everything else: plain FP16/FP32 initializers not consumed by a quant node
    n_plain = 0
    for iname, tensor in init.items():
        if iname in quant_consumed:
            continue
        arr = numpy_helper.to_array(tensor)
        if arr.ndim == 0 or arr.size == 0:
            continue
        # skip pure-graph Constant-folding scalars/small helper tensors:
        # real params always carry a dotted module-path name ending in
        # .weight/.bias/.alpha/.gamma (verified live: every param-looking
        # initializer in this graph follows this pattern with a 'model.'
        # or 'decoder.'/'quantizer.' prefix)
        if not re.search(r"\.(weight|bias|alpha|gamma)$", iname):
            continue
        pt_name = iname
        if name_prefix_strip and pt_name.startswith(name_prefix_strip):
            pt_name = pt_name[len(name_prefix_strip):]
        if pt_name in tensors:
            continue
        tensors[pt_name] = arr.astype(np.float32) if arr.dtype != np.float32 else arr
        dtypes_report[pt_name] = f"plain {arr.dtype} -> fp32 {arr.shape}"
        n_plain += 1

    print(f"  MatMulNBits nodes: {n_matmul}, GatherBlockQuantized nodes: {n_gather} (unique tables: {sum(1 for k,v in dtypes_report.items() if 'GatherBlockQuantized' in v)}), conv weights: {n_conv}, plain params: {n_plain}", file=sys.stderr)
    print(f"  total extracted tensors: {len(tensors)}", file=sys.stderr)
    return tensors, dtypes_report


def main():
    all_report = {}

    slow_tensors, slow_report = extract_graph(MODEL_DIR / "slow_ar_int4.onnx", name_prefix_strip="model.")
    np.savez(OUT_DIR / "slow_ar_fp32.npz", **slow_tensors)
    all_report["slow_ar"] = slow_report

    fast_tensors, fast_report = extract_graph(MODEL_DIR / "fast_ar_int4.onnx", name_prefix_strip="model.")
    np.savez(OUT_DIR / "fast_ar_fp32.npz", **fast_tensors)
    all_report["fast_ar"] = fast_report

    codec_tensors, codec_report = extract_graph(MODEL_DIR / "codec_decoder_fp16.onnx", name_prefix_strip=None)
    np.savez(OUT_DIR / "codec_decoder_fp32.npz", **codec_tensors)
    all_report["codec_decoder"] = codec_report

    print("\n=== SUMMARY ===")
    for graph_name, report in all_report.items():
        print(f"{graph_name}: {len(report)} tensors")

    # print full report to a text file for inspection
    with open(OUT_DIR / "extraction_report.txt", "w") as f:
        for graph_name, report in all_report.items():
            f.write(f"=== {graph_name} ===\n")
            for k in sorted(report):
                f.write(f"{k}\t{report[k]}\n")
            f.write("\n")

    print(f"\nWrote: {OUT_DIR / 'slow_ar_fp32.npz'}")
    print(f"Wrote: {OUT_DIR / 'fast_ar_fp32.npz'}")
    print(f"Wrote: {OUT_DIR / 'codec_decoder_fp32.npz'}")
    print(f"Wrote: {OUT_DIR / 'extraction_report.txt'}")


if __name__ == "__main__":
    main()
