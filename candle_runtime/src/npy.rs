use std::io::Read;
use std::path::Path;

/// Minimal reader for a 2D uint16, little-endian, C-order .npy file - exactly
/// what onnx_runtime voice registration writes (codes.npy, dtype uint16 per
/// meta.json). Not a general .npy parser. Ported verbatim from
/// rust_runtime/src/npy.rs (zero ort dependency, pure I/O).
pub fn read_npy_u16_2d(path: &Path) -> anyhow::Result<ndarray::Array2<u16>> {
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY" {
        anyhow::bail!("not a .npy file: {:?}", path);
    }
    let mut version = [0u8; 2];
    file.read_exact(&mut version)?;

    let header_len = if version[0] == 1 {
        let mut len_bytes = [0u8; 2];
        file.read_exact(&mut len_bytes)?;
        u16::from_le_bytes(len_bytes) as usize
    } else {
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)?;
        u32::from_le_bytes(len_bytes) as usize
    };

    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let header_str = String::from_utf8_lossy(&header);

    if !header_str.contains("'<u2'") && !header_str.contains("uint16") {
        anyhow::bail!("expected uint16 dtype, header was: {}", header_str);
    }

    let shape = parse_shape(&header_str)?;
    if shape.len() != 2 {
        anyhow::bail!("expected 2D array, got shape {:?}", shape);
    }

    let total = shape[0] * shape[1];
    let mut raw = vec![0u8; total * 2];
    file.read_exact(&mut raw)?;

    let values: Vec<u16> = raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    Ok(ndarray::Array2::from_shape_vec((shape[0], shape[1]), values)?)
}

fn parse_shape(header: &str) -> anyhow::Result<Vec<usize>> {
    let key = "'shape':";
    let start = header.find(key).ok_or_else(|| anyhow::anyhow!("no shape in npy header"))?;
    let rest = &header[start + key.len()..];
    let paren_start = rest.find('(').ok_or_else(|| anyhow::anyhow!("no shape tuple"))?;
    let paren_end = rest.find(')').ok_or_else(|| anyhow::anyhow!("unterminated shape tuple"))?;
    let inner = &rest[paren_start + 1..paren_end];
    let dims: Result<Vec<usize>, _> = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect();
    Ok(dims?)
}
