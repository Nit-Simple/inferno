use std::collections::HashMap;
use std::io;

use super::reader::GgufReader;
use super::value::GgufValue;

#[derive(Debug, Clone)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    pub shape: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct Tensors<'a> {
    pub tensors: Vec<TensorInfo<'a>>,
    pub data_offset: u64,
}

impl<'a> Tensors<'a> {
    /// Constructs and reads all tensor metadata entries from GgufReader.
    pub fn read(
        reader: &mut GgufReader<'a>,
        tensor_count: u64,
        mdkv_map: &HashMap<&'a str, GgufValue<'a>>,
    ) -> io::Result<Self> {
        let mut tensors = Vec::with_capacity(tensor_count as usize);

        for _ in 0..tensor_count {
            let name = reader.read_str()?;
            let n_dims = reader.read_u32()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(reader.read_u64()?);
            }
            let ggml_type = reader.read_u32()?;
            let offset = reader.read_u64()?;

            tensors.push(TensorInfo {
                name,
                shape,
                ggml_type,
                offset,
            });
        }

        let current_position = reader.cursor() as u64;

        let alignment = match mdkv_map.get("general.alignment") {
            Some(GgufValue::Uint32(a)) => *a as u64,
            _ => 32,
        };
        // Calculates the data offset aligned to `alignment` bytes
        let data_offset = (current_position + alignment - 1) & !(alignment - 1);

        Ok(Self {
            tensors,
            data_offset,
        })
    }
}
