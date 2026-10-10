use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, Read};
use std::io::{Seek, SeekFrom};

use crate::gguf::GgufValue;

use super::read_string;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct Tensors {
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
}

impl Tensors {
    /// Constructs and reads all tensor metadata entries from the reader.
    pub fn read<R: Read + Seek>(
        reader: &mut R,
        tensor_count: u64,
        mdkv_map: &HashMap<String, GgufValue>,
    ) -> io::Result<Self> {
        let mut tensors = Vec::with_capacity(tensor_count as usize);

        for _ in 0..tensor_count {
            let name = read_string(reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(reader.read_u64::<LittleEndian>()?);
            }
            let ggml_type = reader.read_u32::<LittleEndian>()?;
            let offset = reader.read_u64::<LittleEndian>()?;

            tensors.push(TensorInfo {
                name,
                shape,
                ggml_type,
                offset,
            });
        }

        let current_position = reader.stream_position()?;

        let alignment = match mdkv_map.get("general.alignment") {
            Some(GgufValue::Uint32(a)) => *a as u64,
            _ => 32,
        };

        let data_offset = (current_position + alignment - 1) & !(alignment - 1);

        Ok(Self {
            tensors,
            data_offset,
        })
    }
}
