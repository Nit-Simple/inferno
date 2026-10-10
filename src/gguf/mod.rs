pub mod header;
pub mod tensor;
pub mod value;
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{self, Read};

pub use header::GgufHeader;
pub use tensor::{TensorInfo, Tensors};
pub use value::{GgufValue, read_string, read_value};

/// Loads the metadata key-value pairs described in the GGUF header.
pub fn load_metadata<R: Read>(
    reader: &mut R,
    header: &GgufHeader,
) -> io::Result<HashMap<String, GgufValue>> {
    let mut mdkv_map = HashMap::with_capacity(header.mdkv_count as usize);

    for _ in 0..header.mdkv_count {
        let key = read_string(reader)?;
        let value_type = reader.read_u32::<LittleEndian>()?;
        let value = read_value(reader, value_type)?;

        mdkv_map.insert(key, value);
    }

    Ok(mdkv_map)
}
