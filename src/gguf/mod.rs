pub mod header;
pub mod model;
pub mod reader;
pub mod tensor;
pub mod value;

use std::collections::HashMap;
use std::io;

pub use header::GgufHeader;
pub use model::Modelview;
pub use reader::GgufReader;
pub use tensor::{TensorInfo, Tensors};
pub use value::{GgufValue, read_value};

/// Loads the metadata key-value pairs described in the GGUF header.
pub fn load_metadata<'a>(
    reader: &mut GgufReader<'a>,
    header: &GgufHeader<'a>,
) -> io::Result<HashMap<&'a str, GgufValue<'a>>> {
    let mut mdkv_map = HashMap::with_capacity(header.mdkv_count as usize);

    for _ in 0..header.mdkv_count {
        let key = reader.read_str()?;
        let value_type = reader.read_u32()?;
        let value = read_value(reader, value_type)?;

        mdkv_map.insert(key, value);
    }

    Ok(mdkv_map)
}
