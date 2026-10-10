use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{self, Read, Seek};

#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub magic: String,
    pub version: u32,
    pub tensor_count: u64,
    pub mdkv_count: u64,
}

impl GgufHeader {
    pub fn read<R: Read + Seek>(reader: &mut R) -> io::Result<Self> {
        let mut magic_bytes = [0u8; 4];
        reader.read_exact(&mut magic_bytes)?;

        let magic = String::from_utf8_lossy(&magic_bytes).to_string();
        if magic != "GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Expected GGUF magic header, found: {}", magic),
            ));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let mdkv_count = reader.read_u64::<LittleEndian>()?;

        Ok(Self {
            magic,
            version,
            tensor_count,
            mdkv_count,
        })
    }
}
