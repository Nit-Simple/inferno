use super::reader::GgufReader;
use std::io::{self, Error, ErrorKind};

#[derive(Debug, Clone)]
pub struct GgufHeader<'a> {
    pub magic: &'a str,
    pub version: u32,
    pub tensor_count: u64,
    pub mdkv_count: u64,
}

impl<'a> GgufHeader<'a> {
    pub fn read(reader: &mut GgufReader<'a>) -> io::Result<Self> {
        let magic_bytes = reader.read_bytes(4)?;
        let magic = std::str::from_utf8(magic_bytes).unwrap();
        if magic != "GGUF" {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Expected GGUF magic header, found: {}", magic),
            ));
        }

        let version = reader.read_u32()?;
        let tensor_count = reader.read_u64()?;
        let mdkv_count = reader.read_u64()?;

        Ok(Self {
            magic,
            version,
            tensor_count,
            mdkv_count,
        })
    }
}
