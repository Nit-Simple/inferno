use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek};
use std::vec;

#[derive(Debug)]
pub struct GgufHeader {
    pub magic: String,
    pub version: u32,
    pub tensor_count: u64,
    pub mdkv_count: u64,
}

#[derive(Debug)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>), // Arrays can hold other GgufValues
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufHeader {
    fn read<R: Read + Seek>(reader: &mut R) -> io::Result<Self> {
        let mut magic_bytes = [0u8; 4];

        reader.read_exact(&mut magic_bytes)?;

        let magic = String::from_utf8_lossy(&magic_bytes).to_string();
        if magic != "GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Expected GGUF magic header, found: {}", magic),
            ));
        };
        let version = reader.read_u32::<LittleEndian>()?;

        // 3. Read tensor count (u64, little-endian)
        let tensor_count = reader.read_u64::<LittleEndian>()?;

        // 4. Read metadata key-value count (u64, little-endian)
        let mdkv_count = reader.read_u64::<LittleEndian>()?;

        Ok(Self {
            magic,
            version,
            tensor_count,
            mdkv_count,
        })
    }
}

fn readgguf_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}
fn readgguf_value<R: Read>(reader: &mut R, value_type: u32) -> io::Result<GgufValue> {
    match value_type {
        0 => Ok(GgufValue::Uint8(reader.read_u8()?)),
        1 => Ok(GgufValue::Int8(reader.read_i8()?)),
        2 => Ok(GgufValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufValue::Int16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufValue::Int32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufValue::Float32(reader.read_f32::<LittleEndian>()?)),
        7 => Ok(GgufValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(GgufValue::String(readgguf_string(reader)?)),
        9 => {
            // Arrays are special: they store the inner type, the length, then the items
            let inner_type = reader.read_u32::<LittleEndian>()?;
            let len = reader.read_u64::<LittleEndian>()? as usize;
            let mut array = Vec::with_capacity(len);
            for _ in 0..len {
                array.push(readgguf_value(reader, inner_type)?);
            }
            Ok(GgufValue::Array(array))
        }
        10 => Ok(GgufValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(GgufValue::Int64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(GgufValue::Float64(reader.read_f64::<LittleEndian>()?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown metadata value type: {}", value_type),
        )),
    }
}

fn loadkv<R: Read>(reader: &mut R, header: &GgufHeader) -> io::Result<HashMap<String, GgufValue>> {
    let mut mdkv_map = HashMap::new();

    for _ in 0..header.mdkv_count {
        let key = readgguf_string(reader)?;

        let value_type = reader.read_u32::<LittleEndian>()?;

        let value = readgguf_value(reader, value_type)?;

        mdkv_map.insert(key, value);
    }
    Ok(mdkv_map)
}

fn main() -> io::Result<()> {
    let mut file = File::open("../gguf/model.gguf").unwrap();
    let header = GgufHeader::read(&mut file)?;
    println!("Magic: {}", header.magic);
    println!("Format Version: {}", header.version);
    println!("Total Tensors (Weight matrices): {}", header.tensor_count);
    println!("Metadata Entries: {}", header.mdkv_count);
    let mdkv = loadkv(&mut file, &header)?;

    for (key, value) in &mdkv {
        match value {
            GgufValue::Array(arr) => {
                println!("key : {} == Array({} items)", key, arr.len());
            }
            GgufValue::String(s) if s.len() > 80 => {
                println!("key : {} == String(\"{}...\")", key, &s[..80]);
            }
            _ => println!("key : {} == {:?}", key, value),
        }
    }

    Ok(())
}
