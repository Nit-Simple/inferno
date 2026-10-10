use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{self, Read};

#[derive(Debug, Clone, PartialEq)]
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
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

pub fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

pub fn read_value<R: Read>(reader: &mut R, value_type: u32) -> io::Result<GgufValue> {
    match value_type {
        0 => Ok(GgufValue::Uint8(reader.read_u8()?)),
        1 => Ok(GgufValue::Int8(reader.read_i8()?)),
        2 => Ok(GgufValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufValue::Int16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufValue::Int32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufValue::Float32(reader.read_f32::<LittleEndian>()?)),
        7 => Ok(GgufValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(GgufValue::String(read_string(reader)?)),
        9 => {
            let inner_type = reader.read_u32::<LittleEndian>()?;
            let len = reader.read_u64::<LittleEndian>()? as usize;
            let mut array = Vec::with_capacity(len);
            for _ in 0..len {
                array.push(read_value(reader, inner_type)?);
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
