use std::io::{self, Error, ErrorKind};
use super::reader::GgufReader;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue<'a> {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(&'a str),
    Array(Vec<GgufValue<'a>>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

pub fn read_value<'a>(reader: &mut GgufReader<'a>, value_type: u32) -> io::Result<GgufValue<'a>> {
    match value_type {
        0 => Ok(GgufValue::Uint8(reader.read_u8()?)),
        1 => Ok(GgufValue::Int8(reader.read_i8()?)),
        2 => Ok(GgufValue::Uint16(reader.read_u16()?)),
        3 => Ok(GgufValue::Int16(reader.read_i16()?)),
        4 => Ok(GgufValue::Uint32(reader.read_u32()?)),
        5 => Ok(GgufValue::Int32(reader.read_i32()?)),
        6 => Ok(GgufValue::Float32(reader.read_f32()?)),
        7 => Ok(GgufValue::Bool(reader.read_bool()?)),
        8 => Ok(GgufValue::String(reader.read_str()?)),
        9 => {
            let inner_type = reader.read_u32()?;
            let len = reader.read_u64()? as usize;
            let mut array = Vec::with_capacity(len);
            for _ in 0..len {
                array.push(read_value(reader, inner_type)?);
            }
            Ok(GgufValue::Array(array))
        }
        10 => Ok(GgufValue::Uint64(reader.read_u64()?)),
        11 => Ok(GgufValue::Int64(reader.read_i64()?)),
        12 => Ok(GgufValue::Float64(reader.read_f64()?)),
        _ => Err(Error::new(
            ErrorKind::InvalidData,
            format!("Unknown metadata value type: {}", value_type),
        )),
    }
}
