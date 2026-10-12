use std::io::{self, Error, ErrorKind};

#[derive(Debug, Clone)]
pub struct GgufReader<'a> {
    buffer: &'a [u8],
    cursor: usize,
}

impl<'a> GgufReader<'a> {
    pub fn new(buffer: &'a [u8]) -> Self {
        Self { buffer, cursor: 0 }
    }

    /// Current cursor position in bytes
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Remaining bytes in the buffer
    pub fn remaining(&self) -> usize {
        self.buffer.len().saturating_sub(self.cursor)
    }

    /// Total buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether buffer has reached the end
    pub fn is_empty(&self) -> bool {
        self.cursor >= self.buffer.len()
    }

    /// Sets cursor position directly
    pub fn seek(&mut self, pos: usize) -> io::Result<()> {
        if pos > self.buffer.len() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "Seek position {} exceeds buffer size {}",
                    pos,
                    self.buffer.len()
                ),
            ));
        }
        self.cursor = pos;
        Ok(())
    }

    /// Advances the cursor to the next multiple of `alignment`
    pub fn align_to(&mut self, alignment: usize) -> io::Result<()> {
        if alignment == 0 {
            return Ok(());
        }
        let remainder = self.cursor % alignment;
        if remainder != 0 {
            let next = self.cursor + (alignment - remainder);
            self.seek(next)?;
        }
        Ok(())
    }

    /// Reads exactly `len` bytes zero-copy, advancing the cursor
    pub fn read_bytes(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self.cursor + len;
        if end > self.buffer.len() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "Unexpected EOF: needed {} bytes at offset {}, buffer has {} bytes",
                    len,
                    self.cursor,
                    self.buffer.len()
                ),
            ));
        }
        let slice = &self.buffer[self.cursor..end];
        self.cursor = end;
        Ok(slice)
    }

    /// Reads a 64-bit length-prefixed UTF-8 string zero-copy (&'a str)
    pub fn read_str(&mut self) -> io::Result<&'a str> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes).map_err(|e| Error::new(ErrorKind::InvalidData, e))
    }

    pub fn read_u8(&mut self) -> io::Result<u8> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0])
    }

    pub fn read_i8(&mut self) -> io::Result<i8> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0] as i8)
    }

    pub fn read_u16(&mut self) -> io::Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_i16(&mut self) -> io::Result<i16> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_u32(&mut self) -> io::Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_i32(&mut self) -> io::Result<i32> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_f32(&mut self) -> io::Result<f32> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_u64(&mut self) -> io::Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_i64(&mut self) -> io::Result<i64> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_f64(&mut self) -> io::Result<f64> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().unwrap()))
    }

    pub fn read_bool(&mut self) -> io::Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    /// Access the underlying buffer slice
    pub fn buffer(&self) -> &'a [u8] {
        self.buffer
    }
}
