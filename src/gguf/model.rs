use std::collections::HashMap;
use std::io::{self, Read};
use std::io::{Seek, SeekFrom};

use crate::gguf::GgufValue;
use crate::gguf::Tensors;
use crate::gguf::{GgufHeader, header};

#[derive(Debug, Clone)]
pub struct Modelview<'a> {
    pub header: &'a GgufHeader,
    pub tensors: &'a Tensors,
    pub mdkv_map: &'a HashMap<String, GgufValue>,
    pub strt_data_offset: usize,
    pub mmp_slice: &'a [u8],
}

impl<'a> Modelview<'a> {
    pub fn new(
        header: &'a GgufHeader,
        tensors: &'a Tensors,
        mdkv_map: &'a HashMap<String, GgufValue>,
        strt_offset: usize,
        mmp_slice: &'a [u8],
    ) -> Self {
        Modelview {
            header,
            tensors,
            mdkv_map,
            strt_data_offset: strt_offset,
            mmp_slice,
        }
    }
}
