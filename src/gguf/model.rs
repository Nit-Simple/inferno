use std::collections::HashMap;

use crate::gguf::{GgufHeader, GgufValue, TensorInfo, Tensors};

#[derive(Debug, Clone)]
pub struct Modelview<'a> {
    pub header: &'a GgufHeader<'a>,
    pub tensors: &'a Tensors<'a>,
    pub mdkv_map: &'a HashMap<&'a str, GgufValue<'a>>,
    pub strt_data_offset: usize,
    pub mmp_slice: &'a [u8],
}

impl<'a> Modelview<'a> {
    pub fn new(
        header: &'a GgufHeader,
        tensors: &'a Tensors<'a>,
        mdkv_map: &'a HashMap<&'a str, GgufValue<'a>>,
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

    /// Returns a slice of the raw weight data for the given tensor
    pub fn tensor_data(&self, tensor: &TensorInfo<'a>) -> &'a [u8] {
        let offset = self.strt_data_offset + tensor.offset as usize;
        &self.mmp_slice[offset..]
    }
}
