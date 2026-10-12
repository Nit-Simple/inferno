use std::fs::File;
use std::io;

use inferno_core::gguf::{GgufHeader, GgufReader, Modelview, Tensors, load_metadata};
use memmap2::{Mmap, MmapOptions};

fn main() -> io::Result<()> {
    let file = File::open("../gguf/model.gguf")?;
    let mmap = unsafe { MmapOptions::new().populate().map(&file)? };

    //tells the kernel that we are reading linearly and we dont need breakodd in between
    mmap.advise(memmap2::Advice::Sequential)?;

    #[cfg(target_os = "linux")]
    //use large pages to read the data from
    let _ = mmap.advise(memmap2::Advice::HugePage);
    let mut reader = GgufReader::new(&mmap);

    let header = GgufHeader::read(&mut reader)?;
    println!("Magic: {}", header.magic);
    println!("Format Version: {}", header.version);
    println!("Total Tensors (Weight matrices): {}", header.tensor_count);
    println!("Metadata Entries: {}", header.mdkv_count);

    let mdkv = load_metadata(&mut reader, &header)?;
    println!("\nLoaded {} metadata entries (zero-copy).", mdkv.len());

    let tensors = Tensors::read(&mut reader, header.tensor_count, &mdkv)?;
    println!("Loaded {} tensors (zero-copy).", tensors.tensors.len());

    // Print the first 5 tensors as a sample
    for tensor in tensors.tensors.iter().take(5) {
        println!(
            "Tensor: {:<35} | Shape: {:?} | Type: {} | Offset: {}",
            tensor.name, tensor.shape, tensor.ggml_type, tensor.offset
        );
    }
    println!("Tensors data offset: {}", tensors.data_offset);

    let start_offset = tensors.data_offset as usize;

    let model_view = Modelview::new(&header, &tensors, &mdkv, start_offset, &mmap);
    println!(
        "Modelview initialized. Total mapped size: {} bytes, Data offset: {}",
        model_view.mmp_slice.len(),
        model_view.strt_data_offset
    );

    Ok(())
}
