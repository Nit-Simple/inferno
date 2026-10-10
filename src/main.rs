use std::fs::File;
use std::io;

use inferno_core::gguf::{GgufHeader, Tensors, load_metadata};

fn main() -> io::Result<()> {
    let mut file = File::open("../gguf/model.gguf")?;
    let header = GgufHeader::read(&mut file)?;

    println!("Magic: {}", header.magic);
    println!("Format Version: {}", header.version);
    println!("Total Tensors (Weight matrices): {}", header.tensor_count);
    println!("Metadata Entries: {}", header.mdkv_count);

    let mdkv = load_metadata(&mut file, &header)?;

    println!("\nLoaded {} metadata entries.", mdkv.len());

    let tensors = Tensors::read(&mut file, header.tensor_count, &mdkv)?;
    println!("Loaded {} tensors.", tensors.tensors.len());

    // Print the first 5 tensors as a sample
    for tensor in tensors.tensors.iter().take(5) {
        println!(
            "Tensor: {:<35} | Shape: {:?} | Type: {} | Offset: {}",
            tensor.name, tensor.shape, tensor.ggml_type, tensor.offset
        );
    }
    println!("{:?}", tensors.data_offset);
    Ok(())
}
