use candle_core::{Device, Tensor};
use candle_nn::{Embedding, Module};
use rust_gpt::dataset::{self, GPTDatasetV1};
use std::fs;
use tiktoken_rs::r50k_base;
fn main() {
    let file_content =
        fs::read_to_string("./data/the-verdict.txt").expect("the file should be present");
    let bpe = r50k_base().unwrap();
    let encoded_file = bpe.encode_with_special_tokens(&file_content[..]);
    let max_length = 4;
    let mut dataloader =
        dataset::create_dataloader_v1(&file_content, Some(8), Some(max_length), Some(4), None, None);

    let device = Device::Cpu;
    let output_dim = 256;
    let vocab_size = 50257;
    let t_embeddings = Tensor::randn(0f32, 1.0, (vocab_size, output_dim), &device).unwrap();
    let t_embeddings = Embedding::new(t_embeddings, output_dim);
    let p_embeddings = Tensor::randn(0f32, 1.0, (max_length, output_dim), &device).unwrap();
    let p_embeddings = Embedding::new(p_embeddings, output_dim);
    if let Some(Ok((inputs, targets))) = dataloader.next() {
        // println!("{inputs:?}, {targets:?}");
        // println!("inputs: {}\n outputs: {}", inputs, targets);
        let token_embeddings = t_embeddings.forward(&inputs).unwrap();
        let range = Tensor::arange(0u32, max_length as u32, &device).unwrap();
        let pos_embeddings = p_embeddings.forward(&range).unwrap();
        println!("{:?}, {:?}", token_embeddings , pos_embeddings);
        let e = (token_embeddings.broadcast_add(&pos_embeddings)).unwrap();
        println!("{}", e);
    }

}
