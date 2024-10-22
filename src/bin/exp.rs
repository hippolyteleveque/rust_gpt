use std::fmt::Error;

use candle_core::{Device, Tensor};
use candle_nn::{Embedding, Module};

fn main() {
    let device = Device::Cpu;
    let test = Tensor::from_vec(vec![1u32, 2u32, 3u32], (1, 3), &device).unwrap();
    println!("{test}");
    let embeddings = Tensor::randn(0f32, 1.0, (100, 784), &device).unwrap();
    println!("{embeddings}");
    let embeddings = Embedding::new(embeddings, 784);
    let res = embeddings.forward(&test).unwrap();
    println!("{res}");
}
