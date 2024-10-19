use tiktoken_rs::CoreBPE;
use candle_core::{Tensor, Device};

pub struct GPTDatasetV1 {
    tokenizer: CoreBPE,
    input_ids: Vec<Tensor>,
    target_ids: Vec<Tensor>,
}

impl GPTDatasetV1 {
    pub fn new(txt: &str, tokenizer: CoreBPE, max_length: usize, stride: usize) -> GPTDatasetV1 {
        let mut input_ids: Vec<Tensor> = Vec::new();
        let mut target_ids: Vec<Tensor> = Vec::new();
        let token_ids = tokenizer.encode_with_special_tokens(txt);
        let mut i = 0;
        let num_tokens = token_ids.len();
        while i + max_length < num_tokens {
            let input_chunk = &token_ids[i..i+max_length];
            let target_chunk = &token_ids[i+1..max_length];
            input_ids.push(Tensor::from_vec(input_chunk.to_vec(), (1, max_length),&Device::Cpu).unwrap());
            target_ids.push(Tensor::from_vec(target_chunk.to_vec(), (1, max_length),&Device::Cpu).unwrap());
            i += stride;
        };
        GPTDatasetV1 {
            input_ids,
            target_ids,
            tokenizer
        }
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn get(&self, ix: usize) -> (Tensor, Tensor) {
        (self.input_ids[ix].clone(), self.target_ids[ix].clone())
    }
}