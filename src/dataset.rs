use ai_dataloader::indexable::DataLoader;
use candle_core::{Device, Tensor};
use tiktoken_rs::CoreBPE;
// use ai_dataloader::indexable::DataLoader;
// use ai_dataloader::collate::{DefaultCollate, Collate};
use candle_core::Error;
use candle_datasets::{Batcher, batcher::IterResult2};
use std::vec::IntoIter;
use tiktoken_rs::r50k_base;
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
            let input_chunk = &token_ids[i..i + max_length];
            let target_chunk = &token_ids[i + 1..i + max_length + 1];
            input_ids.push(
                Tensor::from_vec(input_chunk.to_vec(), (1, max_length), &Device::Cpu).unwrap(),
            );
            target_ids.push(
                Tensor::from_vec(target_chunk.to_vec(), (1, max_length), &Device::Cpu).unwrap(),
            );
            i += stride;
        }
        GPTDatasetV1 {
            input_ids,
            target_ids,
            tokenizer,
        }
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn get(&self, ix: usize) -> (Tensor, Tensor) {
        (self.input_ids[ix].clone(), self.target_ids[ix].clone())
    }
}

type GPTIterBatch = std::iter::Map<
    std::iter::Zip<std::vec::IntoIter<Tensor>, std::vec::IntoIter<Tensor>>,
    fn((Tensor, Tensor)) -> Result<(Tensor, Tensor), Error>,
>;

impl IntoIterator for GPTDatasetV1 {
    type Item = Result<(Tensor, Tensor), Error>;
    type IntoIter = GPTIterBatch;

    fn into_iter(self) -> Self::IntoIter {
        self.input_ids
            .into_iter()
            .zip(self.target_ids.into_iter())
            .map(|(input_ids, target_ids)| Ok((input_ids, target_ids)))
    }
}

pub fn create_dataloader_v1(
    txt: &str,
    batch_size: Option<usize>,
    max_length: Option<usize>,
    stride: Option<usize>,
    shuffle: Option<bool>,
    drop_last: Option<bool>,
) -> Batcher<IterResult2<GPTIterBatch>> {
    let batch_size = batch_size.unwrap_or(4);
    let max_length = max_length.unwrap_or(256);
    let stride = stride.unwrap_or(128);
    let shuffle = shuffle.unwrap_or(true);
    let drop_last = drop_last.unwrap_or(true);
    let tokenizer = r50k_base().unwrap();
    let dataset = GPTDatasetV1::new(txt, tokenizer, max_length, stride);
    // let mut dataloader = DataLoader::builder(dataset).batch_size(batch_size);
    let dataloader = Batcher::new_r2(dataset.into_iter()).batch_size(batch_size);
    // for res in dataloader {
    //     match res {
    //         Ok((inputs, targets)) => {
    //             println!("{inputs:?}, {targets:?}");
    //         }
    //         _ => {
    //             println!("Something went wrong")
    //         }
    //     }
    //     println!("\n")
    // } // not shuffled and last iter is not dropped but it works
    dataloader
}
