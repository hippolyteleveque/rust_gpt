use candle_core::{Device, Tensor};
use rust_gpt::attention;

fn main() {
    let device = Device::Cpu;
    let inputs = vec![
        0.43, 0.15, 0.89, 0.55, 0.87, 0.66, 0.57, 0.85, 0.64, 0.22, 0.58, 0.33, 0.77, 0.25, 0.10,
        0.05, 0.80, 0.55,
    ];
    let inputs = Tensor::from_vec(inputs, (6, 3), &device).unwrap();
    let attn = inputs.matmul(&inputs.transpose(0,1).unwrap()).unwrap();
    // println!("{attn}");

    let sum_lines = attn.sum_keepdim(0).unwrap();
    println!("{sum_lines}");
    let attn_weights = attn.broadcast_div(&sum_lines.transpose(0, 1).unwrap()).unwrap();
    println!("{attn_weights}");

    let attn_weights_softmax = attention::softmax_naive(&attn).unwrap();
    println!("{attn_weights_softmax}");
}
