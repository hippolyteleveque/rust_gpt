use candle_core::{Device, Module, Tensor};
use candle_nn::ops;
use rust_gpt::attention::{self, CausalSelfAttention, MultiHeadAttentionWrapper, SelfAttentionV1, SelfAttentionV2};

fn main() {
    let device = Device::Cpu;
    let inputs = vec![
        0.43, 0.15, 0.89, 0.55, 0.87, 0.66, 0.57, 0.85, 0.64, 0.22, 0.58, 0.33, 0.77, 0.25, 0.10,
        0.05, 0.80, 0.55,
    ];
    let inputs = Tensor::from_vec(inputs, (6, 3), &device).unwrap();
    let attn = inputs.matmul(&inputs.transpose(0, 1).unwrap()).unwrap();
    // println!("{attn}");

    let sum_lines = attn.sum_keepdim(0).unwrap();
    // println!("{sum_lines}");
    let attn_weights = attn
        .broadcast_div(&sum_lines.transpose(0, 1).unwrap())
        .unwrap();
    // println!("{attn_weights}");

    let attn_weights_softmax = attention::softmax_naive(&attn).unwrap();
    // println!("{attn_weights_softmax}");

    let attn_weights = ops::softmax(&attn, 1).unwrap();
    // println!("{attn_weights}");

    let ctx_vec = attn_weights.matmul(&inputs).unwrap();
    // println!("{ctx_vec}");

    let d_in = inputs.shape().dims()[1];
    let d_out: usize = 2;
    let mut Wq = Tensor::randn(0f64, 1f64, (d_in, d_out), &device).unwrap();
    let mut Wk = Tensor::randn(0f64, 1f64, (d_in, d_out), &device).unwrap();
    let mut Wv = Tensor::randn(0f64, 1f64, (d_in, d_out), &device).unwrap();

    let Q = inputs.matmul(&Wq).unwrap();
    let K = inputs.matmul(&Wk).unwrap();
    let V = inputs.matmul(&Wv).unwrap();

    let tmp = Q.matmul(&K.transpose(0, 1).unwrap()).unwrap();
    let tmp = ops::softmax(&tmp, 1).unwrap();
    let tmp = (tmp / (d_out as f64).powf(0.5)).unwrap();
    let out = tmp.matmul(&V).unwrap();

    // println!("{out}");

    let attn_layer = SelfAttentionV1::new(d_in, d_out);
    let out = attn_layer.forward(&inputs).unwrap();
// 
    // println!("{out}");
    let attn_layer = SelfAttentionV2::new(d_in, d_out);
    let out = attn_layer.forward(&inputs).unwrap();
    // println!("{out}");

    let tril = Tensor::tril2(6, candle_core::DType::F64, &device).unwrap();

    // Causal attention
    let xs = inputs;
    let q = xs.matmul(&Wq).unwrap();
    let k = xs.matmul(&Wk).unwrap();
    let v = xs.matmul(&Wv).unwrap();

    let tmp = q.matmul(&k.transpose(0, 1).unwrap()).unwrap();
    let tmp = (tmp / (d_out as f64).powf(0.5)).unwrap();
    let tmp = ops::softmax(&tmp, 1).unwrap();

    // println!("{tmp}");
    // let sum_rows = tmp.sum(1).unwrap();
    // println!("{sum_rows}");

    let tmp = (tmp * tril).unwrap();
    let sum_rows = tmp.sum_keepdim(1).unwrap();
    let tmp = tmp.broadcast_div(&sum_rows).unwrap();
    // println!("{tmp}");
    let sum_rows = tmp.sum(1).unwrap();
    // println!("{sum_rows}");

    let tmp = ops::dropout(&tmp, 0.5f32).unwrap();

    // println!("{tmp}");

    // let causal_attention = CausalSelfAttention::new(d_in, d_out, 6, 0.5, false);
    // let out = causal_attention.forward(&xs).unwrap();
    // println!("{out}");

    // let multi_head = MultiHeadAttentionWrapper::new(d_in, d_out, 6, 0.5f32, 2, true);
    // let out = multi_head.forward(&xs).unwrap();
    // println!("{out}");

    let inputs = vec![
        0.43, 0.15, 0.89, 0.55, 0.87, 0.66, 0.57, 0.85, 0.64, 0.22, 0.58, 0.33, 0.77, 0.25, 0.10,
        0.05, 0.80, 0.55,
    ];
    let inputs = Tensor::from_vec(inputs, (1, 6, 3), &device).unwrap();
    let causal_attention = CausalSelfAttention::new(d_in, d_out, 6, 0.5, false);
    println!("{inputs}");
    let out = causal_attention.forward(&inputs).unwrap();
    println!("{out}");


}
