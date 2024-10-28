use candle_core::{Device, Result, Tensor};
use candle_nn::ops;
use candle_nn::{Linear, Module};

pub fn softmax_naive(t: &Tensor) -> Result<Tensor> {
    // naive implementation of softmax
    let xp = t.exp()?;
    let sxp = xp.sum_keepdim(0)?;
    xp.broadcast_div(&sxp.transpose(0, 1)?)
}

pub struct SelfAttentionV1 {
    d_out: usize,
    wq: Tensor,
    wv: Tensor,
    wk: Tensor,
}

impl SelfAttentionV1 {
    pub fn new(d_in: usize, d_out: usize) -> SelfAttentionV1 {
        let wq = Tensor::randn(0f64, 1f64, (d_in, d_out), &Device::Cpu).unwrap();
        let wv = Tensor::randn(0f64, 1f64, (d_in, d_out), &Device::Cpu).unwrap();
        let wk = Tensor::randn(0f64, 1f64, (d_in, d_out), &Device::Cpu).unwrap();
        SelfAttentionV1 { d_out, wq, wv, wk }
    }
}

impl Module for SelfAttentionV1 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let q = xs.matmul(&self.wq)?;
        let k = xs.matmul(&self.wk)?;
        let v = xs.matmul(&self.wv)?;

        let tmp = q.matmul(&k.transpose(0, 1).unwrap())?;
        let tmp = (tmp / (self.d_out as f64).powf(0.5))?;
        let tmp = ops::softmax(&tmp, 1)?;
        let out = tmp.matmul(&v).unwrap();
        Ok(out)
    }
}

pub struct SelfAttentionV2 {
    d_out: usize,
    q: Linear,
    v: Linear,
    k: Linear,
}

impl SelfAttentionV2 {
    pub fn new(d_in: usize, d_out: usize) -> SelfAttentionV2 {
        // Be careful to transpose size of the the shape when applying forward
        let wq = Tensor::randn(0f64, 1f64, (d_out, d_in), &Device::Cpu).unwrap();
        let wv = Tensor::randn(0f64, 1f64, (d_out, d_in), &Device::Cpu).unwrap();
        let wk = Tensor::randn(0f64, 1f64, (d_out, d_in), &Device::Cpu).unwrap();
        let q = Linear::new(wq, None);
        let v = Linear::new(wv, None);
        let k = Linear::new(wk, None);
        SelfAttentionV2 { d_out, q, v, k }
    }
}

impl Module for SelfAttentionV2 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let q = self.q.forward(xs)?;
        let v = self.v.forward(xs)?;
        let k = self.k.forward(xs)?;
        let tmp = q.matmul(&k.transpose(0, 1)?)?;
        let tmp = (tmp / (self.d_out as f64).powf(0.5))?;
        let tmp = ops::softmax(&tmp, 1)?;
        let out = tmp.matmul(&v).unwrap();
        Ok(out)
    }
}

pub struct CausalSelfAttention {
    d_out: usize,
    q: Linear,
    v: Linear,
    k: Linear,
    dropout_rate: f32,
    context_length: usize,
}

impl CausalSelfAttention {
    pub fn new(
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout: f32,
        qkv_bias: bool,
    ) -> CausalSelfAttention {
        let wq = Tensor::randn(0f64, 1f64, (d_out, d_in), &Device::Cpu).unwrap();
        let wv = Tensor::randn(0f64, 1f64, (d_out, d_in), &Device::Cpu).unwrap();
        let wk = Tensor::randn(0f64, 1f64, (d_out, d_in), &Device::Cpu).unwrap();
        let mut bq = None;
        let mut bv = None;
        let mut bk = None;
        if qkv_bias {
            bq = Some(Tensor::randn(0f64, 1.0, (d_out,), &Device::Cpu).unwrap());
            bv = Some(Tensor::randn(0f64, 1.0, (d_out,), &Device::Cpu).unwrap());
            bk = Some(Tensor::randn(0f64, 1.0, (d_out,), &Device::Cpu).unwrap());
        }
        let q = Linear::new(wq, bq);
        let v = Linear::new(wv, bv);
        let k = Linear::new(wk, bk);
        CausalSelfAttention {
            d_out,
            dropout_rate: dropout,
            q,
            k,
            v,
            context_length,
        }
    }
}

impl Module for CausalSelfAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let q = self.q.forward(xs)?;
        let v = self.v.forward(xs)?;
        let k = self.k.forward(xs)?;
        let tmp = q.matmul(&k.transpose(0, 1)?)?;
        let tmp = (tmp / (self.d_out as f64).powf(0.5))?;
        let tmp = ops::softmax(&tmp, 1)?;
        let tril = Tensor::tril2(self.context_length, candle_core::DType::F64, &Device::Cpu)?;
        let tmp = (tmp * tril)?;
        let sum_rows = tmp.sum(1)?;
        let tmp = tmp.broadcast_div(&sum_rows)?;
        let tmp = ops::dropout(&tmp, self.dropout_rate)?;
        let out = tmp.matmul(&v).unwrap();
        Ok(out)
    }
}

pub struct MultiHeadAttentionWrapper {
    heads: Vec<CausalSelfAttention>,
}

impl MultiHeadAttentionWrapper {
    pub fn new(
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout: f32,
        num_heads: usize,
        qkv_bias: bool,
    ) -> MultiHeadAttentionWrapper {
        let mut heads = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            heads.push(CausalSelfAttention::new(d_in, d_out, context_length, dropout, qkv_bias));
        }
        MultiHeadAttentionWrapper { heads }
    }
}

impl Module for MultiHeadAttentionWrapper {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let size = self.heads.len();
        let mut res = Vec::with_capacity(size);
        for i in 0..size {
            let z = self.heads[i].forward(xs)?;
            res.push(z);
        }
        let last_dim = res[0].shape().dims().len();
        let out = Tensor::cat(&res[..], last_dim - 1)?;
        Ok(out)
    }
}
