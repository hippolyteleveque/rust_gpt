use candle_core::{Tensor, Result};

pub fn softmax_naive(t: &Tensor) -> Result<Tensor> {
    // naive implementation of softmax
    let xp = t.exp()?;
    let sxp = xp.sum_keepdim(0)?;
    xp.broadcast_div(&sxp.transpose(0, 1)?)
}