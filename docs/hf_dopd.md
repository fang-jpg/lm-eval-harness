# 独立的 d-OPD 评测后端

`hf_dopd_xyf2.sh` 只调用 `hf-dopd`，不会调用或继承 `hf-diffusion` 的生成逻辑。
两条路径分别为：

- `hf-diffusion`：2026-10-12 的 hybrid block-diffusion 逻辑。Prompt prefill
  和完成 block 的 KV 写入使用 causal attention；当前待去噪 block 使用双向
  attention；每个 block 的第一个 token 由 AR 预测提供。
- `hf-dopd`：response-aligned d-OPD 逻辑。Prompt 使用 start-aligned
  staircase block mask；每个 response block 从全 MASK 开始；每个 block
  使用固定去噪步数；每一个完成的 block（包括最后一个或包含 EOS 的 block）
  都通过一次 full-visible SAVING forward 写入 KV cache。

当前论文对齐脚本固定使用 block length 4、每 block 4 个去噪步骤、
temperature 1、top-k 1、top-p 1，并使用 Qwen3 chat template。模型转换、
CANN 路径和 checkpoint 路径仍由 `hf_dopd_xyf2.sh` 独立管理。

d-OPD 使用自己的：

- 模型注册名：`hf-dopd`
- 解码实现：`lm_eval/models/hf_dopd.py`
- 性能指标变量：`HF_DOPD_METRICS_DIR`
- 缓存版本：`standalone-hf-dopd-paper-c4-v1`
- 汇总程序：`scripts/summarize_dopd_metrics.py`

因此，修改普通 diffusion 的 block size、总扩散步数或阈值，不会改变
d-OPD 解码器；修改 d-OPD 的固定 block schedule，也不会进入普通
diffusion 的评测路径。
