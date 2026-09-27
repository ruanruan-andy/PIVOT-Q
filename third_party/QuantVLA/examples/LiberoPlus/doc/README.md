# LIBERO-Plus 实验入口

本目录只保留入口；完整且唯一维护的教程位于仓库根目录：

- `docs/FP16_QUANTVLA_EVAL_CN.md`：FP16 与 QuantVLA eval；
- `docs/QUANTVLA_PIVOT_Q_TRAIN_EVAL_CN.md`：PIVOT-Q train、resume 与 eval；
- `docs/METHOD_CN.md`：当前方法定义；
- `docs/EXPERIMENTS_CN.md`：正式实验矩阵与记录表。

正式协议：

```text
Shared-560: configs/libero_plus/shared560-first20.json
每个 suite/category 取 task index 最小的 20 条；train 与 eval 共用 task IDs。
```

这是 same-task transductive adaptation，不是 held-out-task generalization。

最短入口：

```bash
./eval_fp16.sh --help
./eval_quantvla.sh --help
./train_quantvla_pivot_q.sh --help
./eval_quantvla_pivot_q.sh --help
./monitor_eval.sh --once
./collect_eval.sh --require-complete
```

所有 launcher 都支持 card、suite、port、output、checkpoint、Standard LIBERO / LIBERO-Plus、seed 与 resume；不要再手工组合 inference server 和 evaluator。
