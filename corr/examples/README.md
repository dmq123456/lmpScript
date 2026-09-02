# 用法示例

每个脚本都能直接跑，输出落在 `out/`（已被 `.gitignore` 忽略）。帧数和 q 点数都调小了，几秒钟跑完 —— 目的是展示命令怎么写，不是产出可用的结果。

```bash
cd corr/examples
./run_sqw.sh
```

换成自己的数据：

```bash
DATA_DIR=/path/to/your/data ./run_sqw.sh
# 或者单独指定
SPIN_DUMP=/path/to/dump.lammpstrj QPATH=/path/to/qpath.txt ./run_sqw.sh
```

共享设置都在 `common.sh` 里。

## 脚本

| 脚本 | 演示 |
|---|---|
| `run_sqw.sh` | `sqw_spin_corr.py` —— 默认迹、多通道一次算、带图的完整运行 |
| `run_dos.sh` | `sqw_dos.py` —— 无需 q 路径与超胞；`L`/`T` 为何被拒绝 |
| `run_average.sh` | 分时间窗多次运行 + `average_sqw.py` 求平均降方差 |
| `run_corrplus.sh` | `--save-corr-plus` 读寿命，以及会污染包络的两个默认设置 |
| `run_animate.sh` | `tools/frame.py` 单帧调参 → `tools/animate.py` 出动画 |

## 三个容易踩的地方

**① 方括号必须加引号。** `c_outsp[1]` 在 bash 里是通配符。当前目录恰好没有同名文件时它会原样传过去，所以很容易一直没出问题 —— 直到某天出问题。

```bash
--field-columns 'c_outsp[1]' 'c_outsp[2]' 'c_outsp[3]'
```

**② 频域和时域要的默认值相反。**

| 用途 | `--window` | `--corr-norm` |
|---|---|---|
| $S(\qvec,\omega)$ | `hann` | `biased` |
| `--save-corr-plus` 读 $C(\qvec,\tau)$ | `none` | `unbiased` |

用错了不会报错，只会让你把窗函数的形状当成磁振子寿命。`run_corrplus.sh` 里有量化演示。

**③ 平均降方差要求各次运行独立。** 同一条轨迹切成连续窗口只是近似独立，`average_sqw.py` 会把实测散布和理论 $1/\sqrt M$ 一起打出来 —— 实测远小于 1.0 就说明窗口之间相关，收益达不到预期。`run_average.sh` 演示的正是这种情况（s4 是确定性信号，散布为 0）。

## 数据

- `examples/s4.lammpstrj` —— 20×20 三角格子上 4.0 THz 的相干自旋波，10000 帧。峰位、通道选择这些都可以拿它对照
- `test/test.lammpstrj` —— 只有 2 帧，但带 `c_outbec[*]` 列，用于动画示例
