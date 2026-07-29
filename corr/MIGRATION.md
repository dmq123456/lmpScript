# 从 `--projection` + `--components` 迁移到 `--component`

旧代码有两套互不相通的通道接口，这里合并成一套。

## 命令行对照

| 旧写法 | 新写法 |
|---|---|
| `--projection cartesian --components xyz`（旧默认） | `--component 1+5+9`（新默认，可省略） |
| `--projection cartesian --components x` | `--component 1`（或 `xx`） |
| `--projection cartesian --components xy` | `--component 1+5` |
| `--projection longitudinal` | `--component L` |
| `--projection transverse` | `--component T` |
| `--component 1 2 3`（旧张量路线） | 不变 |

## 语法

```
token := term ('+' term)*
```

**token 之间用空格分隔 = 分开输出；token 内部用 `+` 连接 = 求和成一个输出。**

```
--component 1 2 3      →  3 个输出：S^xx, S^xy, S^xz
--component 1+2 3      →  2 个输出：S^xx+S^xy, S^xz
--component 1+5+9      →  1 个输出：迹 S^xx+S^yy+S^zz
--component T L 1+5+9  →  3 个输出
```

可用的 term：`1`..`9`（行优先：1=xx, 2=xy, 3=xz, 4=yx, …）、`xx`..`zz`、`x`/`y`/`z`（即 `xx`/`yy`/`zz`）、`L`、`T`。

注意：**`xy` 是张量元素 $S^{xy}$，不是 $S^{xx}+S^{yy}$。** 旧的 `--components xy` 表示后者，现在必须写成 `1+5` 或 `xx+yy`。这个歧义是旧接口里同一个字符串在两个参数下含义相反造成的，新接口不再有。

## 为什么可以这样合并

所有通道都是同一个东西 —— 对结构因子张量的一个实线性泛函：

$$\text{channel}(q,\omega)=\sum_{ab}W_{ab}\,\operatorname{Re}S^{ab}(q,\omega)$$

| token | $W$ |
|---|---|
| `1`..`9` / `xx`..`zz` | $e_ae_b^{\mathsf T}$ |
| `L` | $\hat q\hat q^{\mathsf T}$ |
| `T` | $I-\hat q\hat q^{\mathsf T}$ |
| `a+b` | $W_a+W_b$ |

旧的投影路线只是 $W$ 取特定形式的特例：

- 纵向：$s^L=\sum_a\hat q^as^a$ ⟹ $C^L=\sum_{ab}\hat q^a\hat q^bC^{ab}$，与 `L` 通道**逐式相同**
- 横向：$P=I-\hat q\hat q^{\mathsf T}$ 幂等对称 ⟹ $\sum_aC^{T,aa}=\sum_{cd}(P^{\mathsf T}P)_{cd}C^{cd}=\sum_{cd}P_{cd}C^{cd}$，与 `T` 通道**逐式相同**

这一等价性在旧代码上实测确认过（相对差 ~1e-8，即 complex64 精度）。

## clip 规则的变化

旧规则是「对角分量 clip 到 $\ge0$，非对角保号」。分组之后这条判不了了（`1+2` 两边都不是）。

新规则：**当 $W$ 对称半正定时 clip，否则保号。**

依据是 $S^{ab}(q,\omega)$ 作为谱密度矩阵在每个频率处 Hermitian 半正定，所以 $\sum_{ab}W_{ab}S^{ab}\ge0$ 当且仅当 $W$ 对称半正定 —— 此时该通道才是物理强度，负值必属数值噪声。这是旧规则的严格推广：单个对角元、迹、`L`、`T` 都落在 clip 一侧。

## 新增能力

旧代码里这三个组合会直接报错，现在都可用：

- `--component` + `--bz-folded`
- `--component` + `--save-corr-plus`（现在导出完整的 $C^{ab}$，形状 `(nq, nt, 3, 3)`，是旧版 `(nq, nt, 3)` 的超集）
- 任意分组求和

## 输出文件命名

- 单通道：不加后缀，`--output foo.npz` 就写 `foo.npz`（与旧默认行为一致）
- 多通道：每通道一个文件，`foo_T.npz`、`foo_1p5p9.npz`（`+` 在文件名里写作 `p`）

---

# `--window` 的语义变更（破坏性）

## 变了什么

| | 旧 | 新 |
|---|---|---|
| 作用对象 | $C(\tau)$（滞后窗，长 $2N_t-1$） | $s(\qvec,t)$（数据窗，长 $N_t$） |
| 作用位置 | 相关**之后** | 相关**之前** |
| 治的问题 | 方差（统计涨落） | 泄漏（有限记录造成的假强度） |

`--window none` 的行为不变（两种情况下都不加窗）。**`--window hann` 的结果会变**，这是有意的。

## 为什么

滞后窗只在**物理线宽 ≫ 频率分辨率 $1/(N_t\Delta t)$** 时才可用。实测：当真实 FWHM 小于分辨率时，全长/截断滞后窗把线宽量成真值的 **11 倍** —— 测到的是窗宽，不是磁振子寿命。而旧默认恰好是打开滞后窗，等于默认破坏线宽测量。

数据窗则是无条件划算的：它消除泄漏（实测可把强峰旁弱支的高估从 821 倍降到 1.1 倍），代价只是主瓣略微变宽。

若确实需要降方差，正确做法是**平均多条独立轨迹**或对称等价 q 点，而不是滞后窗 —— 加长单条轨迹无效（帧数增加 8 倍，每点相对涨落纹丝不动）。

## 一个重要后果：corr 与 periodogram 现在等价

加数据窗之后，$C(\tau)$ 成了加窗数据的自相关，其傅里叶变换恰是 $|\mathcal F[s\cdot w]|^2$ —— 也就是加窗周期图。实测两者差 **6e-16**（`none` 和 `hann` 均如此）。

唯一的差别是频率网格：corr 走 $2N_t-1$ 点，periodogram 走 $N_t$ 点。多出来的点是**补零插值，不是分辨率提升** —— 真实分辨率由 $N_t\Delta t$ 决定。

因此 corr 路线现在仅剩两项独有能力：

1. `--save-corr-plus` —— 直接拿到 $C^{ab}(\qvec,\tau)$
2. `--corr-norm unbiased` —— 但它破坏半正定性，谱会出负值，实用价值存疑

如果这两项都不需要，corr 路线可以整体删掉，改用 periodogram（后者还省一次逆变换）。
