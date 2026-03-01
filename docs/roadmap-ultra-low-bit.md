# llm-quantize 開発ロードマップ: Ultra-Low-Bit 量子化

**日付**: 2026-03-01 (更新)
**ステータス**: Phase 5 完了 — 全能動的手段を試行済み、受動的待機に移行
**対象**: `_release/llm-quantize/`, `_release/llm-pruning/`

## 背景

Qwen3.5-27B 評価（[qwen35-27b-evaluation.md](./qwen35-27b-evaluation.md)）により、以下の課題が明らかになった:

1. **TERNARY モード**: FP32 全量ロード（108GB）が必要で大規模モデルに使用不可。出力が numpy 形式で推論エンジン非互換
2. **IQ 量子化**: llama-quantize への単純委譲のみ。imatrix 生成が未実装
3. **入力制限**: HuggingFace safetensors のみ対応。GGUF 入力不可
4. **SSM 非対応**: Mamba/ハイブリッドアーキテクチャのテンソル構造を考慮していない
5. **スケーリング**: per-tensor スケーリングのみ。BitNet b1.58 は per-channel を要求

---

## 重要な前提: 事後量子化 vs 三値訓練

Qwen3.5-27B 評価により判明した最重要知見:

| 手法 | 品質 | 速度 | 状況 |
|------|------|------|------|
| **事後量子化** (PTQ → TQ1_0/IQ1_S) | 壊滅 | 高速 | ツールあり、品質不足 |
| **三値訓練** (QAT → BitNet b1.58) | 維持 | 最高速 | モデル最大 3B |

1.58-bit で品質を維持するには**モデル自体が三値重みで訓練**される必要がある。事後量子化では 3 bpw 以下で品質が崩壊する。この制約を踏まえ、llm-quantize は**両方のアプローチ**をサポートする方針とする。

---

## アーキテクチャ方針

### 原則

1. **デュアルバックエンド**: llama.cpp (GGUF/TQ1_0) と bitnet.cpp の両方を推論ターゲットとする
2. **ストリーミング処理**: メモリ制約環境（24GB）でも 27B+ モデルを処理可能にする
3. **imatrix 統合**: 超低ビット量子化の品質を imatrix で担保する
4. **アーキテクチャ認識**: SSM/MTP テンソルを正しくハンドリングする
5. **三値訓練モデル対応**: BitNet b1.58 形式のモデルを bitnet.cpp 用 GGUF に変換可能にする

### 非目標

- 独自推論カーネルの開発（llama.cpp / bitnet.cpp に委ねる）
- 三値訓練（QAT）自体の実装（訓練は Microsoft BitNet / HuggingFace 側）
- GPU 量子化カーネルの実装（CUDA/Metal）
- GPTQ/AWQ 形式の超低ビット対応

---

## Phase 1: TERNARY → GGUF/TQ1_0 出力 (P0)

### 目標

`_quantize_ternary()` の出力を numpy `.npz` から GGUF TQ1_0 形式に変更する。

### 現状の問題

```python
# 現在: FP32 全量ロード → numpy 出力
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
# → 27B モデルで 108GB 必要、出力が推論エンジンに非互換
```

### 実装方針

#### 1a. Per-channel スケーリング（BitNet b1.58 準拠）

```python
# 現在: per-tensor
scale = np.mean(np.abs(weights))
ternary = np.clip(np.round(weights / scale), -1, 1)

# 変更後: per-channel (output channel axis)
scale = np.mean(np.abs(weights), axis=-1, keepdims=True)
ternary = np.clip(np.round(weights / scale), -1, 1)
```

**根拠**: per-tensor では大きなチャネルのスケールが小さなチャネルを支配し、情報損失が増大する。BitNet b1.58 論文で per-channel が標準。

#### 1b. ストリーミング GGUF 書込み

```python
from gguf import GGUFWriter

def quantize_ternary_to_gguf(input_path, output_path):
    writer = GGUFWriter(output_path, arch="llama")  # or "qwen35"

    # メタデータ書込み（config.json から取得）
    writer.add_name(model_name)
    writer.add_context_length(ctx_len)
    # ...

    # シャードごとにストリーミング処理
    for shard in sorted(glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if should_quantize(name):
                    scale = tensor.abs().mean(dim=-1, keepdim=True)
                    ternary = torch.clamp(torch.round(tensor / scale), -1, 1)
                    writer.add_tensor(name, ternary, raw_dtype=GGMLQuantizationType.TQ1_0)
                else:
                    writer.add_tensor(name, tensor)
        # オプション: シャード削除でディスク回収

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
```

**ディスクバジェット**: シャードごとにピーク 5.3GB（1シャード分のメモリ）。

#### 1c. TQ1_0 テンソルエンコーディング

TQ1_0 は llama.cpp 独自のブロックフォーマット:

```
ブロックサイズ: 256 要素
格納: 5 要素を 1 バイトにパック（3^5 = 243 < 256）
ブロック構造: [packed_quants: 51B][scales: 2B] = 53B/256要素
BPW: 53*8/256 = 1.656 → 公称 1.69 BPW
```

`gguf-py` の `GGUFWriter.add_tensor()` に `raw_dtype=GGMLQuantizationType.TQ1_0` を渡す。gguf-py が対応していない場合は、ブロックエンコーディングを自前実装する。

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `lib/quantizers/advanced/ultra_low_bit.py` | `_quantize_ternary()` をストリーミング GGUF 出力に書換え |
| `lib/quantizers/advanced/ultra_low_bit.py` | per-channel scaling 実装 |
| `tests/test_advanced/test_ultra_low_bit.py` | per-channel テスト、GGUF 出力テスト追加 |

### 検証基準

- [ ] 7B モデル（16GB 環境）で GGUF TQ1_0 生成が成功すること
- [ ] per-channel と per-tensor のスケーリング差分が検証可能なこと
- [ ] 出力 GGUF が `llama-quantize --dry-run` でメタデータ読取り可能なこと

---

## Phase 2: GGUF 入力サポート (P1)

### 目標

GGUF ファイルを直接入力として受け付け、再量子化（例: Q4_K_M → TQ1_0）を可能にする。

### 現状の問題

```python
# 現在: HuggingFace モデルのみ
model = AutoModelForCausalLM.from_pretrained(path)
```

Ollama で利用可能な GGUF ファイルから直接超低ビット量子化したいケースに対応できない。

### 実装方針

```python
# model_loader.py に GGUF 検出追加
def create_source_model(model_path: str) -> SourceModel:
    if model_path.endswith('.gguf'):
        return _load_from_gguf(model_path)
    else:
        return _load_from_hf(model_path)

def _load_from_gguf(path: str) -> SourceModel:
    from gguf import GGUFReader
    reader = GGUFReader(path)
    arch = reader.fields['general.architecture'].parts[-1]
    param_count = _count_parameters(reader)
    return SourceModel(
        model_path=path,
        architecture=arch,
        parameter_count=param_count,
        num_layers=_get_num_layers(reader),
    )
```

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `lib/model_loader.py` | `.gguf` 拡張子検出、GGUF メタデータ読取り |
| `lib/quantizers/advanced/ultra_low_bit.py` | GGUF テンソル読出しパス追加 |
| `tests/test_model_loader.py` | GGUF 入力テスト追加 |

### 検証基準

- [ ] Ollama GGUF blob からアーキテクチャ・テンソル情報を読取れること
- [ ] GGUF → GGUF 再量子化パイプラインが動作すること

---

## Phase 3: imatrix 統合 (P1)

### 目標

`_get_or_compute_imatrix()` stub を実装し、IQ1_S/IQ2_XXS の品質を向上させる。

### 現状の問題

```python
def _get_or_compute_imatrix(self) -> Optional[Path]:
    # For now, return None - full implementation would compute imatrix
    return None
```

IQ1_S/IQ2_XXS は imatrix なしでは品質が大幅に劣化する。

### 実装方針

```python
def _get_or_compute_imatrix(self) -> Optional[Path]:
    # 1. キャッシュ済み imatrix を検索
    cache_path = self._imatrix_cache_path()
    if cache_path.exists():
        return cache_path

    # 2. キャリブレーションデータの取得
    cal_data = self._get_calibration_data()
    if cal_data is None:
        logger.warning("No calibration data; skipping imatrix")
        return None

    # 3. llama-imatrix 呼出し
    cmd = [
        "llama-imatrix",
        "-m", str(self.source_model.model_path),
        "-f", str(cal_data),
        "-o", str(cache_path),
        "--chunks", "32",
    ]
    subprocess.run(cmd, check=True)
    return cache_path
```

### キャリブレーションデータ戦略

| ソース | 用途 |
|--------|------|
| `--calibration-data` CLI 引数 | ユーザー指定（最優先）|
| `calibration.txt` (モデルディレクトリ) | モデル同梱 |
| WikiText-2 サブセット | デフォルトフォールバック |
| umigame プロンプト集 (EN/JA) | ドメイン特化（サービス固有） |

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `lib/quantizers/advanced/ultra_low_bit.py` | `_get_or_compute_imatrix()` 実装 |
| `lib/calibration.py` | 新規: キャリブレーションデータ管理 |
| `tests/test_advanced/test_ultra_low_bit.py` | imatrix 統合テスト追加 |

### 検証基準

- [ ] `llama-imatrix` が利用可能な場合、imatrix.dat が生成されること
- [ ] 生成された imatrix が `llama-quantize --imatrix` で使用可能なこと
- [ ] キャリブレーションデータのフォールバックチェーンが正しく動作すること

---

## Phase 4: SSM/ハイブリッドアーキテクチャ対応 (P2)

### 目標

Mamba (SSM) + Transformer ハイブリッドモデル（Qwen3.5, Jamba 等）のテンソルを正しくハンドリングする。

### 現状の問題

Qwen3.5-27B のテンソル構成:

| テンソル種別 | 例 | 数 | 量子化可否 |
|-------------|----|----|-----------|
| Transformer attention | `blk.*.attn_q/k/v/o.weight` | ~400 | 可 |
| MLP | `blk.*.ffn_gate/up/down.weight` | ~300 | 可 |
| **Mamba SSM** | `blk.*.ssm_a`, `blk.*.ssm_dt`, `blk.*.ssm_conv1d` | ~300 | **要選択的処理** |
| **MTP** | `mtp.fc.weight`, `mtp.norm.weight` | ~10 | 可（低スパース率推奨） |
| Embedding / Norm | `token_embd`, `output_norm` | ~10 | 除外 |

### 実装方針

```python
# テンソル分類器
def classify_tensor(name: str, arch: str) -> TensorRole:
    if 'ssm_a' in name or 'ssm_dt' in name:
        return TensorRole.SSM_STATE  # 量子化除外 or 高精度
    if 'ssm_conv1d' in name:
        return TensorRole.SSM_CONV   # 高精度推奨
    if 'ssm_alpha' in name or 'ssm_in_proj' in name:
        return TensorRole.SSM_PROJ   # 量子化可
    if 'mtp' in name:
        return TensorRole.MTP        # 低スパース率
    if 'embed' in name or 'norm' in name:
        return TensorRole.PRESERVE   # 量子化除外
    return TensorRole.STANDARD       # 通常量子化

# レイヤー別量子化設定
SSM_QUANT_POLICY = {
    TensorRole.SSM_STATE:  QuantPolicy(skip=True),          # 状態遷移行列は除外
    TensorRole.SSM_CONV:   QuantPolicy(min_bits=8),         # 畳み込みは高精度
    TensorRole.SSM_PROJ:   QuantPolicy(sparsity_cap=0.15),  # 投影は控えめに
    TensorRole.MTP:        QuantPolicy(sparsity_cap=0.10),  # MTP は保守的
    TensorRole.PRESERVE:   QuantPolicy(skip=True),
    TensorRole.STANDARD:   QuantPolicy(),                    # デフォルト
}
```

### Ollama / llama.cpp テンソル名マッピング

```python
# SSM テンソル名の正規化（Ollama ↔ Homebrew 差分を吸収）
TENSOR_NAME_MAP = {
    "ssm_dt.bias": "ssm_dt",      # Homebrew → Ollama
    "ssm_dt":      "ssm_dt.bias", # Ollama → Homebrew
}
```

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `lib/quantizers/advanced/ultra_low_bit.py` | テンソル分類・SSM ポリシー適用 |
| `lib/tensor_classifier.py` | 新規: アーキテクチャ別テンソル分類 |
| `tests/test_advanced/test_tensor_classifier.py` | 新規: 分類テスト |

### 検証基準

- [ ] Qwen3.5 テンソル名リストから全テンソルが正しく分類されること
- [ ] SSM 状態テンソルが量子化されないこと
- [ ] Ollama / Homebrew 両方のテンソル名を受け付けること

---

## Phase 5: 構造的プルーニング統合 (実装完了 — 品質壊滅)

> **ステータス**: 実装完了・評価完了 (2026-03-01)
> **判定**: パイプラインは技術的に成功。品質は壊滅。**fine-tuning なしの構造的プルーニングは実用不可。**

### 目標 (当初)

llm-pruning の均衡プルーニングを**構造的**（ヘッド/チャネル/レイヤー単位）に拡張し、GGUF サイズ削減に直結させる。Qwen3.5-27B を **40% 構造削減 → Q4_K_M ≈ 10GB** で M2 24GB に収めることが最終目標。

### 実装成果

**実装ファイル**:

| ファイル | 内容 |
|---------|------|
| `llm-pruning/scripts/structural_prune.py` | 構造的プルーニングメインスクリプト |
| `llm-pruning/tests/test_structural_prune.py` | 57 テスト全通過 |

**対応アーキテクチャ**: Transformer (標準) + Gated DeltaNet (Qwen3.5 ハイブリッド)

**機能**:
- 均衡ゲーム理論によるグループレベル重要度スコアリング
- MLP チャネル削減、Full Attention ヘッド削減、Linear V-head 削減
- レイヤー除去 (`full_attention_interval` 検出時に自動無効化)
- DeltaNet 専用テンソル (in_proj_z, in_proj_a/b, A_log, dt_bias, conv1d) の正しいプルーニング
- V-head 数の K-head 整除性自動保証 (GGUF 互換)
- safetensors.index.json 自動再生成
- in-place モード (ディスク制約環境対応)
- dry-run モード (プラン出力のみ)

### Qwen3.5-27B 実機結果

**プルーニング設定**:

| 対象 | Before | After | 削減率 |
|------|--------|-------|--------|
| MLP channels | 17408 | 12160 | 30% |
| Full Attn Q heads | 24 | 16 | 33% |
| Linear V heads | 48 | 32 | 33% |
| Layers | 64 | 64 | 0% (自動無効化) |

**サイズ・速度**: 目標達成

| 指標 | 元モデル | プルーニング後 | 目標 |
|------|---------|-------------|------|
| GGUF サイズ | 17GB | 12GB | ≤ 10GB |
| tok/s | 2.5-3.0 | 5-10 | 8-12 |
| swap | 3GB | 0 | 0 |

**品質**: 壊滅

| テスト | 結果 |
|--------|------|
| chat completion | `<think>` 2トークンで停止 |
| raw completion | 無意味な英字列 (gibberish) |
| umigame 正答率 | 0/3 (応答生成不可能) |

### 失敗分析

30% 構造的プルーニング (fine-tuning なし) が壊滅した原因:

1. **Fine-tuning 不在**: 残存パラメータ間の整合性が崩壊
2. **DeltaNet 感度**: 線形注意は V-head 削減に特に敏感
3. **削減率過大**: 30% が限界を超えている可能性
4. **均一削減の限界**: レイヤー間の役割差異を無視

### 結論

```
構造的プルーニング (fine-tuning なし) は Q2_K と同じ結論に帰着:

  Q2_K:     27B のビットを削る → 品質壊滅
  構造的:   27B のパラメータを削る → 品質壊滅

  どちらも「重み情報の大幅削減」を fine-tuning なしで行うため、
  一定の閾値を超えると品質が崩壊する。
```

**構造的プルーニングを実用化するには**: プルーニング後に LoRA fine-tuning が必須。これには A100 等の GPU 環境が必要であり、M2 MacBook Air 単体では実現不可能。

### 検証基準 (結果)

- [x] ~~小規模モデル (125M-1B) で構造的プルーニング 40% → GGUF 変換が成功すること~~ (未実施 — 27B で直接テスト)
- [x] GGUF サイズが削減率に比例して減少すること → **17GB → 12GB (30% 削減)**
- [x] プルーニング済み GGUF が llama-server でロード可能なこと → **成功** (Ollama は非対応)
- [ ] ~~perplexity が元モデル Q4_K_M の +30% 以内であること~~ → **壊滅 (計測不能)**
- [ ] ~~Qwen3.5-27B 40% プルーニング → Q4_K_M ≈ 10GB で swap なし動作~~ → **速度達成、品質壊滅**

---

## Phase 6: bitnet.cpp 統合 (P1)

### 目標

Microsoft BitNet (bitnet.cpp) を推論バックエンドとして統合し、三値訓練済みモデルの変換・推論パイプラインを提供する。

### 背景: bitnet.cpp とは

[microsoft/BitNet](https://github.com/microsoft/BitNet) は 1.58-bit LLM 専用推論エンジン。llama.cpp の TQ1_0 と根本的にアプローチが異なる。

**カーネル比較**:

| | llama.cpp TQ1_0 | bitnet.cpp TL2 |
|---|---|---|
| 演算方式 | MAD (乗算-加算) | **LUT (ルックアップテーブル)** |
| 乗算 | あり（スケール乗算） | **なし**（加算/減算のみ） |
| Intel i7 速度比 | 1.0x | **2.32x** |
| M2 Ultra 速度比 | 1.0x | **1.19x** |
| エネルギー効率 | 標準 | **55-82% 削減** |

LUT 方式は三値重み {-1,0,+1} の性質を活用し、入力活性値のテーブルを事前計算することで乗算を完全に排除する。

**利用可能モデル** (2026年3月時点):

| モデル | パラメータ | HuggingFace |
|--------|-----------|-------------|
| bitnet-b1.58-2B-4T | 2B | microsoft/bitnet-b1.58-2B-4T |
| bitnet_b1_58-3B | 3.3B | 1bitLLM/bitnet_b1_58-3B |
| bitnet_b1_58-large | 0.7B | 1bitLLM/bitnet_b1_58-large |

### 実装方針

#### 6a. bitnet.cpp ビルド統合

```python
class BitNetBackend:
    """bitnet.cpp inference backend."""

    def __init__(self, bitnet_path: Optional[Path] = None):
        self.bitnet_path = bitnet_path or self._find_bitnet()

    def _find_bitnet(self) -> Path:
        """Discover bitnet.cpp installation."""
        candidates = [
            Path.home() / "BitNet",
            Path("/usr/local/lib/bitnet"),
            # Homebrew future support
        ]
        for p in candidates:
            if (p / "build" / "bin").exists():
                return p
        raise RuntimeError("bitnet.cpp not found")

    def convert_model(self, hf_model_path: str, output_path: Path,
                      quant_type: str = "i2_s") -> Path:
        """Convert HuggingFace BitNet model to bitnet.cpp GGUF."""
        cmd = [
            "python3", str(self.bitnet_path / "convert_pt_to_gguf.py"),
            "--input", hf_model_path,
            "--output", str(output_path),
            "--quant-type", quant_type,  # i2_s, tl1, tl2
        ]
        subprocess.run(cmd, check=True)
        return output_path

    def benchmark(self, gguf_path: Path, threads: int = 4) -> dict:
        """Run inference benchmark."""
        cmd = [
            str(self.bitnet_path / "build" / "bin" / "llama-cli"),
            "-m", str(gguf_path),
            "-t", str(threads),
            "-p", "Hello",
            "-n", "128",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return self._parse_benchmark(result.stdout)
```

#### 6b. カーネル選択ガイダンス

```python
KERNEL_RECOMMENDATION = {
    "x86_desktop":  "i2_s",   # 高スレッド環境
    "x86_laptop":   "tl2",    # 省電力重視
    "arm_m_series": "tl2",    # Apple Silicon 最適
    "arm_mobile":   "tl1",    # メモリ制約
}

def recommend_kernel(platform: str, model_size_b: float) -> str:
    """Recommend optimal bitnet.cpp kernel for platform."""
    if model_size_b > 10 and platform.startswith("arm"):
        return "tl2"  # 大規模モデル + ARM = 最高圧縮
    return KERNEL_RECOMMENDATION.get(platform, "i2_s")
```

#### 6c. Embedding 量子化オプション

bitnet.cpp は 2026年1月に Embedding の Q6_K 量子化をサポート。llm-quantize でも対応:

```python
def quantize_for_bitnet(model_path: str, output_path: str,
                        quant_type: str = "tl2",
                        quantize_embeddings: bool = True):
    """End-to-end BitNet quantization pipeline."""
    backend = BitNetBackend()

    # 1. HF → bitnet.cpp GGUF
    gguf_path = backend.convert_model(model_path, Path(output_path), quant_type)

    # 2. Embedding 量子化 (オプション)
    if quantize_embeddings:
        _quantize_embeddings_q6k(gguf_path)

    # 3. ベンチマーク
    metrics = backend.benchmark(gguf_path)
    return gguf_path, metrics
```

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `lib/backends/bitnet.py` | 新規: bitnet.cpp バックエンド統合 |
| `lib/quantizers/advanced/ultra_low_bit.py` | TERNARY モード → bitnet.cpp パス分岐 |
| `tests/test_advanced/test_bitnet_backend.py` | 新規: BitNet 統合テスト |

### 検証基準

- [ ] bitnet.cpp がインストール済みの環境で、HF モデル → GGUF 変換が成功すること
- [ ] i2_s/tl1/tl2 全カーネルでベンチマークが実行可能なこと
- [ ] bitnet.cpp 未インストール時に明確なエラーメッセージを返すこと
- [ ] bitnet-b1.58-2B-4T で推論が動作し、tok/s が計測可能なこと

### 制約事項

- bitnet.cpp は **Ollama 非互換** — 独自サーバーとして動作する
- 三値訓練モデルのみ対応 — 通常モデルの事後量子化には使用不可
- 27B クラスの公開三値訓練モデルが存在するまで、実用的な効果は限定的

---

## 実装優先度

| Phase | 優先度 | 状態 | 工数 | 効果 | 根拠 |
|-------|--------|------|------|------|------|
| **Phase 5: 構造的プルーニング** | ~~P0~~ | **完了 — 品質壊滅** | L | サイズ削減は成功、品質は壊滅 | fine-tuning なしでは実用不可と判明 |
| Phase 3: imatrix | P1 | 未着手 | S | IQ2_XXS 品質向上 | 次に試す価値あり |
| Phase 2: GGUF 入力 | P1 | 未着手 | S | 既存 GGUF の再量子化 | |
| Phase 4: SSM 対応 | P2 | 部分実装 (Phase 5 で DeltaNet 対応済み) | M | ハイブリッドモデル対応 | |
| Phase 1: TERNARY → GGUF | P3 | 未着手 | M | 事後三値化 | 品質壊滅のため優先度低下 |
| Phase 6: bitnet.cpp 統合 | P3 | 未着手 | S | 三値訓練モデル対応 | 27B 三値モデル未公開のため待機 |

### 優先度変更の履歴

**v1 (旧方針)**: TERNARY → GGUF 出力 (P0) を最優先

**v2 (2026-03-01 前半)**: 構造的プルーニング (P0) を最優先

**v3 (2026-03-01 後半, 現在)**: Phase 5 完了。全能動的手段を試行済み。

```
確定した事実:
  1. 事後量子化 3 bpw 以下 → 品質壊滅（Q2_K, TQ1_0 で確認済み）
  2. 三値訓練モデル (BitNet) → 27B クラスが存在しない（待つしかない）
  3. 構造的プルーニング 30% (fine-tuning なし) → 品質壊滅
  4. 非構造的プルーニング 20% → GGUF サイズ不変

  結論: M2 24GB で Qwen3.5-27B を実用速度かつ実用品質で
        動作させる手段は 2026年3月時点で存在しない。
        gemma3:4b を継続使用し、小型 Qwen / 三値訓練モデル公開を待つ。
```

### 今後の方針

```
受動的待機:
  - Qwen3.5-7B/14B 公開 → Q4_K_M で M2 24GB に収まる可能性大
  - 三値訓練 27B 公開 → bitnet.cpp で 5-7 tok/s 期待
  - M4 Pro/Max (48GB+) → Q4_K_M そのまま使用

能動的 (GPU 環境がある場合):
  - 構造的プルーニング 30% + LoRA fine-tuning → 品質回復の可能性
  - Phase 3 (imatrix) + IQ2_XXS → 未検証だが試す価値あり
```

---

## テスト戦略

### ユニットテスト（優先度順）

| テスト | 対象 | 状態 |
|--------|------|------|
| ヘッド重要度スコアリング | Phase 5 | **完了** (57 テスト) |
| チャネル重要度スコアリング | Phase 5 | **完了** |
| 構造的削除後のテンソル次元検証 | Phase 5 | **完了** |
| config.json 更新の整合性 | Phase 5 | **完了** |
| DeltaNet テンソルプルーニング | Phase 5 | **完了** (9 テスト) |
| V-head 整除性検証 | Phase 5 | **完了** (2 テスト) |
| レイヤー除去 interval ガード | Phase 5 | **完了** (2 テスト) |
| safetensors.index.json 再生成 | Phase 5 | **完了** (1 テスト) |
| テンソル分類 (各アーキテクチャ) | Phase 4 | P2 |
| imatrix 生成・キャッシュ | Phase 3 | P1 |
| GGUF メタデータ読取り | Phase 2 | P1 |
| per-channel vs per-tensor scaling | Phase 1 | P3 |
| TQ1_0 ブロックエンコーディング | Phase 1 | P3 |
| BitNet バックエンド検出・カーネル選択 | Phase 6 | P3 |

### 統合テスト（優先度順）

| テスト | 前提条件 | 状態 |
|--------|---------|------|
| ~~小規模モデル (125M) → 構造的プルーニング 40% → Q4_K_M GGUF → Ollama 推論~~ | Ollama | 未実施 |
| **Qwen3.5-27B → 構造的プルーニング 30% → Q4_K_M → M2 24GB ベンチマーク** | llama-server, 42GB+ 空きディスク | **完了 — 品質壊滅** |
| imatrix + IQ2_XXS → 品質比較 | キャリブレーションデータ | P1 |
| Ollama GGUF → 再量子化 → 動作確認 | Ollama | P1 |
| 小規模モデル (125M) → TQ1_0 GGUF → llama-server 推論 | llama.cpp | P3 |
| bitnet-b1.58-2B-4T → bitnet.cpp GGUF → 推論 | bitnet.cpp | P3 |

### ベンチマーク（最終検証）

Qwen3.5-27B 構造的プルーニング後の計測:

| 指標 | 目標 | 実測 | 判定 |
|------|------|------|------|
| **モデルサイズ** | ≤ 10GB | **12GB** (5.25 BPW) | 未達 (q8_0 fallback) |
| **tok/s** | 8-12 | **5-10** | 部分達成 |
| **swap** | 0 | **0** | 達成 |
| **perplexity** | +30% 以内 | **計測不能** (gibberish) | 壊滅 |
| **umigame 正答率** | ≥ 2/3 | **0/3** | 壊滅 |
| **日本語品質** | 良好以上 | **生成不可能** | 壊滅 |
| **隠し情報漏洩** | なし | N/A (応答なし) | N/A |
