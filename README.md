# seq2seq-fingerprint
Seq2Seq Fingerprint

## Installation

We right now depend on the `tensorflow-gpu==0.12.0`. There are huge changes in tensorflow 1.0.1 and we yet have not resource to support it.

## Example Usage


### Decode

#### Sample

```bash
python decode.py sample ~/expr/test/gru-2-256/ ~/expr/seq2seq-fp/pretrain/pm2.vocab ~/expr/seq2seq-fp/pretrain/pm2_10k.tmp --sample_size 500
```

Example output:
```
Loading seq2seq model definition from /home/zhengxu/expr/test/gru-4-256/model.json...
Loading model weights from checkpoint_dir: /home/zhengxu/expr/test/gru-4-256/weights/
: CC(OC1=CC=C2/C3=C(/CCCC3)C(=O)OC2=C1C)C(=O)N[C@@H](CC4=CC=CC=C4)C(O)=O
> CC(OC1=CC=C2/C3=C(/CCCC3)C(=O)OC2=C1C)C(=O)N[C@@H](CC4=CC=CC=C4)C(O)=O

: CSC1=CC=C(CN(C)CC(=O)NC2=C(F)C=CC=C2F)C=C1
> CSC1=CC=C(CN(C)CC(=O)NC2=C(F)C=CC=C2F)C=C1

: CC1\C=C(\C)CC2C1C(=O)N(CCC[N+](C)(C)C)C2=O
> CC1\C=C(\C)CC2C1C(=O)N(CCC[N+](C)(C)C)C2=O

: CCC(=O)C1=CC(F)=C(C=C1)N2CCN(CC2)C(=O)C3=CC=C(F)C=C3
> CCC(=O)C1=CC(F)=C(C=C1)N2CCN(CC2)C(=O)C3=CC=C(F)C=C3

: CC1=CC(=CC=C1)C(=O)NC2=CC(=CC=C2)C(=O)N3CCOCC3
> CC1=CC(=CC=C1)C(=O)NC2=CC(=CC=C2)C(=O)N3CCOCC3
```

#### All FP

```bash
python decode.py fp ~/expr/test/gru-2-256/ ~/expr/seq2seq-fp/pretrain/pm2.vocab ~/expr/seq2seq-fp/pretrain/pm2_10k.tmp ~/expr/test_2.fp
```

Example Output:
```
Progress: 200/10000
Progress: 400/10000
Progress: 600/10000
Progress: 800/10000
Progress: 1000/10000
Progress: 1200/10000
Progress: 1400/10000
Progress: 1600/10000
Progress: 1800/10000
Progress: 2000/10000
Progress: 2200/10000
Progress: 2400/10000
Progress: 2600/10000
Progress: 2800/10000
Progress: 3000/10000
Progress: 3200/10000
Progress: 3400/10000
Progress: 3600/10000
Progress: 3800/10000
Progress: 4000/10000
Progress: 4200/10000
Progress: 4400/10000
Progress: 4600/10000
Progress: 4800/10000
Progress: 5000/10000
Progress: 5200/10000
Progress: 5400/10000
Progress: 5600/10000
Progress: 5800/10000
Progress: 6000/10000
Progress: 6200/10000
Progress: 6400/10000
Progress: 6600/10000
Progress: 6800/10000
Progress: 7000/10000
Progress: 7200/10000
Progress: 7400/10000
Progress: 7600/10000
Progress: 7800/10000
Progress: 8000/10000
Progress: 8200/10000
Progress: 8400/10000
Progress: 8600/10000
Progress: 8800/10000
Progress: 9000/10000
Progress: 9200/10000
Progress: 9400/10000
Progress: 9600/10000
Progress: 9800/10000
Exact match count: 9665/10000
```

### Train

```bash
python train.py train ~/expr/test/gru-2-256/ ~/expr/seq2seq-fp/pretrain/pm2.tokens ~/expr/seq2seq-fp/pretrain/pm2_10k.tokens --batch_size 64
```

Example Output:
```
global step 145600 learning rate 0.1200 step-time 0.314016 perplexity 1.000712
  eval: bucket 0 perplexity 1.001985
  eval: bucket 1 perplexity 1.002438
  eval: bucket 2 perplexity 1.000976
  eval: bucket 3 perplexity 1.002733
global step 145800 learning rate 0.1200 step-time 0.265477 perplexity 1.001033
  eval: bucket 0 perplexity 1.003763
  eval: bucket 1 perplexity 1.001052
  eval: bucket 2 perplexity 1.000259
  eval: bucket 3 perplexity 1.001401
```