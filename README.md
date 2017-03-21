# seq2seq-fingerprint
Seq2Seq Fingerprint

## Example Usage


### Decode

```bash
python decode.py sample ~/expr/test/pretrain/pm2_10k.tmp ~/expr/test/gru-4-256/ ~/expr/seq2seq-fp/pretrain/pm2.vocab --sample_size 100 
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