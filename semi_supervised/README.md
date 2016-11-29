# Prepare data

```bash
python -m semi_supervised.data
```


# Train 

```bash
python -m semi_supervised.pretrain
```

You can also do pretrain on both development and training set.
```bash
python -m semi_supervised.pretrain --train_with_dev
```

Sample Output:

```
global step 94200 learning rate 0.1849 step-time 0.38 perplexity 1.000061
  eval: bucket 0 perplexity 1.11
  eval: bucket 1 perplexity 1.07
  eval: bucket 2 perplexity 1.06
  eval: bucket 3 perplexity 1.08
global step 94400 learning rate 0.1849 step-time 0.36 perplexity 1.000107
  eval: bucket 0 perplexity 1.09
  eval: bucket 1 perplexity 1.08
  eval: bucket 2 perplexity 1.06
  eval: bucket 3 perplexity 1.09
```

# Decode Random Samples

```bash
python -m semi_supervised.pretrain --decode 1 --decode_size 50
```

Sample Output:

```
: CC(=Cc1ccccc1)C(=O)O
> CC(=C1cccc1cccc1)C(=

: Cc1ccccc1C(=O)O
> Cc1cccc1C(=O)OC

: CCCN(CCCC(NC(=O)C)C(=O)NCc1ccccc1)C(=O)NC
> CCCN(CCCC(NC(=O)C)C(=O)NC1cccc1)C(=O)NC

: NNc1ccc(cc1N(=O)=O)N(=O)=O
> NN1ccc1c(N=O)(=O)N=C(O)=O)=C(C)=C1

: COc1ccc(CCNCC(O)COc2cccc(C)c2)cc1OC
> COc1cc(CCNCCNC(OC)CO2cccc(C)c2)c1OC

: CCOP(=S)(OCC)Oc1ccc(cc1)N(=O)=O
> CCOP(=O)(COC)O1ccc(c1)N(=O)=O

: OCC1OC(O)C(NC(=O)N(CCCl)N=O)C(O)C1O
> OCC1OC(O)C(NC(=O)N(CCCl)N=C(O)C(O)C1O

: Nc1c(Cl)c(Oc2ccccc2)ccc1N(=O)=O
> Nc1C(c)cc2cccccccc)ccc1N(=O)=O

: Nc1nc2[nH]cnc2c(=O)[nH]1
> Nc12c[[HH]2[NH]2[NH]2[NH]2[NH]2c[NH]2c(O

: ICCCI
> CCCCCC

Exact match: 0/10
```

# Generate all fingerprints for logp data

```bash
python -m semi_supervised.pretrain --get_fp 1
```

Sample Output:

```
Reading model parameters from /tmp/seq2seq-fp/pretrain/train/seq2seq_pretrain.ckpt-94000
Progress: 200/10851
Progress: 400/10851
Progress: 600/10851
Progress: 800/10851
Progress: 1000/10851
[omit several lines...]
Progress: 10400/10851
Progress: 10600/10851
Progress: 10800/10851
Exact match: 8086/10851
```