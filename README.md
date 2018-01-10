This code implement sequence to sequence fingerprint The Zinc dataset can be used.

Installation requirements
  1.We right now depend on the tensorflow==1.4.1/tensorflow-gpu==1.3.0. #fix me.
  2.smile is required(for Ubuntu OS, pip install smile)

References:
Zheng Xu, Sheng Wang, Feiyun Zhu, and Junzhou Huang,2017, Seq2seq Fingerprint: An Unsupervised Deep MolecularEmbedding for Drug Discovery,BCB’17, Aug 2017, Boston, Massachusetts USA

Directory structure:
/unsupervised - source files with Python 2.7 grammar
/data         - smile data we use (/smile/nfs/projects/nih_drug/data/ was used in our example)
/demos        - bash script be used to submit jobs and running the program

Input and output path and files:
smi_path   /smile/nfs/projects/nih_drug/data/pm2/pm2.smi  (input smile data for building vocab)
vocab_path ~/expr/seq2seq-fp/pretrain/pm2.vocab           (the path and file to save vocab)
out_path ~/expr/seq2seq-fp/pretrain/pm2.tokens            (the path and file to save tokens)
tmp_path ~/expr/seq2seq-fp/pretrain/pm2.tmp               (the path and file to temporary data)


Running workflow:

1. Prepare data

a) Build vocabulary

 Use the build_vocab switch to turn on building vocabulary functionality.

   python -m unsupervised.data --build_vocab 1 --smi_path /smile/nfs/projects/nih_drug/data/pm2/pm2.smi --vocab_path ~/expr/seq2seq-fp/pretrain/pm2.vocab --out_path ~/expr/seq2seq-fp/pretrain/pm2.tokens --tmp_path ~/expr/seq2seq-fp/pretrain/pm2.tmp

Example Output:

Creating temp file...
Building vocabulary...
Creating vocabulary /home/username/expr/test/pretrain/pm2.vocab from data /tmp/tmpcYVqV0
  processing line 100000
  processing line 200000
  processing line 300000
Translating vocabulary to tokens...
Tokenizing data in /tmp/tmpcYVqV0
  tokenizing line 100000
  tokenizing line 200000
  tokenizing line 300000

b) If vocabulary already exsits
  Translate the SMI file using existing vocabulary
  Switch off build_vocab option, or simply hide it from the command line.

python -m unsupervised.data --smi_path /smile/nfs/projects/nih_drug/data/logp/logp.smi --vocab_path ~/expr/seq2seq-fp/pretrain/pm2.vocab --out_path ~/expr/seq2seq-fp/pretrain/logp.tokens --tmp_path ~/expr/seq2seq-fp/pretrain/logp.tmp

Example Output:

Creating temp file...
Reading vocabulary...
Translating vocabulary to tokens...
Tokenizing data in /tmp/tmpmP8R_P

2. Train
python train.py train ~/expr/test/gru-2-256/ ~/expr/seq2seq-fp/pretrain/pm2.tokens ~/expr/seq2seq-fp/pretrain/pm2_10k.tokens --batch_size 64
Example Output:

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

From fresh
(Note: run step 1 again and generate new weights e.g.:put them in unsup-seq2seq/models/gru-2-128)
python train.py train ~/expr/unsup-seq2seq/models/gru-2-128/ ~/expr/unsup-seq2seq/data/pm2.tokens ~/expr/unsup-seq2seq/data/logp.tokens --batch_size 256
python train.py train ~/expr/unsup-seq2seq/models/gru-3-128/ ~/expr/unsup-seq2seq/data/pm2.tokens ~/expr/unsup-seq2seq/data/logp.tokens --batch_size 256
python train.py train ~/expr/unsup-seq2seq/models/gru-2-256/ ~/expr/unsup-seq2seq/data/pm2.tokens ~/expr/unsup-seq2seq/data/logp.tokens --batch_size 256 --summary_dir ~/expr/unsup-seq2seq/models/gru-2-256/summary/
Example output
global step 200 learning rate 0.5000 step-time 0.317007 perplexity 7.833510
  eval: bucket 0 perplexity 32.720735
  eval: bucket 1 perplexity 24.253715
  eval: bucket 2 perplexity 16.619440
  eval: bucket 3 perplexity 13.854121
global step 400 learning rate 0.5000 step-time 0.259872 perplexity 6.460571
  eval: bucket 0 perplexity 31.408722
  eval: bucket 1 perplexity 22.750650
  eval: bucket 2 perplexity 15.665839
  eval: bucket 3 perplexity 12.682373


3. Decode
# model.json and weights in the subdirectory of ~/expr/test/gru-2-256/ are necessary to run decode
python decode.py sample ~/expr/test/gru-2-256/  ~/expr/seq2seq-fp/pretrain/pm2.vocab ~/expr/seq2seq-fp/pretrain/pm2_10k.tmp --sample_size 500

Example output:

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
All FP

python decode.py fp ~/expr/test/gru-2-256/ ~/expr/seq2seq-fp/pretrain/pm2.vocab ~/expr/seq2seq-fp/pretrain/pm2_10k.tmp ~/expr/test_2.fp

Example Output:

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


4. Generate all fingerprints for logp data

python -m unsupervised.pretrain --get_fp 1 --dev_file logp.tmp --fp_file logp.fp
Sample Output:

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

Note: you can use the following example to debug
Debug

python train.py train ~/expr/unsup-seq2seq/models/debug/ ~/expr/unsup-seq2seq/data/pm2.tokens ~/expr/unsup-seq2seq/data/logp.tokens --batch_size 32 --summary_dir ~/expr/unsup-seq2seq/models/debug/summary/
Debug-Acc

python train.py train ~/expr/unsup-seq2seq/models/debug-acc/ ~/expr/seq2seq-fp/pretrain/pm2.tokens ~/expr/seq2seq-fp/pretrain/pm2_10k.tokens --batch_size 32 --summary_dir ~/expr/unsup-seq2seq/models/debug-acc/summary/


