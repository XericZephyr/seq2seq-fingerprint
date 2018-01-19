local job_tmpl = import "job.libsm";
local train_job = job_tmpl {
  params: {
    train_dir: "~/expr/test/gru-2-128/",
    train_data_file: "~/expr/seq2seq-fp/pretrain/pm2.tokens",
    eval_data_file: "~/expr/seq2seq-fp/pretrain/pm2.tokens",
    batch_size: 5
  },
  script_command: "train",
  args: {
    batch_size: $.params.batch_size
  },
  binary: "python train.py",
  args_extra: "%s %s %s %s" % [
      self.script_command, self.params.train_dir, self.params.train_data_file,
      self.params.eval_data_file],
  cmd: "%s %s %s" % [self.binary, self.args_extra, self.flag_string]
};
train_job
