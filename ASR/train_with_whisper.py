#!/usr/bin/env python3
"""Recipe for training a whisper-based ASR system with CommonVoice.
The system employs whisper from OpenAI (https://cdn.openai.com/papers/whisper.pdf).
This recipe take the whisper encoder-decoder to fine-tune on.

To run this recipe, do the following:
> python train_with_whisper.py hparams/train_<locale>_hf_whisper.yaml

 * Adel Moumen 2024
"""

import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from speechbrain.utils.data_utils import undo_padding
from hyperpyyaml import load_hyperpyyaml
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from speechbrain.utils.streaming import split_fixed_chunks, split_wav_lens

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos

        # Stage.TEST torch.Size([4, 98688])
        # print(stage, wavs.shape)
        # print(wavs.shape[1] / 16000)

        if stage != sb.Stage.TRAIN:
            chunk_size_of_30_seconds = 30 * 16000
            chunks = split_fixed_chunks(wavs, chunk_size_of_30_seconds, dim=1)
            chunked_wav_lens = split_wav_lens([c.size(1) for c in chunks], wav_lens)

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            bos_tokens = self.hparams.wav_augment.replicate_labels(bos_tokens)

        hyps = None
        if stage == sb.Stage.TRAIN:
            # We compute the padding mask and replace the values with the pad_token_id
            # that the Whisper decoder expect to see.
            abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
            pad_mask = (
                torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
                < abs_tokens_lens[:, None]
            )
            bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

            # Forward encoder + decoder
            enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)

        else:
            # print("HERE")
            hyps = None
            logits = None
            inp_tokens = [self.tokenizer.prefix_tokens] * wavs.shape[0]
            inp_tokens = torch.tensor(inp_tokens, device="cuda")
            for chunk, wav_lens in zip(chunks, chunked_wav_lens):

                # Forward encoder + decoder
                enc_out, chunk_logits, _ = self.modules.whisper(chunk, inp_tokens)

                hyp, _, _, _ = self.hparams.searcher(
                    enc_out.detach(), wav_lens
                )
                
                if hyps is None:
                    hyps = hyp
                else:
                    # add the hyps to the previous hyps
                    hyps = [h + hyp[i] for i, h in enumerate(hyps)]
                # predicted_words = self.tokenizer.batch_decode(
                #    hyps, skip_special_tokens=True
                # )
                # print("predicted_words = ", predicted_words[0])
                if logits is None:
                    logits = chunk_logits
                else:
                    logits = torch.cat([logits, chunk_logits], dim=1)
        
        return logits, hyps, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        logits, hyps, _, = predictions
        batch = batch.to(self.device)
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # Augment Labels
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
            tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
                tokens_eos_lens
            )

        log_probs = self.hparams.log_softmax(logits)

        # torch.Size([12, 269])
        # torch.Size([12])
        # torch.Size([12, 538, 51865])

        # We compute the padding mask and replace the values with the pad_token_id
        if tokens_eos.shape != log_probs.shape:
            # pad tokens_eos to match log_probs
            total_tokens = log_probs.shape[1] - tokens_eos.shape[1]
            # print(tokens_eos.shape)
            tokens_eos_length = tokens_eos.size(1) * tokens_eos_lens
            # print(tokens_eos_length)
            # exit()
            tokens_eos = torch.nn.functional.pad(
                tokens_eos, (0, total_tokens)
            )
            
            # tokens_eos_lens is the relative length of tokens_eos
            # print(tokens_eos.shape)
            # print(total_tokens)
            # print(tokens_eos_lens)
            tokens_eos_lens = tokens_eos_length / tokens_eos.shape[1]
            # print(tokens_eos_lens)
            # exit()
            

        # abs_tokens_lens = (tokens_eos_lens * tokens_eos.shape[1]).long()
        # pad_mask = (
        #     torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
        #     < abs_tokens_lens[:, None]
        # )
        # tokens_eos[~pad_mask] = self.tokenizer.pad_token_id

        loss = self.hparams.nll_loss(
            log_probs, tokens_eos, length=tokens_eos_lens,
        )

        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens

            
            #  hyps = [hyp[0] if len(hyp) > 0 else [] for hyp in hyps]
                        
            # Decode token terms to words
            predicted_words = self.tokenizer.batch_decode(
                hyps, skip_special_tokens=True
            )

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                target_words, skip_special_tokens=True
            )

            predicted_words = [
                self.tokenizer._normalize(text).split(" ")
                for text in predicted_words
            ]

            target_words = [
                self.tokenizer._normalize(text).split(" ")
                for text in target_words


            ]

            print("predicted_words = ", " ".join(predicted_words[0]))
            print("predicted_words = ", predicted_words[0])
            print("target_words = ", target_words[0])
            print(loss)
            exit()
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            old_lr_whisper, new_lr_whisper = self.hparams.lr_annealing_whisper(
                stage_stats["loss"]
            )

            sb.nnet.schedulers.update_learning_rate(
                self.optimizer, new_lr_whisper
            )
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_whisper": old_lr_whisper},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = "/local_disk/ether/ylabrak/SpokenMedicalQA/"

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    from pathlib import Path
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration",
            # reverse=True,
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        wrd = tokenizer._normalize(wrd)
        yield wrd
        tokens_list = tokenizer.encode(wrd)
        # avoid bos and eos tokens.
        tokens_list = tokens_list[1:-1]
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens"],
    )


    return train_data, valid_data, test_datasets


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer
    language = LANGUAGES[hparams["locale"]]

    tokenizer.set_prefix_tokens(language, "transcribe", False)

    # we need to prepare the tokens for searchers
    hparams["searcher"].set_decoder_input_tokens(tokenizer.prefix_tokens)
    hparams["searcher"].set_language_token(tokenizer.prefix_tokens[1])

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected()

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )


    # Testing
    import os
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    for k in test_datasets.keys(): 
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="WER",
        )
