#!/usr/bin/env/python3
""" 

Authors
 * Adel Moumen 2024
"""

import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
import torchaudio
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
import os 
from speechbrain.utils.streaming import split_fixed_chunks, split_wav_lens

logger = logging.getLogger(__name__)

NO_ANSWER = False 
duration_name = ""
wav_name = ""
if NO_ANSWER:
    duration_name = "duration_no_answer"
    wav_name = "wav_no_answer"
else:
    duration_name = "duration"
    wav_name = "wav"
n_of_frames_to_take = 16_000 * 30

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # wavs = split_fixed_chunks(wavs, n_of_frames_to_take, dim=1)[-1]
        if wavs.size(1) > n_of_frames_to_take:
            size_elements = wavs.size(1) - n_of_frames_to_take
            wavs = wavs[:, size_elements:]

        # Forward pass
        feats = self.modules.weighted_ssl_model(wavs)
        
        # last dim will be used for AdaptativeAVG pool
        outputs = self.hparams.avg_pool(feats, wav_lens)
        outputs = outputs.view(outputs.shape[0], -1)

        # outputs = self.modules.enc(outputs)
        outputs = self.modules.classifier_lin(outputs)
        outputs = self.hparams.log_softmax(outputs)
        # print(outputs.shape) # 8, 306, 4
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        input_class_id, _ = batch.input_class_encoder
        input_class_id = input_class_id.squeeze(1)
        loss = self.hparams.compute_cost(predictions, input_class_id)
        
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, input_class_id)

        return loss


    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stats["error_rate"]
            )
            old_lr_weights, new_lr_weights = self.hparams.lr_annealing_weights(
                stats["error_rate"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.weights_optimizer, new_lr_weights
            )
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_model": old_lr_model},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            if if_main_process():
                with open(self.hparams.output_file, "w") as w:
                    self.error_metrics.write_stats(w)


    def init_optimizers(self):
        "Initializes the weights optimizer and model optimizer"
        if hasattr(self.modules.weighted_ssl_model, "module"):
            weights_module = self.modules.weighted_ssl_model.module.weights
        else:
            weights_module = self.modules.weighted_ssl_model.weights
        
        self.weights_optimizer = self.hparams.weights_opt_class(
            [weights_module]
        )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
            "weights_optimizer": self.weights_optimizer,
        }
        # Initializing the weights
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
            self.checkpointer.add_recoverable(
                "weights_opt", self.weights_optimizer
            )


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key=duration_name,
            key_max_value={"duration": 30},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key=duration_name,
            reverse=True,
            key_max_value={"duration": 30},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it i s faster to validate
    valid_data = valid_data.filtered_sorted(sort_key=duration_name)

    # test is separate
    test_datasets = {}
    from pathlib import Path
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        print(name)
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key=duration_name
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes(wav_name)
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="class",
    )

    # ind2lab = label_encoder.ind2lab
    # vocab_list = [ind2lab[x] for x in range(len(ind2lab))] # A, B, C, D

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("class")
    @sb.utils.data_pipeline.provides("input_class", "input_class_encoder")
    def text_pipeline(input_class):
        yield input_class
        input_class_encoded = label_encoder.encode_label_torch(input_class)
        yield input_class_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "input_class_encoder"],
    )

    return train_data, valid_data, test_datasets, label_encoder


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
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

    train_data, valid_data, test_datasets, tokenizer = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected()

    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    import os
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)
    for k in test_datasets.keys(): 
        print(k)
        asr_brain.hparams.output_file = os.path.join(hparams["output_wer_folder"], f"{k}.txt")
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="error_rate",
        )

