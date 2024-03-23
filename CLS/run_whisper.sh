#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --account=rrg-ravanelm
#SBATCH --job-name=w2v2_large
#SBATCH --output=w2v2_large%A_%a.out
#SBATCH --error=w2v2_large%A_%a.err

# sbatch --job-name=whisper-large-v2 --output=whisper-large-v2%A_%a.out --error=whisper-large-v2%A_%a.err /home/adelmou/interspeech23/spoken_medical_qa/BioSpeechQA/src/CLS/run_whisper.sh -e whisper-large-v2 -s openai/whisper-large-v2 -d 1280 -n 10
# sbatch --job-name=whisper-medium --output=whisper-medium%A_%a.out --error=whisper-medium%A_%a.err /home/adelmou/interspeech23/spoken_medical_qa/BioSpeechQA/src/CLS/run_whisper.sh -e whisper-medium -s openai/whisper-medium -d 1024 -n 10
# sbatch --job-name=whisper-small --output=whisper-small%A_%a.out --error=whisper-small%A_%a.err /home/adelmou/interspeech23/spoken_medical_qa/BioSpeechQA/src/CLS/run_whisper.sh -e whisper-small -s openai/whisper-small -d 768 -n 10

# Exit if any command fails and if an undefined variable is used
set -eu
# echo of launched commands
set -x

# Default values
experiment_name="wav2vec2-large"
ssl_hub="facebook/wav2vec2-large"
encoder_dim=1024
number_of_epochs=5

# Generate a random port number
MASTER_PORT=$((RANDOM + 2000))

# Function to display usage information
usage() {
  echo "Usage: $0 [-e|--experiment-name EXPERIMENT_NAME] [-s|--ssl-hub SSL_HUB] [-d|--encoder-dim ENCODER_DIM] [-n|--number-of-epochs NUMBER_OF_EPOCHS]"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--experiment-name)
      experiment_name="$2"
      shift 2
      ;;
    -s|--ssl-hub)
      ssl_hub="$2"
      shift 2
      ;;
    -d|--encoder-dim)
      encoder_dim="$2"
      shift 2
      ;;
    -n|--number-of-epochs)
      number_of_epochs="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

module purge
module load python/3.9.6
module load cuda/11.7
module load ffmpeg/4.2.2

source $HOME/spoken_medical_qa/bin/activate

scp -r $HOME/projects/def-ravanelm/datasets/spoken_medical_wa $SLURM_TMPDIR/

cd $SLURM_TMPDIR/
mkdir SpokenMedicalQA_data
cd SpokenMedicalQA_data

unzip -q $SLURM_TMPDIR/spoken_medical_wa/audios_wav_ffmpeg_concat.zip
unzip -q $SLURM_TMPDIR/spoken_medical_wa/audios_wav_ffmpeg_train_concat.zip
unzip -q $SLURM_TMPDIR/spoken_medical_wa/json+transcripts+dev.zip

cd $HOME/interspeech23/spoken_medical_qa/BioSpeechQA/src/

python transform_json_to_csv.py --input_folder=$SLURM_TMPDIR/SpokenMedicalQA_data/ --output_folder=$SLURM_TMPDIR/SpokenMedicalQA_data/data-30+/

bash prepare_csv.sh

cd CLS

echo "Starting training..."
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_ssl.py hparams/whisper_ssl.yaml --data_folder=$SLURM_TMPDIR/SpokenMedicalQA_data/data-30+/ --experiment_folder=$SCRATCH/results/SpokenMedicalQA/ssl/ --experiment_name=$experiment_name --whisper_hub=$ssl_hub --encoder_dim=$encoder_dim --number_of_epochs=$number_of_epochs --precision=fp16 --find_unused_parameters
echo "Training completed."