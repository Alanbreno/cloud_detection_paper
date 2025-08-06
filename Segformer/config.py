# Encoders disponíveis no Segmentation Models Pytorch

ENCODER_NAME_MIT_B0 = "mit_b0"
ENCODER_NAME_MIT_B1 = "mit_b1"


NAME_MIT_B0 = "Segformer_mit_b0"
NAME_MIT_B1 = "Segformer_mit_b1"


# Hiperparâmetros do modelo
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 32
CLASSES = 4
IN_CHANNELS = 4 # Blue, Green, Red, NIR
ACCELERATOR = "auto"

# Diretórios
DIR_BASE = "/content/drive/MyDrive/taco_CloudSen12/"
DIR_LOG = "/content/drive/MyDrive/Segformer_4_bands_l1c/lightning_logs/"

# Diretórios raiz dos modelos

DIR_ROOT_MIT_B0 = "/content/drive/MyDrive/Segformer_4_bands_l1c/lightning_logs/Segformer_mit_b0"
DIR_ROOT_MIT_B1 = "/content/drive/MyDrive/Segformer_4_bands_l1c/lightning_logs/Segformer_mit_b1"

