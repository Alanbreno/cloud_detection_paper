import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import CoreDataModule
from model import UNet_CD_Sentinel_2
import metrics
import tacoreader
import torch
import argparse

def main():
    """
    Função principal que configura os argumentos e inicia a lógica de treinamento.
    """
    # 1. INICIALIZAÇÃO DO PARSER
    # O description aparecerá na tela de ajuda (-h)
    parser = argparse.ArgumentParser(
        description="Script para configurar e iniciar o treinamento de um modelo de segmentação de imagens de satélite."
    )

    # 2. DEFINIÇÃO DOS ARGUMENTOS

    # --- Argumentos Obrigatórios (Posicionais) ---

    parser.add_argument(
        "tipo_imagem",
        type=str,
        choices=['l1c', 'l2a'], # O argparse vai garantir que só um desses valores seja aceito
        help="Tipo da imagem de satélite a ser usada (l1c ou l2a)."
    )

    parser.add_argument(
        "bandas_usadas",
        type=int,
        nargs='+', # '+' significa que ele deve capturar 1 ou mais valores em uma lista
        help="Uma lista com os índices das bandas a serem usadas (ex: 2 3 4 8)."
    )

    # --- Argumentos Opcionais ---
    
    parser.add_argument(
        "-m", "--nome_modelo",
        type=str,
        default="unet", # Valor padrão se não for fornecido
        help="Nome do modelo a ser treinado (padrão: 'unet')."
    )
    
    parser.add_argument(
        "-e", "--encoder",
        type=str,
        default="efficientnet-b1",
        help="Tipo do encoder a ser usado."
    )

    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=32, # Valor padrão se não for fornecido
        help="Tamanho do lote (batch size) para o treinamento (padrão: 32)."
    )

    # 3. PROCESSAMENTO DOS ARGUMENTOS
    args = parser.parse_args()

    # 4. VALIDAÇÃO ADICIONAL (Boa prática!)
    # Verifica se os índices das bandas estão no intervalo permitido de 1 a 13
    for banda in args.bandas_usadas:
        if not 1 <= banda <= 13:
            # parser.error() encerra o script com uma mensagem de erro clara
            parser.error(f"Índice de banda inválido: {banda}. As bandas devem estar entre 1 e 13.")


    # 5. EXIBIÇÃO DA CONFIGURAÇÃO FINAL
    # É útil imprimir os parâmetros que serão usados para ter um registro
    print("="*50)
    print("CONFIGURAÇÃO DO TREINAMENTO")
    print("="*50)
    print(f"  -> Tipo da Imagem: {args.tipo_imagem}")
    print(f"  -> Bandas Usadas: {args.bandas_usadas}")
    print(f"  -> Nome do Modelo: {args.nome_modelo}")
    print(f"  -> Encoder: {args.encoder}")
    print(f"  -> Batch Size: {args.batch_size}")
    print("="*50)
    print("\nIniciando o processo de treinamento...\n")

    if args.tipo_imagem == "l2a":
        dataset = tacoreader.load(["/home/mseruffo/taco_CloudSen12/cloudsen12-l2a.0000.part.taco",
                                    "/home/mseruffo/taco_CloudSen12/cloudsen12-l2a.0001.part.taco",
                                    #"/home/mseruffo/taco_CloudSen12/cloudsen12-l2a.0002.part.taco",
                                    #"/scratch/MSERUFFO/taco_CloudSen12/cloudsen12-l2a.0003.part.taco",
                                    #"/scratch/MSERUFFO/taco_CloudSen12/cloudsen12-l2a.0004.part.taco",
                                    #"/scratch/MSERUFFO/taco_CloudSen12/cloudsen12-l2a.0005.part.taco",
                                    ])
        df = dataset[(dataset["label_type"] == "high") & (dataset["real_proj_shape"] == 509)]

    # Numero de bandas usadas no treinamento
    num_bandas = len(args.bandas_usadas)

    # Nome do diretório de saída e log
    name_output = f'{args.tipo_imagem}_{args.nome_modelo}_{args.encoder}_{num_bandas}bandas'
    
    dir_log = '/home/mseruffo/lightning_logs/'
    # Define o tensor board logger
    tb_logger = TensorBoardLogger(dir_log, name=name_output)

    # Define the datamodule
    datamodule = CoreDataModule(
        dataframe=df,
        batch_size=args.batch_size,
        bandas=args.bandas_usadas,
    )
    
    # Define the model
    model = UNet_CD_Sentinel_2(
        name=args.nome_modelo,
        encoder_name=args.encoder,
        classes=4,
        in_channels=num_bandas,
        learning_rate=1e-3,
    )
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=dir_log+name_output,
    filename="{epoch}-{train_loss:.2f}-{val_loss:.2f}-trainHigh512",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    )

    earlystopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, mode="min"
    )

    callbacks = [checkpoint_callback, earlystopping_callback]
    
    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=100,
        log_every_n_steps=1,
        callbacks=callbacks,
        accelerator="auto",
        precision="16-mixed",
        logger=tb_logger,
        default_root_dir=dir_log+name_output,
    )
    
    torch.set_float32_matmul_precision('medium')
    # Start the training
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()

