from lightning_model import SegformerLightningModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from datamodule import CoreDataModule
import tacoreader
import torch
import argparse

def main():
    """
    Função principal que configura os argumentos e inicia a lógica de treinamento.
    """
    # 1. INICIALIZAÇÃO DO PARSER
    parser = argparse.ArgumentParser(
        description="Script para configurar e iniciar o treinamento do SegFormer para segmentação de nuvens."
    )

    # 2. DEFINIÇÃO DOS ARGUMENTOS

    # --- Argumentos Obrigatórios (Posicionais) ---
    parser.add_argument(
        "tipo_imagem", type=str, choices=['l1c', 'l2a'],
        help="Tipo da imagem de satélite a ser usada (l1c ou l2a)."
    )
    parser.add_argument(
        "bandas_usadas", type=int, nargs='+',
        help="Uma lista com os índices das bandas a serem usadas (ex: 2 3 4 8)."
    )

    # --- Argumentos Opcionais (MODIFICADOS para o SegFormer) ---
    parser.add_argument(
        "-sf", "--segformer_backbone",
        type=str,
        default="b2",
        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
        help="Tamanho do backbone MiT do SegFormer a ser usado (padrão: 'b2')."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=8,
        help="Tamanho do lote (batch size) para o treinamento (padrão: 8)."
    )
    # NOVO: Argumentos para controlar hiperparâmetros do SegformerLightningModule
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Taxa de aprendizado inicial para o otimizador (padrão: 0.0001)."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500,
        help="Número de passos de aquecimento (warmup) para o scheduler da learning rate (padrão: 500)."
    )

    # 3. PROCESSAMENTO DOS ARGUMENTOS
    args = parser.parse_args()

    # 4. VALIDAÇÃO ADICIONAL
    for banda in args.bandas_usadas:
        if not 1 <= banda <= 13:
            parser.error(f"Índice de banda inválido: {banda}. As bandas devem estar entre 1 e 13.")

    # 5. EXIBIÇÃO DA CONFIGURAÇÃO FINAL
    print("="*50)
    print("CONFIGURAÇÃO DO TREINAMENTO - SEGFORMER")
    print("="*50)
    print(f"  -> Tipo da Imagem: {args.tipo_imagem}")
    print(f"  -> Bandas Usadas: {args.bandas_usadas} (Total: {len(args.bandas_usadas)})")
    print(f"  -> Backbone SegFormer: MiT-{args.segformer_backbone}")
    print(f"  -> Batch Size: {args.batch_size}")
    print(f"  -> Learning Rate: {args.learning_rate}")
    print(f"  -> Warmup Steps: {args.warmup_steps}")
    print("="*50)
    print("\nIniciando o processo de treinamento...\n")

    # Lógica de carregamento do dataset (mantida como estava)
    if args.tipo_imagem == "l2a":
        dataset = tacoreader.load([ r"/workspace/cloudsen12-l2a.0000.part.taco",
                                    r"/workspace/cloudsen12-l2a.0001.part.taco",
                                    r"/workspace/cloudsen12-l2a.0003.part.taco",
                                    r"/workspace/cloudsen12-l2a.0004.part.taco",
                                    r"/workspace/cloudsen12-l2a.0005.part.taco",
                                    ])
        df = dataset[(dataset["label_type"] == "high") & (dataset["real_proj_shape"] == 509)]
    elif args.tipo_imagem == "l1c":
        dataset = tacoreader.load([ r"/workspace/cloudsen12-l1c.0000.part.taco",
                                    r"/workspace/cloudsen12-l1c.0001.part.taco",
                                    r"/workspace/cloudsen12-l1c.0002.part.taco",
                                    r"/workspace/cloudsen12-l1c.0003.part.taco",
                                    r"/workspace/cloudsen12-l1c.0004.part.taco",
                                    ])
        df = dataset[(dataset["label_type"] == "high") & (dataset["real_proj_shape"] == 509)]

    num_bandas = len(args.bandas_usadas)
    
    # MODIFICADO: Nome do diretório de saída para refletir o novo modelo
    name_output = f'{args.tipo_imagem}_segformer_mit-{args.segformer_backbone}_{num_bandas}bandas'
    dir_log = r'./lightning_logs/'
    tb_logger = CSVLogger(dir_log, name=name_output)

    # Datamodule (mantido como estava)
    datamodule = CoreDataModule(
        dataframe=df,
        batch_size=args.batch_size,
        num_workers = 4,
        bandas=args.bandas_usadas,
    )
    
    # MODIFICADO: Instanciação do novo modelo SegformerLightningModule
    # O nome do backbone é passado para o construtor do modelo
    model_name = f"nvidia/segformer-{args.segformer_backbone}-finetuned-ade-512-512"

    model = SegformerLightningModule(
        model_name=model_name,
        num_classes=4,
        in_channels=num_bandas,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
    )
    
    # MODIFICADO: Callbacks otimizados para métrica de segmentação (IoU)
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_log + name_output,
        filename="{epoch}-{val_iou:.4f}-{val_loss:.2f}",
        monitor="val_iou", # Monitora a métrica de IoU, que é mais relevante para segmentação
        mode="max",      # Queremos maximizar o IoU
        save_top_k=1,
    )
    earlystopping_callback = EarlyStopping(
        monitor="val_iou", # Também monitora o IoU
        patience=15,       # Aumentei a paciência, pois o IoU pode flutuar um pouco mais que a loss
        mode="max"
    )
    callbacks = [checkpoint_callback, earlystopping_callback]
    
    # Trainer (mantido como estava)
    trainer = pl.Trainer(
        max_epochs=100,
        log_every_n_steps=10, # Logar a cada 10 passos é mais razoável
        callbacks=callbacks,
        accelerator="auto",
        precision="16-mixed",
        logger=tb_logger,
    )
    
    torch.set_float32_matmul_precision('high')
    # Inicia o treinamento
    trainer.fit(model=model, datamodule=datamodule)
    
    print("\nTreinamento concluído. Carregando o melhor modelo para validação e teste...")
    
    # MODIFICADO: Carrega o melhor checkpoint usando o novo nome da classe
    # Os hiperparâmetros são carregados automaticamente do checkpoint, não precisa passá-los de novo
    best_model = SegformerLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    # Executa a validação com o melhor modelo
    print("\nExecutando validação...")
    val_metrics = trainer.validate(model=best_model, datamodule=datamodule, verbose=True)
    print("Métricas de Validação:", val_metrics)

    # Executa o teste com o melhor modelo
    print("\nExecutando teste...")
    test_metrics = trainer.test(model=best_model, datamodule=datamodule, verbose=True)
    print("Métricas de Teste:", test_metrics)

    # MODIFICADO: Salva o modelo treinado (apenas o modelo Hugging Face interno)
    print(f"\nSalvando o modelo final em: {dir_log + name_output}/final_model")
    final_model_to_save = best_model.model
    final_model_to_save.save_pretrained(dir_log + name_output + "/final_model")


if __name__ == "__main__":
    main()