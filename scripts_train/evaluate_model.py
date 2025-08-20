import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import tacoreader

# Importa as suas classes customizadas
from model import SegformerLightningModule
from datamodule import CoreDataModule # Certifique-se que o nome do arquivo está correto

from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, accuracy_score

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Renderiza e salva a matriz de confusão como uma imagem.
    A matriz é normalizada para mostrar percentuais (revocação por classe).
    """
    # Normaliza a matriz de confusão pela linha (representa a revocação de cada classe)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt=".2%",  # Formato de porcentagem com 2 casas decimais
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title("Matriz de Confusão Normalizada (Revocação por Classe)")
    plt.ylabel('Classe Verdadeira (Ground Truth)')
    plt.xlabel('Classe Predita pelo Modelo')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Matriz de confusão salva em: {output_path}")

def main(args):
    """
    Função principal para carregar o modelo e avaliar as métricas.
    """
    print("="*50)
    print("INICIANDO A AVALIAÇÃO DO MODELO")
    print(f"-> Checkpoint: {args.checkpoint_path}")
    print("="*50)

    # --- 1. Carregar Modelo e Dados ---
    print("1. Carregando modelo e datamodule...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carrega o modelo a partir do checkpoint
    model = SegformerLightningModule.load_from_checkpoint(args.checkpoint_path).to(device)
    model.eval() # Coloca o modelo em modo de avaliação

    dataset = tacoreader.load([ r"/workspace/cloudsen12-l1c.0000.part.taco",
                                r"/workspace/cloudsen12-l1c.0001.part.taco",
                                r"/workspace/cloudsen12-l1c.0002.part.taco",
                                r"/workspace/cloudsen12-l1c.0003.part.taco",
                                r"/workspace/cloudsen12-l1c.0004.part.taco",
                                ])
    df = dataset[(dataset["label_type"] == "high") & (dataset["real_proj_shape"] == 509)]
    
    
    # Assume que o datamodule está configurado com os mesmos dados do treinamento
    # Você pode precisar ajustar os argumentos aqui se o seu datamodule precisar
    datamodule = CoreDataModule(
        dataframe=df, # O dataframe será carregado dentro do setup
        batch_size=args.batch_size,
            bandas=[2, 3, 4, 8] 
        )
    datamodule.setup('test') # Configura o conjunto de teste
    test_loader = datamodule.test_dataloader()
    
    # --- 2. Realizar Inferência ---
    print("2. Realizando inferência no conjunto de teste...")
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Desativa o cálculo de gradientes para acelerar
        for batch in tqdm(test_loader, desc="Avaliando"):
            images, labels = batch
            images = images.to(device)

            logits = model(images)
            upsampled_logits = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            preds = torch.argmax(upsampled_logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # --- 3. Preparar Dados para Scikit-learn ---
    print("3. Consolidando predições e preparando para análise...")
    # Concatena os resultados de todos os lotes em um único tensor
    preds_tensor = torch.cat(all_preds)
    labels_tensor = torch.cat(all_labels)
    
    # Converte para NumPy e achata para um vetor 1D
    preds_np = preds_tensor.numpy().flatten()
    labels_np = labels_tensor.numpy().flatten()

    # --- 4. Calcular Métricas e Gerar Relatórios ---
    print("4. Calculando métricas com Scikit-learn...")

    # Gera o relatório de classificação detalhado
    report = classification_report(
        labels_np, 
        preds_np, 
        target_names=args.class_names, 
        digits=4
    )

    # Calcula métricas gerais
    overall_accuracy = accuracy_score(labels_np, preds_np)
    iou_macro = jaccard_score(labels_np, preds_np, average='macro')
    iou_weighted = jaccard_score(labels_np, preds_np, average='weighted')

    # Monta o relatório final
    final_report = f"""
======================================================
        RELATÓRIO DE AVALIAÇÃO FINAL
======================================================
Checkpoint: {os.path.basename(args.checkpoint_path)}

MÉTRICAS GERAIS:
--------------------------------
Acurácia Geral: {overall_accuracy:.4f}
IoU (Macro):    {iou_macro:.4f}
IoU (Ponderado):{iou_weighted:.4f}

RELATÓRIO DETALHADO POR CLASSE:
--------------------------------
{report}
"""
    print(final_report)

    # --- 5. Gerar Matriz de Confusão ---
    print("5. Gerando matriz de confusão...")
    cm = confusion_matrix(labels_np, preds_np)
    
    # Cria o diretório de saída se ele não existir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define os nomes dos arquivos de saída
    base_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    report_path = os.path.join(args.output_dir, f"report_{base_name}.txt")
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{base_name}.png")

    # --- 6. Salvar Resultados ---
    with open(report_path, "w") as f:
        f.write(final_report)
    print(f"\nRelatório de classificação salvo em: {report_path}")

    plot_confusion_matrix(cm, args.class_names, cm_path)

    print("\nAnálise concluída com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de avaliação de modelo de segmentação.")
    parser.add_argument(
        "checkpoint_path", 
        type=str, 
        help="Caminho para o arquivo de checkpoint do modelo (.ckpt)."
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs='+',
        default=['Céu Limpo', 'Nuvem Espessa', 'Nuvem Fina', 'Sombra'],
        help="Nomes das classes para os relatórios e gráficos (na ordem correta)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Tamanho do lote para a inferência."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Diretório para salvar os relatórios e gráficos gerados."
    )
    
    args = parser.parse_args()
    main(args)