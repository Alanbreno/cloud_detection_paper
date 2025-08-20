import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torchmetrics


class CD_Sentinel_2(pl.LightningModule):
    def __init__(self, name, encoder_name, classes, in_channels, learning_rate):
        super().__init__()
        self.model = smp.create_model(
            arch=name,
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            classes=classes,
            in_channels=in_channels,
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        

        metrics = {
            f"{stage}_acuracia": accuracy,
            f"{stage}_dataset_iou": iou,
            f"{stage}_f1_score": f1_score,
        }

        self.log_dict(metrics, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        assert torch.isfinite(y_hat).all(), "Predições no treino contêm valores não finitos (NaN ou Inf)"

        loss = self.loss(y_hat, y)

        output = F.softmax(y_hat, dim=1)
        output = torch.argmax(output, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            output, y, mode="multiclass", num_classes=4
        )

        self.training_step_outputs.append(
            {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        assert torch.isfinite(y_hat).all(), "Predições na validação contêm valores não finitos (NaN ou Inf)"

        loss = self.loss(y_hat, y)

        output = F.softmax(y_hat, dim=1)
        output = torch.argmax(output, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            output, y, mode="multiclass", num_classes=4
        )

        self.validation_step_outputs.append(
            {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        assert torch.isfinite(y_hat).all(), "Predições no teste contêm valores não finitos (NaN ou Inf)"

        loss = self.loss(y_hat, y)

        output = F.softmax(y_hat, dim=1)
        output = torch.argmax(output, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            output, y, mode="multiclass", num_classes=4
        )

        self.test_step_outputs.append(
            {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        self.log("test_loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Define o scheduler para reduzir LR se a perda de validação não diminuir por 4 épocas
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=4, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitora a perda de validação
                "interval": "epoch",  # Aplica o ajuste por época
                "frequency": 1,  # Frequência de monitoramento (1 = a cada época)
            },
        }
        
        
        

class SegformerLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module aprimorado para SegFormer, customizado para detecção de nuvens
    com imagens Sentinel-2 (13 bandas) e o dataset CloudSEN12 (4 classes).

    - Utiliza o encoder mit-b2.
    - Implementa métricas de Acurácia, IoU (Jaccard) e F1-Score.
    - Inclui o passo de teste (`test_step`).
    - Organizado para fácil customização de perda e otimizadores.
    """
    def __init__(self, num_classes=4, in_channels=13, lr=1e-4, warmup_steps=500):
        super().__init__()
        # Salva os hiperparâmetros (serão logados automaticamente pelo Lightning)
        self.save_hyperparameters()

        # 1. Carrega o modelo SegFormer pré-treinado com o encoder B2
        # O modelo base é o `nvidia/segformer-b2-finetuned-ade-512-512`
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=self.hparams.num_classes,
            ignore_mismatched_sizes=True # Permite que a cabeça de classificação seja redimensionada para 4 classes
        )

        # 2. Adapta a primeira camada para aceitar 13 canais de entrada (Sentinel-2)
        # Esta é a etapa crucial para dados multiespectrais.
        original_first_layer = self.model.segformer.encoder.patch_embeddings[0].proj
        
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=self.hparams.in_channels,
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )

        # 3. Define a função de perda (Loss Function)
        # CrossEntropy é um ótimo ponto de partida. Veja a discussão abaixo para outras opções.
        # Para lidar com desbalanceamento de classes, você pode adicionar pesos:
        # class_weights = torch.tensor([0.1, 0.4, 0.4, 0.1]) # Exemplo
        # self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.loss_fn = nn.CrossEntropyLoss()

        # 4. Define as métricas para cada fase (treino, validação, teste)
        # Usar um ModuleDict ajuda a organizar as métricas.
        self.metrics = nn.ModuleDict()
        for phase in ['train', 'val', 'test']:
            self.metrics[phase] = nn.ModuleDict({
                'iou': torchmetrics.JaccardIndex(task="multiclass", num_classes=self.hparams.num_classes),
                'f1': torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes),
                'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
            })

    def forward(self, x):
        return self.model(pixel_values=x).logits

    def _shared_step(self, batch, phase):
        """
        Função auxiliar para evitar repetição de código nos passos de treino, val e teste.
        """
        images, masks = batch
        logits = self(images)
        
        # Redimensiona os logits para o tamanho original da máscara antes de calcular a perda/métrica
        upsampled_logits = F.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )

        # Calcula a perda
        loss = self.loss_fn(upsampled_logits, masks)
        
        # Calcula as métricas
        preds = upsampled_logits.argmax(dim=1)
        self.metrics[phase]['iou'].update(preds, masks)
        self.metrics[phase]['f1'].update(preds, masks)
        self.metrics[phase]['accuracy'].update(preds, masks)

        return loss

    def on_train_epoch_end(self):
        # Loga as métricas ao final de cada época de treino
        metrics = self.metrics['train']
        self.log('train_iou', metrics['iou'].compute(), on_step=False, on_epoch=True)
        self.log('train_f1', metrics['f1'].compute(), on_step=False, on_epoch=True)
        self.log('train_acc', metrics['accuracy'].compute(), on_step=False, on_epoch=True)
        metrics['iou'].reset()
        metrics['f1'].reset()
        metrics['accuracy'].reset()

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'train')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'val')
        self.log('val_loss', loss, prog_bar=True)
    
    def on_validation_epoch_end(self):
        # Loga as métricas ao final de cada época de validação
        metrics = self.metrics['val']
        self.log('val_iou', metrics['iou'].compute(), prog_bar=True)
        self.log('val_f1', metrics['f1'].compute(), prog_bar=True)
        self.log('val_acc', metrics['accuracy'].compute(), prog_bar=True)
        metrics['iou'].reset()
        metrics['f1'].reset()
        metrics['accuracy'].reset()

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'test')
        self.log('test_loss', loss)

    def on_test_epoch_end(self):
        # Loga as métricas ao final do teste
        metrics = self.metrics['test']
        self.log('test_iou', metrics['iou'].compute())
        self.log('test_f1', metrics['f1'].compute())
        self.log('test_acc', metrics['accuracy'].compute())
        metrics['iou'].reset()
        metrics['f1'].reset()
        metrics['accuracy'].reset()


    def configure_optimizers(self):
        """Define o otimizador e o scheduler de learning rate."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        
        # Scheduler: Combinação de aquecimento linear (warmup) com decaimento cosseno
        # Esta é uma estratégia muito eficaz para treinar Transformers.
        def warmup_cosine_decay(current_step):
            if current_step < self.hparams.warmup_steps:
                # Fase de aquecimento linear
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            # Fase de decaimento cosseno
            progress = float(current_step - self.hparams.warmup_steps) / float(max(1, self.trainer.estimated_stepping_batches - self.hparams.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=warmup_cosine_decay),
            "name": "learning_rate",
            "interval": "step", # Atualiza o LR a cada passo de otimização
            "frequency": 1,
        }
        return [optimizer], [scheduler]
