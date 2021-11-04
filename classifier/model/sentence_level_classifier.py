from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.nn import util
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from spacecutter.models import OrdinalLogisticModel, LogisticCumulativeLink
from spacecutter.losses import CumulativeLinkLoss
from allennlp.training.metrics import CategoricalAccuracy, MeanAbsoluteError, FBetaMeasure
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from ..utils import MULTI_LABEL_TO_INDEX


@Model.register('sentence_level_classifier')
class SentenceLevelClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = len(MULTI_LABEL_TO_INDEX)
        self.classifier = nn.Linear(encoder.get_output_dim(), num_labels)
        self.ordinal_logistic = LogisticCumulativeLink(num_classes=num_labels)
        self.accuracy = CategoricalAccuracy()
        self.mar = MeanAbsoluteError()
        self.fbeta = FBetaMeasure(labels=list(MULTI_LABEL_TO_INDEX.values()))

    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        output = {}
        token_embeds = self.embedder(text)
        mask = util.get_text_field_mask(text)
        encoding = self.encoder(token_embeds, mask=mask)
        logits = self.classifier.forward(encoding)
        reshaped = logits.reshape(-1, 1)
        probs = self.ordinal_logistic.forward(reshaped)
        output['probs'] = probs
        if label is not None:
            loss = CumulativeLinkLoss().forward(logits, torch.unsqueeze(label, dim=-1))
            output['loss'] = loss
            self.classifier.apply(self.ascension_callback())
            self.accuracy(logits, label)
            self.fbeta(logits, label)
            self.mar(logits.argmax(dim=1), label)
        return output

    def ascension_callback(margin=0.0, min_val=-1.0e6):

        def _clip(module):
            if isinstance(module, LogisticCumulativeLink):
                cutpoints = module.cutpoints.data
                for i in range(cutpoints.shape[0] - 1):
                    cutpoints[i].clamp_(
                        min_val, cutpoints[i + 1] - margin
                    )

        return _clip

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self.accuracy.get_metric(
            reset), 'mar': self.mar.get_metric(reset), **self.fbeta.get_metric(reset)}
        return metrics
