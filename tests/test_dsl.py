import pytest
import torch

import topmost
from topmost.data.dsl_dataset import DSLDataset

DOCS = [
    "the hockey team won the game in overtime last night",
    "the senate passed the budget bill after a long debate",
    "new graphics card drivers improve game performance on windows",
    "the patient was given antibiotics for the infection",
] * 6


@pytest.fixture(scope="module")
def dataset():
    preprocess = topmost.Preprocess(vocab_size=30, verbose=False)
    return DSLDataset(DOCS, preprocess, model_name="sshleifer/tiny-gpt2", batch_size=8, device="cpu")


def test_dataset_layout(dataset):
    n, v = len(DOCS), dataset.vocab_size
    assert dataset.soft_labels.shape == (n, v)
    assert torch.allclose(torch.tensor(dataset.soft_labels.sum(1)), torch.ones(n), atol=1e-4)
    assert dataset.train_data.shape == (n, v + dataset.contextual_embed_size)


@pytest.mark.parametrize("model_cls", [topmost.ProdLDA, topmost.DecTM, topmost.ECRTM])
def test_models_train_on_soft_labels(dataset, model_cls):
    model = model_cls(dataset.vocab_size, num_topics=3, contextual_embed_size=dataset.contextual_embed_size,
                      temperature=dataset.temperature)
    trainer = topmost.BasicTrainer(model, dataset, epochs=2, num_top_words=5)
    top_words, theta = trainer.train()
    assert len(top_words) == 3 and theta.shape == (len(DOCS), 3)


def test_combinedtm_trains_on_soft_labels(dataset):
    model = topmost.CombinedTM(dataset.vocab_size, dataset.contextual_embed_size, num_topics=3)
    top_words, theta = topmost.BasicTrainer(model, dataset, epochs=2, num_top_words=5).train()
    assert theta.shape == (len(DOCS), 3)


def test_bow_models_unchanged():
    x = torch.rand(8, 30)
    for model_cls in (topmost.ProdLDA, topmost.DecTM, topmost.ECRTM):
        assert model_cls(30, num_topics=3)(x)["loss"].ndim == 0
