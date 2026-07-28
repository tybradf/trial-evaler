"""
Two embedders, used identically downstream but built very differently:

  general  -- sentence-transformers/all-mpnet-base-v2
              Strong general-purpose SBERT model, symmetric (same encoder
              for queries and documents).

  biomed   -- ncbi/MedCPT-{Query,Article}-Encoder
              NIH/NCBI's purpose-built biomedical retrieval model, trained
              on 255M real PubMed query-article search-log pairs. ASYMMETRIC:
              a separate query encoder and article encoder, and the article
              encoder specifically expects (title, body) pairs, matching
              its title+abstract training format -- not a single flat
              string. We mirror that here (title, [conditions+summary+
              eligibility]) rather than passing one concatenated string, so
              MedCPT gets a fair shot rather than a crippled one.

NOTE (verify on first run -- couldn't execute against live HF weights from
the sandbox this was written in): CLS-token pooling (last_hidden_state[:,0,:])
and the (title, body) pair input format for the article encoder are both per
the model card at huggingface.co/ncbi/MedCPT-Article-Encoder as of when this
was written. Spot check a couple of embeddings' cosine similarities against
obviously-related vs unrelated trials to confirm this is behaving sanely
before trusting the retrieval numbers.
"""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


class GeneralEmbedder:
    name = "general_mpnet"

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                  normalize_embeddings=True, convert_to_numpy=True)

    def encode_docs(self, titles: list[str], bodies: list[str], batch_size: int = 64) -> np.ndarray:
        # Symmetric model -- just concatenate title + body into one string.
        texts = [f"{t}. {b}" for t, b in zip(titles, bodies)]
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                  normalize_embeddings=True, convert_to_numpy=True)


class MedCPTEmbedder:
    name = "biomed_medcpt"
    MAX_LEN = 512

    def __init__(self):
        self.q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(DEVICE).eval()
        self.a_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self.a_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(DEVICE).eval()

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # NOTE: MedCPT's own examples use max_length=64, tuned for short
        # PubMed search-log queries. TREC's "queries" here are full
        # multi-sentence clinical vignettes -- 64 tokens silently truncated
        # most of each topic's content (confirmed: nDCG@10 was ~0.10-0.21
        # vs. ~0.46-0.49 for the general model, a gap this method explains).
        # 512 matches the article encoder's capacity and is within the
        # underlying BERT architecture's native limit.
        out = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                enc = self.q_tok(batch, truncation=True, padding=True,
                                  return_tensors="pt", max_length=512).to(DEVICE)
                emb = self.q_model(**enc).last_hidden_state[:, 0, :]
            out.append(emb.cpu().numpy())
        return self._normalize(np.vstack(out))

    def encode_docs(self, titles: list[str], bodies: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(titles), batch_size)):
            t_batch = titles[i:i + batch_size]
            b_batch = bodies[i:i + batch_size]
            with torch.no_grad():
                # (title, body) sentence-pair tokenization, per the article
                # encoder's title+abstract training format.
                enc = self.a_tok(t_batch, b_batch, truncation=True, padding=True,
                                  return_tensors="pt", max_length=self.MAX_LEN).to(DEVICE)
                emb = self.a_model(**enc).last_hidden_state[:, 0, :]
            out.append(emb.cpu().numpy())
        return self._normalize(np.vstack(out))


EMBEDDERS = {"general": GeneralEmbedder, "biomed": MedCPTEmbedder}
