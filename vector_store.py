import os
import pandas as pd
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SHLVectorStore:
    def __init__(self, csv_path="data/shl_catalog_cleaned.csv"):
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.documents = self.df['description'].fillna("").tolist()
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 1. Initialize BM25
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(" BM25 Keyword Index initialized.")
        
        self.index = None
        self.embeddings = None

    def build_index(self):
        print(f"  Generating OpenAI embeddings for {len(self.df)} items")
        response = self.client.embeddings.create(
            input=self.documents,
            model="text-embedding-3-small"
        )
        self.embeddings = np.array([e.embedding for e in response.data]).astype('float32')
        
        faiss.normalize_L2(self.embeddings)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) 
        self.index.add(self.embeddings)
        print(f" FAISS Vector Index Ready (Dim: {dimension}).")

    def hybrid_search(self, query, top_k=20):
        print(f"\n--  HYBRID SEARCH START --")
        print(f"Query: '{query[:70]}'")
        
        # A. BM25 Lexical Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][0]
        print(f"BM25 Top Match: '{self.df.iloc[bm25_top_idx]['name']}' (Score: {bm25_scores[bm25_top_idx]:.2f})")
        
        # B. Vector Semantic Search
        q_resp = self.client.embeddings.create(input=[query], model="text-embedding-3-small")
        q_emb = np.array([q_resp.data[0].embedding]).astype('float32')
        faiss.normalize_L2(q_emb)
        
        D, I = self.index.search(q_emb, len(self.df))
        print(f"Vector Top Match: '{self.df.iloc[I[0][0]]['name']}' (Sim: {D[0][0]:.4f})")
        
        # C. RRF Fusion
        combined_scores = {}
        for rank, idx in enumerate(np.argsort(bm25_scores)[::-1]):
            combined_scores[idx] = combined_scores.get(idx, 0) + 1.0 / (rank + 60)
        for rank, idx in enumerate(I[0]):
            combined_scores[idx] = combined_scores.get(idx, 0) + 1.0 / (rank + 60)
            
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        print(f"Fusion complete. Top Candidate: '{self.df.iloc[sorted_indices[0]]['name']}'")
        return sorted_indices[:top_k], q_emb[0]

    def mmr_rerank(self, query_emb, candidate_indices, top_n=10, lambda_param=0.5):
        print(f"\n{'='*20} BALANCED MMR RERANKING {'='*20}", flush=True)
        if not candidate_indices: return []
        
        selected_indices = [candidate_indices[0]]
        remaining_indices = candidate_indices[1:]
        
        print(f"INITIALLY SELECTED: {self.df.iloc[selected_indices[0]]['name']} [{self.df.iloc[selected_indices[0]]['test_type']}]", flush=True)
        
        while len(selected_indices) < top_n and remaining_indices:
            best_score = -np.inf
            best_idx = -1
            
            # Count current balance
            selected_types = [self.df.iloc[i]['test_type'] for i in selected_indices]
            knowledge_count = selected_types.count('Knowledge & Skills')
            behavior_count = selected_types.count('Personality & Behaviour')

            for idx in remaining_indices:
                item = self.df.iloc[idx]
                relevance = np.dot(query_emb, self.embeddings[idx])
                similarity_to_selected = max([np.dot(self.embeddings[idx], self.embeddings[s_idx]) for s_idx in selected_indices])
                
                # MMR Base Score
                score = lambda_param * relevance - (1 - lambda_param) * similarity_to_selected
                
                # --- ADVANCED CATEGORY BALANCING ---
                # If we already have 2+ Knowledge tests and 0 Behavior tests, 
                # give a small "bonus" to Behavior tests to help them surface.
                if item['test_type'] == 'Personality & Behaviour' and behavior_count == 0:
                    score += 0.1  # The "Diversity Boost"
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx == -1: break
            
            new_item = self.df.iloc[best_idx]
            print(f"↳ ADDING: {new_item['name'][:30]}... [{new_item['test_type']}] (Score: {best_score:.4f})", flush=True)
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
        print(f" BALANCED MMR COMPLETE: {len(selected_indices)} candidates ready.", flush=True)
        return selected_indices

    def get_diverse_candidates(self, query, top_n=10):
        candidate_indices, q_emb = self.hybrid_search(query)
        diverse_indices = self.mmr_rerank(q_emb, candidate_indices, top_n=top_n)
        return self.df.iloc[diverse_indices]