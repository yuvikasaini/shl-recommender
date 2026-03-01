import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SHLRecommender:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_final_recommendations(self, job_description: str):
        # 1. Get 10 diverse candidates from Vector Store
        candidates_df = self.vector_store.get_diverse_candidates(job_description, top_n=10)
        
        # 2. Prepare the candidate pool with all keys
        candidate_pool = []
        for _, row in candidates_df.iterrows():
            candidate_pool.append({
                "name": row['name'],
                "description": row['description'],
                "test_type": row['test_type'], 
                "duration": int(row['duration']),
                "url": row['url']
            })

        # 3. The Prompt - Strict Key Enforcement
        prompt = f"""
        You are an expert HR Consultant. Analyze this JD: "{job_description}"
        
        TASK: Select the 5 best assessments from the pool below.
        
        RULES:
        1. Select 5 items total. 
        2. Ensure at least 1 is a REAL Behavioral/Personality test (OPQ, Scenarios, etc.).
        3. Ensure 2-4 are Technical tests matching the JD.
        
        REQUIRED JSON SCHEMA (Appendix 3):
        Return a JSON object with key "recommended_assessments".
        Each assessment MUST contain: "name", "description", "test_type", "duration", "url".
        DO NOT omit 'test_type'.
        
        POOL:
        {json.dumps(candidate_pool)}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a JSON-only response bot. You must include 'test_type' for every item."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0
            )
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"LLM Error: {e}")
            fallback = candidates_df.head(5).to_dict(orient='records')
            return {"recommended_assessments": fallback}