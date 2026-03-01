import pandas as pd
import os

def fix_schema_and_enrich():
    input_path = 'data/shl_catalog.csv'
    output_path = 'data/shl_catalog_final.csv'

    # Ensure the data folder exists
    if not os.path.exists('data'):
        print(" Error: 'data' folder not found. Please make sure your scraped CSV is in a folder named 'data'.")
        return

    # Load the catalog
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f" Error: {input_path} not found.")
        return

    def determine_exact_type(row):
        # Combine name and description for better context
        text = f"{row['name']} {row['description']}".lower()
        
        # EXACT strings from the PDF Appendix 3
        # Note: These are lists because the schema shows ["Type"]
        TYPE_SOFT = "Personality & Behaviour"
        TYPE_HARD = "Knowledge & Skills"

        # Keywords for Soft Skill / Behavioral / Personality
        behavioral_hints = [
            'behavior', 'personality', 'leadership', 'situational', 'judgment', 
            'competency', 'fit', 'mindset', 'attitude', 'sales', 'service', 
            'management', 'social', 'emotional', 'professional'
        ]
        
        # Keywords for Hard Skill / Knowledge / Technical
        skill_hints = [
            'knowledge', 'programming', 'software', 'technical', 'math', 
            'coding', 'excel', 'word', 'sql', 'python', 'java', 'typing',
            'mechanical', 'clerical', 'calculation', 'proficiency', 'simulation'
        ]

        # Logic check:
        if any(hint in text for hint in behavioral_hints):
            return TYPE_SOFT
        if any(hint in text for hint in skill_hints):
            return TYPE_HARD
        
        # Default fallback
        return TYPE_HARD

    print(" Rendering Test Types to match PDF Schema...")
    df['test_type'] = df.apply(determine_exact_type, axis=1)
    
    # Ensure duration is an integer (as seen in Appendix 3)
    df['duration'] = df['duration'].fillna(15).astype(int)
    
    # Save the final version in the data folder
    df.to_csv(output_path, index=False)
    print(f" Success! Created: {output_path}")
    print(f"Final Row Count: {len(df)}")

if __name__ == "__main__":
    fix_schema_and_enrich()