import pandas as pd

# 1. Load the original catalog
df = pd.read_csv('/home/yuvika/Documents/SHL Smart Recommender/data/shl_catalog_final.csv')

# 2. Define the logic for corrections
# We use keywords to find items that are clearly in the wrong category
tech_patterns = [
        'developer', 'development', 'java', 'python', 'sql', 'aws', 'amazon','informatica', 'salesforce', 'frameworks', 'platform', 'zabbix','services', 'ssis', 'ssrs', 'ssas', 'writing', 'spanish', 'english',
        'hiring concepts', 'virtual assessment', 'banking', 'financial'
    ]

    # These terms are EXCLUSIVELY behavioral/personality
personality_patterns = ['opq', 'personality', 'motivation', 'leadership', 'scenarios','styles', 'team impact', 'behavioral', 'behavioural', 'trait', 
        'dependability', 'safety', 'empathy']

def fix_type(row):
    name = str(row['name']).lower()
    # If a tech keyword is found, it must be Knowledge & Skills
    if any(kw in name for kw in tech_patterns):
        return 'Knowledge & Skills'
    # If a behavioral keyword is found, it must be Personality & Behaviour
    if any(kw in name for kw in personality_patterns):
        return 'Personality & Behaviour'
    return row['test_type']

# 3. Apply the fix
df['test_type'] = df.apply(fix_type, axis=1)

# 4. Save with ALL original columns
df.to_csv('shl_catalog_cleaned.csv', index=False)
print("shl_catalog_cleaned.csv created with all columns and corrected types.")