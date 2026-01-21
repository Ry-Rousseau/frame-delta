import psycopg2
from psycopg2.extras import execute_values
import ast
import os
from dotenv import load_dotenv

load_dotenv()

# 1. SETUP CLEANING LOGIC -----------------------------------------------------
official_labels = [
    "Economic", "Capacity and resources", "Morality", "Fairness and equality",
    "Legality, constitutionality and jurisprudence", "Policy prescription and evaluation",
    "Crime and punishment", "Security and defense", "Health and safety",
    "Quality of life", "Cultural identity", "Public opinion", "Political",
    "External regulation and reputation", "Other"
]
valid_map = {label.lower(): label for label in official_labels}
custom_fixes = {
    "legality, constitutionality and jurispudence": "Legality, constitutionality and jurisprudence",
    "public opinion": "Public opinion",
    "quality of life": "Quality of life",
    "policy prescription and evaluation": "Policy prescription and evaluation",
    "cultural identity": "Cultural identity",
    "fairness and equality": "Fairness and equality",
    "safety and health": "Health and safety",
    "race and ethnicity": "Cultural identity",
}
valid_map.update(custom_fixes)

def clean_frame_row(frame_str):
    try:
        if not frame_str: return []
        raw_list = ast.literal_eval(frame_str)
        if not isinstance(raw_list, list): return []
        
        clean_list = []
        for label in raw_list:
            lbl_lower = str(label).lower().strip()
            if lbl_lower in valid_map:
                clean_list.append(valid_map[lbl_lower])
            else:
                clean_list.append("Other")
        
        # Deduplicate and remove 'Other' if specific tags exist (optional preference)
        clean_list = list(set(clean_list))
        return clean_list if clean_list else []
    except:
        return []

# 2. EXECUTE MIGRATION --------------------------------------------------------
try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cur = conn.cursor()

    print("1. Creating temporary ARRAY column...")
    cur.execute("ALTER TABLE mm_framing_full ADD COLUMN IF NOT EXISTS frames_array text[];")
    conn.commit()

    print("2. Fetching data...")
    # We only need the ID and the old text column
    cur.execute("SELECT url, text_generic_frame FROM mm_framing_full")
    rows = cur.fetchall()
    
    print(f"3. Processing {len(rows)} rows in memory...")
    # Prepare list of tuples for bulk update: (cleaned_list_as_array, url_id)
    update_data = []
    for url, raw_str in rows:
        cleaned_list = clean_frame_row(raw_str)
        update_data.append((cleaned_list, url))

    print("4. Performing BULK UPDATE (Fast)...")
    # This is the magic. page_size=10000 sends data in massive chunks.
    query = """
        UPDATE mm_framing_full AS t 
        SET frames_array = v.new_frames 
        FROM (VALUES %s) AS v(new_frames, url_key) 
        WHERE t.url = v.url_key
    """
    execute_values(cur, query, update_data, page_size=10000)
    conn.commit()
    print("   Update complete.")

    print("5. Swapping columns...")
    # Drop old string column, rename new array column to old name
    cur.execute("""
        ALTER TABLE mm_framing_full DROP COLUMN text_generic_frame;
        ALTER TABLE mm_framing_full RENAME COLUMN frames_array TO text_generic_frame;
    """)
    conn.commit()
    
    print("SUCCESS! 'text_generic_frame' is now type text[] (ARRAY) and fully cleaned.")

except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    cur.close()
    conn.close()