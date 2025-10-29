import torch
from torch.utils.data import Dataset
import ast
import re 
# בקובץ data_processing.py (או למעלה ב-dataset.py)***נכתב ע"י gemini***


def clean_and_format_schema(raw_table_string: str) -> str:
    """
    Extracts column headers from the raw table string using a robust regex 
    that handles the NumPy array format and multi-line strings.
    """
    try:
        # 1. הדפוס מחפש את התוכן שבתוך array([...]) ששייך למפתח 'header'
        pattern = r"'header':\s*array\(\s*\[(.*?)\]"
        
        # 2. החיפוש מתבצע עם הדגל re.DOTALL כדי להתמודד עם ירידות שורה
        match = re.search(pattern, raw_table_string, re.DOTALL)
        
        if match:
            # 3. חילוץ התוכן שנלכד מהדפוס
            headers_string = match.group(1)
            
            # 4. ניקוי המחרוזת והפיכתה לרשימה נקייה של כותרות
            headers = [h.strip().replace("'", "") for h in headers_string.split(',')]
            headers = [h for h in headers if h] # סינון מחרוזות ריקות
            
            if headers:
                # 5. עיצוב סופי עבור המודל: אותיות קטנות וקווים תחתונים
                clean_headers = [h.replace(" ", "_").lower() for h in headers]
                return f"table ( {', '.join(clean_headers)} )"

    except Exception:
        # במקרה של שגיאה בלתי צפויה, הפונקציה תחזיר מחרוזת ריקה
        return ""
        
    return "" # החזרת מחרוזת ריקה אם לא נמצאה התאמה
#===========================================================================================

class SQLDataset (Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.dataset['schema_input'] = self.dataset['table'].apply(clean_and_format_schema)


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        question = self.dataset['question'][index].lower() 
        query = self.dataset['sql'][index].lower()
        schema_string = self.dataset['schema_input'][index]
        T5_prefix = "translate English to SQL: "
        full_request = f"{T5_prefix}{schema_string} | {question}"
        source_incoding = self.tokenizer(full_request, max_length= self.max_length, truncation=True, padding= 'max_length', return_tensors='pt')
        target_incoding = self.tokenizer(query, max_length= self.max_length, truncation=True, padding= 'max_length', return_tensors='pt')
        
        dict_data = {
        'input_ids': source_incoding['input_ids'].squeeze(),
        'labels': target_incoding['input_ids'].squeeze(),
        'attention_mask': source_incoding['attention_mask'].squeeze()}

        return dict_data