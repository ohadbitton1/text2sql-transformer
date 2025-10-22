import pandas as pd
import ast

df = pd.read_csv("data/validation.csv")
print(df['sql'].iloc[0])
#safe convert
def safe_convert (sql_string):
    #cleaning first
    sql_string = sql_string.replace('array([','[')
    sql_string = sql_string.replace('], dtype=int32)', ']')
    sql_string = sql_string.replace('], dtype=object)', ']')
    try:
        sql_string = ast.literal_eval(sql_string)
        return sql_string
    except:
        return None
    

 #needs apply()- every cell in seperate
df['sql'] = df['sql'].apply(safe_convert)
df.dropna(subset=['sql'], inplace=True)
 



# ---- הוכחה שזה עבד ----
print("the new: ")
print(df['sql'].iloc[0])
print(type(df['sql'].iloc[0]))
print(df.info())