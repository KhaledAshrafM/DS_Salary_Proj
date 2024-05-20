import pandas as pd 
df = pd.read_csv('data/glassdoor_jobs.csv')

# 1-salary parsing

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer'] = df['Salary Estimate'].apply(lambda x: 1 if "employer provided salary" in x.lower() else 0)
df = df[df['Salary Estimate'] != '-1']
salary = df['Salary Estimate'].apply(lambda x: x.split("(")[0])
minus_kd = salary.apply(lambda x: x.replace("K", "").replace("$", ""))
minus_hr = minus_kd.apply(lambda x: x.lower().replace("per hour", "").replace("employer provided salary:", ""))

df["min_salary"], df["max_salary"] = minus_hr.apply(lambda x: int(x.split("-")[0])), minus_hr.apply(lambda x: int(x.split("-")[1]))
df["avg_salary"] = (df["min_salary"] + df["max_salary"]) / 2
df["avg_salary"]

# 2- Company name text only
df["company_name"] = df.apply(lambda x: x["Company Name"].replace("\n", "") if x["Rating"] < 0 else x["Company Name"][:-3].replace("\n", ""), axis=1)

# 3- State field
df["job_state"] = df["Location"].apply(lambda x: x.split(',')[1])
df["job_state"].value_counts()

df["same_state"] = df.apply(lambda x: 1 if x["Location"] == x["Headquarters"] else 0, axis=1)
# 4- Age of company
df["age_of_company"] = df["Founded"].apply(lambda x: x if x < 0 else 2024-x)

# 5- Parsing of job description (python, etc.)
## Check for Python
df["Python"] = df["Job Description"].apply(lambda x: 1 if "python" in x.lower() else 0)
## Check for Machine learning
df["ML"] = df["Job Description"].apply(lambda x: 1 if "ml" in x.lower() or "machine learning" in x.lower() else 0)
## Check for Deep Learning
df["DL"] = df["Job Description"].apply(lambda x: 1 if "dl" in x.lower() or "deep learning" in x.lower() else 0)
## Check for Spark
df["Spark"] = df["Job Description"].apply(lambda x: 1 if "spark" in x.lower() else 0)
## Check for AWS
df["AWS"] = df["Job Description"].apply(lambda x: 1 if "aws" in x.lower() else 0)
## Check for Excel
df["Excel"] = df["Job Description"].apply(lambda x: 1 if "excel" in x.lower() else 0)
## Check for masters
df["MS"] = df["Job Description"].apply(lambda x: 1 if "masters" in x.lower() else 0)
# Final output
df_final = df.drop('Unnamed: 0', axis=1)
df_final.to_csv("data/salary_data_cleaned.csv", index=False)
pd.read_csv("data/salary_data_cleaned.csv")
