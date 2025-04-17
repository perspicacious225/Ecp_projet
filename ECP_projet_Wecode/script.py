import csv, sqlite3

con = sqlite3.connect('ecp_db.db') 
cur = con.cursor()
# cur.execute("DROP TABLE Product")
cur.execute("CREATE TABLE Product (id, prod_name, price);")
with open('product_to_db.csv','r') as fin:
    dr = csv.DictReader(fin)
    to_db = [(i['itemid'],i['0'], i['1']) for i in dr]

cur.executemany("INSERT INTO Product (id, prod_name, price) VALUES (?,?,?);", to_db)
con.commit()
con.close() 

# import pandas as pd
# df=pd.read_csv('results.csv')
# df2=pd.read_csv('features_svd_new.csv')

# itemid= [i for i in df2['itemid']]
# # print(len(itemid))
# df['itemid'] = itemid
# df=df.drop('2', axis=1)
# print(df)
# df.to_csv('product_to_db.csv', index=False)