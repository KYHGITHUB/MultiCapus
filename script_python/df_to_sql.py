import pymysql
import pandas as pd
from sqlalchemy import create_engine
import module_mysql as mm

'''
data=[{ 'id': 154, 'name': 'Chocolate Heaven' },
{ 'id': 155, 'name': 'Tasty Lemons' },
{ 'id': 156, 'name': 'Vanilla Dreams' }]
'''
data = [{ 'id': 1, 'name': 'John', 'fav': 154},
{ 'id': 2, 'name': 'Peter', 'fav': 154},
{ 'id': 3, 'name': 'Amy', 'fav': 155},
{ 'id': 4, 'name': 'Hannah', 'fav':0},
{ 'id': 5, 'name': 'Michael', 'fav':0}]

df = pd.DataFrame(data)

print(df)

engine = create_engine('mysql+pymysql://pmth'+'@localhost:3306/mydatabase?charset=utf8')

#df.to_sql(name='products', con=engine, if_exists='append', index=False)
df.to_sql(name='users', con=engine, if_exists='append', index=False)