import pymysql
import pandas as pd
from sqlalchemy import create_engine

#mydb = pymysql.connect(host='localhost', user='root', password='zxc098')
mydb = pymysql.connect(host='localhost', user='pmth', db='mydatabase')

mycursor = mydb.cursor()

def print_result(sql, cur=mycursor):
    cur.execute(sql)
    result = mycursor.fetchall()
    if result:
        for i in result:
            print(i)
        print('=='*10)

def show_query():
    sql = 'show databases;'
    print_result(sql)

    sql = 'use keos;'
    print_result(sql)

    sql = 'show tables;'
    print_result(sql)

def create_db():
    sql = 'create database mydatabase'
    print_result(sql)
    

    sql = 'show databases;'
    print_result(sql)

def create_tables():
    sql = 'create table members(name varchar(255), address varchar(255));'
    print_result(sql)

    sql = 'show tables;'
    print_result(sql)

def show_table(table_name):
    sql = (f'desc {table_name}')
    print_result(sql)

def insert_data(table_name):
    sql = f"insert into {table_name} values('Tom', 'Highway 28')"
    print_result(sql)

    sql = f'select * from {table_name}'
    print_result(sql)

def insert_data2(cur=mycursor):
    sql = 'insert into members (name, address) values (%s, %s);'
    val = ('Tomson', 'Highway 29')
    cur.execute(sql, val)
    print(cur.rowcount, ' record inserted.')
    mydb.commit()

    sql = 'select * from members'
    print_result(sql)

def insert_many_data(db, cur=mycursor):
    sql = f'insert into {db} (name, address) values (%s, %s)'
    val = [
            ('Peter', 'Lowstreet 4'),
            ('Amy', 'Apple st 652'),
            ('Hannah', 'Mountain 21'),
            ('Michael', 'Valley 345'),
            ('Sandy', 'Ocean blvd 2'),
            ('Betty', 'Green Grass 1'),
            ('Richard', 'Sky st 331'),
            ('Susan', 'One way 98'),
            ('Vicky', 'Yellow Garden 2'),
            ('Ben', 'Park Lane 38'),
            ('William', 'Central st 954'),
            ('Chuck', 'Main Road 989'),
            ('Viola', 'Sideway 1633')
            ]
    cur.executemany(sql, val)
    mydb.commit()

    print(cur.rowcount, ' record insert.')
    print('id : ', cur.lastrowid)

    sql = f'select * from {db}'
    print_result(sql)

def select_(cur=mycursor):
    sql = 'select * from members where address like %s'
    val = ('%way%')
    cur.execute(sql, val)
    result = cur.fetchall()
    for x in result:
        print(x)
    return result

def delete_commit(cur=mycursor):
    sql = "delete from members where address = 'Highway 28';"
    cur.execute(sql)
    mydb.commit()
    print(cur.rowcount)
    print_result('select * from members;')

def order_by(cur=mycursor):
    sql = 'select * from members order by name desc;'
    print_result(sql)

def update_data(cur=mycursor):
    sql = 'select * from members where address = %s;'
    val = ('Valley 345')
    cur.execute(sql, val)
    result = cur.fetchall()
    for x in result:
        print(x)

    sql = 'update members set address = %s where address = %s;'
    val = ('Canyon 123', 'Valley 345')
    cur.execute(sql, val)
    mydb.commit()

    sql = 'select * from members;'
    print_result(sql)

def select_limit():
    sql = 'select * from members limit 5 offset 2'
    print_result(sql)

def close_conx():
    mydb.close()

def join_():
    sql = 'select users.name, products.name from users \
    	join products on users.fav=products.id;'
    print_result(sql)