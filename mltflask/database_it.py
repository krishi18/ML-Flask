import mysql.connector
mydb = mysql.connector.connect(
host="localhost",
user="root",
password="krishi123",
database="testdb1"
)
mycursor = mydb.cursor()
#mycursor.execute("CREATE DATABASE testdb1")
#mycursor.execute("CREATE TABLE signupdetails (name VARCHAR(255),email VARCHAR(255), password  VARCHAR(255))")

def sqlsignup(emal,pwd):
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="krishi123",
      #database="testdb1"
    )

    mycursor = mydb.cursor()

    
    sql = "INSERT INTO signup (email,password) VALUES (%s,%s)"
    val = (emal,pwd)
    mycursor.execute(sql, val)
    mydb.commit()


def sqllogin(email,password):
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      password="krishi123",
      database="testdb1"
    )

    mycursor = mydb.cursor()

    query = f"SELECT password FROM signup WHERE email='{email}'"
    mycursor.execute(query)
    pas=mycursor.fetchone()
    if not pas:
        return "nouser"
    elif pas[0]==password:
        return "found"
    else:
         return "wrongpass"
     
        
     
