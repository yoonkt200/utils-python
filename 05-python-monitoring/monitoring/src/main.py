'''

* version : 1.0
* 파일명 : main.py
* 작성일자 : 2018.05.29 
* 작성자 : Frank
* 설명 : main file
* 수정자 :
* 수정 내역 :

'''

#from pyhive import hive
import sys ; sys.path.append('../')
import config

#def test():
 #   conn = hive.Connection(host='10.52.111.105', port=10000, username='gsshop')
  #  df = pd.read_sql('SELECT * FROM GSSHOP.DEALTEMINFO LIMIT 10', conn)
   # print(df.columns)

#def test2():
 #   connection = pymysql.connect(host=config.DATABASE_JENKINS_CONFIG['host'], 
  #                                          user=config.DATABASE_JENKINS_CONFIG['user'], 
   #                                         passwd=config.DATABASE_JENKINS_CONFIG['password'], 
    #                                        database=config.DATABASE_JENKINS_CONFIG['dbname'])
#    cursor = connection.cursor()
 #   query1 = 'SELECT * FROM TB_COMM'
  #  cursor.execute(query1)
   # rows = cursor.fetchall()
    #for row in rows:
     #   print(row)
#    connection.close()

if __name__ == '__main__':
    test = config.RECMNDTN_MNTRNG_CONFIG['volm']['asscr']
    print(test)