'''

* version : 1.0
* 파일명 : connector.py
* 작성일자 : 2018.05.29 
* 작성자 : Frank
* 설명 : 모니터링 데이터를 가져오고 병합하는 역할의 클래스
* 수정자 :
* 수정 내역 :

'''

import pymysql
import fetcher
import pandas as pd

class Monitor:
    # 추천 영역별 fetcher object 정의
    def __init__(self):
        self.dataVolumeFetcher = fetcher.DataVolumeFetcher()
        self.recmndtnCntFetcher = fetcher.RecmndtnCntFetcher()
        self.serverStatusFetcher = fetcher.ServerStatusFetcher()
        self.serviceDbFetcher = fetcher.ServiceDbFetcher()
        print('Start connecting for monitoring system...')
        
    # 각 영역 데이터 get
    def getMonitoringData(self):
        try:
            self.dataVolumeFetcher.fetchData()
            self.recmndtnCntFetcher.fetchData()
            self.serverStatusFetcher.fetchData()
            self.serviceDbFetcher.fetchData()
        except:
            # fetchData 예외 상황 핸들링
            pass
        else:
            self.mergeMonitoringData()
        
    # 모니터링 데이터를 dataframe 형태로 merge
    def mergeMonitoringData(self):
        pass