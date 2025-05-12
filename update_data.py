# update_data.py
import pickle
from app import init_data   # 기존 app.py 안의 init_data()만 가져옵니다

if __name__=="__main__":
    # 1) 기존 init_data() 를 호출해서 새 데이터 받아오고
    data_tuple = init_data()
    # 2) 파일로 저장
    with open("latest_data.pkl", "wb") as f:
        pickle.dump(data_tuple, f)
    print("✅ 데이터 업데이트 완료")
