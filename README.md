# Anomaly Detection 프로젝트

DACON에서 주관하는 Anomaly Detection 이상치 탐지 프로젝트에 참가하면서 작성했던 파일들입니다. <br>
대회 링크는 https://dacon.io/competitions/official/235894/overview/description 입니다.

"01제안서_진행파일" 폴더에는 제안서 작성까지 제작된 파일들을 모아두었습니다. <br>
"02중간보고서_진행파일" 폴더에는 중간보고서 작성까지 제작된 파일들을 모아두었습니다. <br>
"03최종보고서_진행파일" 폴더에는 최종보고서 작성까지 제작된 파일들을 모아두었습니다. <br>
"imgs" 폴더에는 EDA 또는 기타 세미나를 진행할 때 제작된 이미지 파일들을 모아두었습니다. <br>
"others" 폴더에는 대회 종료 후 다른 참가팀이 사용한 기법에 대한 세미나를 진행했을 때 코드를 직접 구현해 보려고 제작된 파일들을 모아두었습니다. <br>
"weight_file" 폴더에는 모델의 weight 파일들을 모아두었습니다. 용량 상의 문제로 최종 제출 때 사용한 모델의 weight 파일만 남기고 전부 삭제한 상황입니다. <br>
"\_\_pycache__" 폴더와 "runs" 폴더는 대회 중 TensorBoard 사용을 시도한 적이 있었는데 그 때 생성된 폴더 같습니다. <br>

# MVTec AD Dataset

대회에서 사용된 MVTec AD 데이터셋을 사진으로 정리했습니다.

## ①제품의 Class 별 제품의 Label 자료
![image](https://user-images.githubusercontent.com/93433004/194214175-f6c5ab0d-e987-4841-a917-7c370460db89.png)
![image](https://user-images.githubusercontent.com/93433004/194214278-bc84e267-83bf-4e3d-a161-4482364373c4.png)
![image](https://user-images.githubusercontent.com/93433004/194214290-1339845a-a6e9-45a2-ae56-ad4498ce46a0.png)
![image](https://user-images.githubusercontent.com/93433004/194214334-b793a39d-ace3-48f9-82f1-ce817f6cb85e.png)
![image](https://user-images.githubusercontent.com/93433004/194214350-9d7ae5b7-69e0-4359-94ff-26eb722e74a2.png)
![image](https://user-images.githubusercontent.com/93433004/194214371-39b56aa6-5453-46ef-b91b-e69ca650b705.png)
![image](https://user-images.githubusercontent.com/93433004/194214384-fb86a57e-bfa2-498f-b0ab-47931e846b2c.png)
![image](https://user-images.githubusercontent.com/93433004/194214392-66730b40-8b2e-467b-8473-90abce92e17d.png)
![image](https://user-images.githubusercontent.com/93433004/194214401-5dc20bc1-671a-4be5-a59d-c718164ad7f2.png)
![image](https://user-images.githubusercontent.com/93433004/194214434-7a5b6c55-fb33-469b-9513-9e6e91cc4b35.png)
![image](https://user-images.githubusercontent.com/93433004/194214442-d4228c09-5d89-46d5-8afd-21659cc0b418.png)
![image](https://user-images.githubusercontent.com/93433004/194214452-1bf6687d-93ca-4b9b-aee7-87b4fe846df2.png)
![image](https://user-images.githubusercontent.com/93433004/194214462-2af88bea-d166-4c15-b0de-7e1ae91c5789.png)
![image](https://user-images.githubusercontent.com/93433004/194214522-1f93bde1-fb29-43eb-aafb-24a41a3cd51b.png)
![image](https://user-images.githubusercontent.com/93433004/194214534-2991c518-7d63-4907-a50c-95bae5b2160a.png)

## ②제품의 Class 별 제품의 State 비율
![image](https://user-images.githubusercontent.com/93433004/194214705-79512d52-b9b9-4dbe-9b8d-234861a803c2.png)
![image](https://user-images.githubusercontent.com/93433004/194214719-b3911d1c-1797-43a2-bbe9-3a65a2e068f2.png)
![image](https://user-images.githubusercontent.com/93433004/194214727-a118f35d-200b-413d-8215-db920e3fa5b6.png)
![image](https://user-images.githubusercontent.com/93433004/194214734-93577c94-3b70-4051-bf9b-f3993e804cdd.png)
![image](https://user-images.githubusercontent.com/93433004/194214739-1abf5dba-f6b5-499d-9ce0-3f7a2fb055dc.png)
![image](https://user-images.githubusercontent.com/93433004/194214748-81d94e32-5a85-4bd8-8573-42bb39d921de.png)
![image](https://user-images.githubusercontent.com/93433004/194214756-23396e0d-9485-4132-93a4-7ed59d236a37.png)
![image](https://user-images.githubusercontent.com/93433004/194214764-452a9243-7d67-479f-8b22-a1c00bebb3dd.png)
![image](https://user-images.githubusercontent.com/93433004/194214778-7b674d1d-ff20-496f-b2f2-9f64393c58f9.png)
![image](https://user-images.githubusercontent.com/93433004/194214787-75325fcb-d326-49aa-9453-e24d620b07e4.png)
![image](https://user-images.githubusercontent.com/93433004/194214793-1fad94c7-8101-4b78-bbde-f89dedd07ddd.png)
![image](https://user-images.githubusercontent.com/93433004/194214798-aca2228a-1a50-4f5c-9e5f-1f8eb2445883.png)
![image](https://user-images.githubusercontent.com/93433004/194214817-9cd324a6-a4c1-45b8-b85f-979b18cfb24c.png)
![image](https://user-images.githubusercontent.com/93433004/194214824-6ea38b45-6456-4df4-8991-65e20d47e570.png)
![image](https://user-images.githubusercontent.com/93433004/194214837-210d5f73-f7bb-403a-95cb-f783f3a9e5da.png)
