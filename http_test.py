# -*- coding:utf8 -*-
import requests
import json
import time

url = 'http://127.0.0.1:8000/get_multitags/'
data = {"uuid": "4596707759",
        "content": "【工信部副部长：全力推进电子信息企业复工复产】财联社3月12日讯，近日，工信部副部长王志军带队，赴北京经济技术开发区，调研骨干电子信息企业疫情防控和复工复产情况。王志军一行走访了冠捷显示科技(中国)有限公司、中芯北方集成电路制造(北京)有限公司、北方华创科技集团股份公司三家企业，实地查看疫情防控工作开展情况，了解复工复产面临的问题。王志军强调，要在做好疫情防控措施的基础上，全力推进企业复工复产，发挥骨干企业作用，带动产业链上下游协同发展。"
        }

t_before = time.time()
header = {'Content-Type':'application/json'}
r =requests.post(url,headers=header,data=json.dumps(data))
t_after = time.time()
print(r.content)
print("耗时:  {}s".format(t_after - t_before))