import requests

results = {}

Cookie = 'EMAP_LANG=zh; THEME=cherry; _WEU=Ps89M4vQPtY3t6aZhMZPY03y5kr7iGNoXNgu3AXn5aLXVphMd_ZGT0OG67WzxKTRitfGakvz4eYL7xWJQdNbl0JCIDvjsM4msvWW_DkomKPAL**ogLHAIBON5zdFAQaUeszWgsGyCNi_YBUTCEAUI_3AF2G4in4mHyxEKqHa4nljDoByheQD_S..; MOD_AUTH_CAS=MOD_AUTH_ST-1622995-5tKCS3ZvQQudTTO0HWjq1705905863786-eCjG-cas; asessionid=2abdb013-5304-4c30-a82e-099fec69b379; amp.locale=undefined; JSESSIONID=Ya4v6cbVhSaQYpaEIdb7QYgWbHFtv1GapKJBXqo3LtOodQPtgg2G!-1206621675; route=af1cbe85b0e47a23f95591fc5926cba3'
Referer = 'https://ehall.szu.edu.cn/new/index.html'

cookie_list = Cookie.split(';')

cookies = {}

for item in cookie_list:
    item = item.strip()
    items = item.split('=')
    cookies[items[0]] = items[1]

headers = {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Origin': 'https://ehall.szu.edu.cn',
    'Referer': Referer,
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}


for i in range(950, 700, -1):
    cj = str(int(i / 10)) if i % 10 == 0 else str(i / 10)

    data = {
        'querySetting': '[{"name":"CJ","caption":"成绩","linkOpt":"AND","builderList":"cbl_String","builder":"equal",'
                        '"value":%s},{"name":"_gotoFirstPage","value":true,"linkOpt":"AND","builder":"equal"}]' % cj,
        'pageSize': '10',
        'pageNumber': '1',
    }

    response = requests.post(
        'https://ehall.szu.edu.cn/gsapp/sys/szdxwdcjapp/modules/wdcj/xscjcx.do',
        cookies=cookies,
        headers=headers,
        data=data,
    )
    print(i)
    print(response.json())
    for j in response.json()['datas']['xscjcx']['rows']:
        results[j['KCMC']] = cj

for k, v in results.items():
    print(k, ':', v)

