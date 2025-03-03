import json
import requests

url1 = ''
url2 = ''


def set_urls(new_url1, new_url2):
    global url1, url2
    url1 = new_url1
    url2 = new_url2


headers = {
    'cookie': 'receive-cookie-deprecation=1; ALIPAYCHAIRBUCJSESSIONID=e19a3701-74d2-4292-984b-032fdf67f7d8; ordv=RU5mj1IbBw..; an=fengli.zy; lg=true; sg=y41; SSO_EMPID_HASH_V2=7f63e390ced141ae5f7ad151d6e5cdf8; SSO_BU_HASH_V2=5d9e78fc6ba1293d01e3c65c1fef1f1a; _CHIPS-ALIPAYCHAIRBUCJSESSIONID=e19a3701-74d2-4292-984b-032fdf67f7d8; bs_n_lang=zh_CN; x_umt=P1gArby1pO9ac0-nh0xrhggKjcjnIooDbKXW0kz-z7DEOOoGoFzsxRZltU1jEUR3FeNQ3t9DnD9yZFj2K1-n5yB-; cna=JK8WHzopnVgBASQBsYD+s2Qs; _CHIPS-yuque_ctoken=I9rt4ZnwJqxHUL1jogfeAS7q; yuque_ctoken=I9rt4ZnwJqxHUL1jogfeAS7q; ck2=231ab88bff216da9489e84be86714aac; dw_bff=455091.%E9%A3%8E%E7%A4%BC.1.0.....0.1727858188396.fengli*zy.2.113bbb2f718d99efe1d9cb793a4327e46152de36c; spark_language=zh_CN; tianshu_corp_user=dingd8e1123006514592_455091; tianshu_csrf_token=4d6f8282-e076-40da-acb3-a823e3779f55; c_csrf=4d6f8282-e076-40da-acb3-a823e3779f55; mustAddPartitionedTag=noNeedToAdd; x_umt=P1gArby1pO9ac0-nh0xrhggKjcjnIooDbKXW0kz-z7DEOOoGoFzsxRZltU1jEUR3FeNQ3t9DnD9yZFj2K1-n5yB-; tianshu_corp_id=alibaba; ALIPAYDWDISSESSIONID=GZ00qW4kR6CGgSPoabkU5s8QwqDQf4capGZ00; SSO_LANG_V2=ZH-CN; aliyun_lang=zh; unredirect_inst=636828fb-4c59-45cb-b21e-5a1f3a06fab8; undirect_procCode=TPROC--GW866081OVP1FR3IAB5GOAEB9GSQ3D64MZH5LE8; tianshu_app_type=APP_OQTYFF52908OYESJDDE1; nebula_user_id=455091; rtk=0zTtGAFINKC+t73oXLh4ptlHz/ODQB4OK0niU9OJNgcvmeP2XL2; sgcookie=E100OOwIEe56Sdxrt/Ao0Jsyk66vT0HuqFI49BsfJsewbNShALRn9EGFFflxLl4dBQ/w/9z+p15CkvqmnMyD5CwpYxJgdwzjX0XnsjTbi/ONwaU=; emplId=455091; c_token=72a115b7225080ed57b265f9a54a98f9; x_mini_wua=Ar1G04gzH+MD5VtDZql6fo2BRVOL7a2+Psrm5Bxf5ZqR3rR0k0zn2zqvRaweL3OVB7l/5GkOr26iINTdoQdYOtGm3QOICcvWvUfI5kZMmZz4u9QOiHeFiO1GnQ82R8dIwxDviuxmjsOLHm5zmC3o8ShsU/+egaDCta9nl0T8VE2O4VvCC4sy0eCCcOd62tuTPlaKZ+qjEo5bzTF9wI3um5xE/fnB3+mJgKJNeIB6qRlVI4taNNp5USuz7b1hkoPtaxaeEXBXTHTUkEhL49oTEkM5i/7PD7hmHQcpGndNDVbBDsG605HSO0dUFgk6vvm/x9VHUxGPOCxdDNi7jTJjChx488RomFKEJ6pj42b1fjBi; x_sign=mb000100102173c9f8257e292f3168f8d21a4996c0716da5db; lvc=sArl3nJXS424bg%3D%3D; _xsrf=2|a8e21340|60230badbb55ee7d5ea3efcd79d47b2d|1732953613; SSO_LANG_V2=ZH-CN; xlly_s=1; SSO_REFRESH_TOKEN=e392e80c9bb244b3be3f40674d9c0d0037c07900; x_mini_wua=Ad3ODQzwhMVT6i3HUAFG8j4GPBo3+f8F2PIyryIl3v6InjQGJsSz6i83x49Qs60TmSB+jS/W9MzihORR5/WtNWIG3VnWQUsK1NVqkiCFA3EBky1y0kFFVVmbnUPzLctl1Gvd1vPUUFJapOKVxe/hdnzOl6+mP+EQcvU3RwcJamXnyX6IWqU7nOGy9dipUr3REtyPoa0rot+uaB7BH5XpsN1OVSeEV+3FSpfd/ZmvUTg/BhkLkgM9PjdfiiE/oMSuEpM3d6/AGyiyFGiKgxfu35MRd+2/3hNl8wMNsBVZUJx4UpPD69gxQsmIJja0WcCQ3B0sFCEkyEP1MLCorrq8f+SHKKjIjTRIR453OW54Tq3F; x_sign=mb00010010deafae5e868355932bc2f196149c7735dc1b8226; isg=BBcaM2O8PCMrM7ltC35-_uZapo1hXOu-dkY1kGlEM-ZNmDfacSx7DtXw_jiGcMM2; tfstk=fXUX7Wx2URUz4Ln5UoCrF1P0lagsdCaf8uO_tWHwuh3YFlwUQnuNmcES582fssjGmz__FRmTuqP4FfF_CSoZk5ys6JyZ3xy45uY_t54ViV5moS0iW9WF8-N0iVAS4hGVU7CtsdSejR7UiS0MGdCU2wJ_rgZGamhBqgIM11PDREY24_mXkB8UXMzKGmMB7YT651ljcvFxF9BLPEijpSHWW9-iB4M-OzBM3A47wrhUea8j57h_dqE5eEw_Mk2i9-QWHjZYOWo7S9LsZuem4jZ563HY9xPKuPJRGA44hPi0J98oHuau5qrPhekiXuzmxrWJl2E3gqrTe1LtCuhO4LLEdSkefQiH5bMFN_tMjt4bKwC3Ho7qDbcA8_1WFhmxZbGu7_tXjmhoMO15NLtG.',
}


def get_response(user_question, history_text, end_tokens='<|im_end|>', n=1, special_model=False, max_tokens=1024,
                 temperature=0.7, tools=None):
    if special_model:
        url = url2
    else:
        url = url1
        end_tokens += ',</tool_call>'

    json_data = {
        'user_question': user_question,
        'model_output': history_text,
        'end_tokens': end_tokens,
        'n': n,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'use_tools': json.dumps(tools) if tools else '{}',
    }

    response = requests.post(url, json=json_data, headers=headers)
    # breakpoint()
    return response.json()['response']


if __name__ == "__main__":
    set_urls('https://nebula-notebook.alibaba-inc.com/notebook-7bc2d2340feb-code/proxy/40000/generate',
             'https://nebula-notebook.alibaba-inc.com/notebook-7bc2d2340feb-code/proxy/40001/generate')

    text = """oyfjdnisdr rtqwainr acxz mynzbhhx -> Think step by step

Use the example above to decode:

oyekaijzdf aaptcg suaokybhai ouow aqht mynznvaatzacdfoulxxz
    """
    # print(text)
    # response = get_response(text,'',n=1,special_model=False,max_tokens=8192,temperature=0)#,tools=tools)
    # print('\n'.join(response))

    print(text)
    response = get_response(text, '', n=1, special_model=True, max_tokens=8192, temperature=0)  # ,tools=tools)
    print('\n'.join(response))
