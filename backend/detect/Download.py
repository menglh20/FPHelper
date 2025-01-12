import os
import requests
import logging
import json

prefixs = [
    'pic_at_rest',
    'pic_forehead_wrinkle',
    'pic_eye_closure',
    'pic_smile',
    'pic_snarl',
    'pic_lip_pucker'
]

class DownloadImage():
    def get(self, fileID, save_path):
        access_token = self.get_access_token()
        file_list = []
        for name, id in fileID.items():
            file_list.append({
                "fileid": id,
                "max_age": 7200
            })
        download_urls = self.get_temp_file_url(file_list, access_token)
        for name, id in fileID.items():
            file_url = download_urls[id]
            self.download_file(file_url, save_path, name + ".jpg")

    def get_access_token(self):
        url = "https://api.weixin.qq.com/cgi-bin/token"
        params = {
            "grant_type": "client_credential",
            "appid": "wx0c07216ab4153ff8",  # 替换为您的小程序 AppID
            "secret": "78474ffdd10832a0d66abab3a6e8c014"  # 替换为您的小程序 AppSecret
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            result = response.json()
            return result['access_token']
        else:
            raise Exception(f"获取 access_token 失败: {response.status_code}")

    def get_temp_file_url(self, file_list, access_token):
        url = f"https://api.weixin.qq.com/tcb/batchdownloadfile?access_token={access_token}"
        
        data = {
            "env": "prod-4ggnzg0z43d1ab28",
            "file_list": file_list
        }

        print(data)
        
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=data, headers=headers)
        print("This is response" + response.text)
        if response.status_code == 200:
            result = response.json()
            download_urls = {}
            if result['errcode'] == 0:
                for res in result['file_list']:
                    download_urls[res["fileid"]] = res["download_url"]
                return download_urls
            else:
                print(result)
                return download_urls
        else:
            raise Exception(f"请求失败: {response.status_code}")

    def download_file(self, file_url, save_path, filename):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 请求图片并保存到本地
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(save_path, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return file_path
        else:
            raise Exception(f"文件下载失败: {response.status_code}")
