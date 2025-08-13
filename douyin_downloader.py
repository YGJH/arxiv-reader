import requests
import json
import os
from urllib.parse import urlparse
import time
from DrissionPage import ChromiumPage, ChromiumOptions
import re
import tempfile
import shutil

    


class DouyinVideoDownloader:
    def __init__(self, download_folder="douyin_videos5", cookie_file="cookies.json"):
        self.download_folder = download_folder
        self.cookie_file = cookie_file
        self.session = requests.Session()
        self.setup_session()
        self.load_cookies()
        self.create_download_folder()
    
    def get_video_list_with_browser(self, user_url):
        """使用瀏覽器獲取視頻列表"""
        user_data_dir = None
        try:
            print("啟動瀏覽器...")
            co = ChromiumOptions()

            # 改為 WSL2 上安裝的 Linux chromium
            chrome_path = '/usr/bin/google-chrome'
            co.set_browser_path(chrome_path)

            # 必要參數
            co.set_argument('--no-sandbox')
            co.set_argument('--disable-dev-shm-usage')
            # co.set_argument('--disable-gpu')
            co.set_argument('--disable-web-security')
            co.set_argument('--disable-features=VizDisplayCompositor')
            co.set_argument('--disable-extensions')
            co.set_argument('--disable-plugins')
            co.set_argument('--disable-images')
            # 這兩行新增：
            co.set_argument('--disable-setuid-sandbox')
            co.set_argument('--remote-debugging-address=0.0.0.0')
            co.set_argument('--headless=new')

            # 設置用戶數據目錄避免衝突
            user_data_dir = tempfile.mkdtemp()
            co.set_user_data_path(user_data_dir)

            # 自動選擇一個空閒端口並啟用 remote debugging
            co.auto_port(9222)
            co.set_argument(f'--remote-debugging-port={9222}')

            print("正在啟動Chrome瀏覽器...")
            page = ChromiumPage(addr_or_opts=co)
            
            # 訪問用戶頁面
            print("訪問用戶頁面...")
            page.get(user_url)
            time.sleep(5)  # 等待頁面加載
            
            # 開始監聽所有網絡請求
            print("開始監聽網絡請求...")
            page.listen.start()
            
            # 檢查新的登入介面
            login_panel = page.ele('#douyin-login-new-id')
            if login_panel:
                print("發現登入介面，關閉它...")
                # 點擊對應的關閉按鈕 (svg rect)
                close_btn = page.ele('rect[fill="url(#pattern0_3645_22461)"]')
                if close_btn:
                    close_btn.click()
                    time.sleep(2)
                else:
                    print("未找到關閉按鈕")

            # 滾動頁面以觸發視頻列表的加載
            print("滾動頁面觸發請求...")
            for i in range(5):  # 增加滾動次數
                page.scroll.to_bottom()
                time.sleep(2)
                print(f"滾動 {i+1}/5...")
            
            # 等待並獲取包含aweme的請求
            print("等待API響應...")
            timeout = 30
            start_time = time.time()
            found_responses = []
            
            while time.time() - start_time < timeout:
                # 獲取所有響應
                responses = page.listen.steps()
                
                for response in responses:
                    if response.url and 'aweme' in response.url:
                        found_responses.append(response.url)
                        print(f"找到響應: {response.url[:100]}...")
                        
                        if 'post' in response.url or True:
                            print(f"找到aweme API請求: {response.url[:100]}...")
                            try:
                                data = response.response.body
                                if data:
                                    # 嘗試解析為JSON
                                    if isinstance(data, str):
                                        text = page.listen.response_text(response, timeout=5)
                                        json_data = json.loads(text)
                                    else:
                                        json_data = data
                                    
                                    if 'aweme_list' in json_data:
                                        print(f"成功獲取 {len(json_data['aweme_list'])} 個視頻")
                                        return json_data['aweme_list']
                            except Exception as e:
                                print(f"解析響應失敗: {e}")
                                continue
                
                time.sleep(1)
            
            print(f"未找到包含aweme_list的響應")
            print(f"找到的響應: {found_responses}")
            return None
                
        except Exception as e:
            print(f"瀏覽器獲取數據失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            try:
                page.quit()
            except:
                pass
            # 清理臨時目錄
            if user_data_dir:
                try:
                    shutil.rmtree(user_data_dir, ignore_errors=True)
                except:
                    pass

    def create_download_folder(self):
        """創建下載資料夾"""
        if not os.path.exists(self.download_folder):
            os.makedirs(self.download_folder)
            print(f"已創建下載資料夾: {self.download_folder}")
    

    def setup_session(self):
        """設置請求頭"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Linux; U; Android 9; zh-cn; Redmi Note 5 Build/PKQ1.180904.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/71.0.3578.141 Mobile Safari/537.36 XiaoMi/MiuiBrowser/11.10.8',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.douyin.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        })
    
    def load_cookies(self):
        """從 cookies.json 文件載入 cookies"""
        if not os.path.exists(self.cookie_file):
            print(f"未找到 cookies 文件: {self.cookie_file}")
            return
        
        try:
            # 讀取 JSON 格式的 cookies 文件
            with open(self.cookie_file, 'r', encoding='utf-8') as f:
                cookies_list = json.load(f)
            
            cookies_count = 0
            for cookie in cookies_list:
                try:
                    name = cookie.get('name')
                    value = cookie.get('value')
                    domain = cookie.get('domain', '.douyin.com')
                    
                    if name and value:
                        self.session.cookies.set(name, value, domain=domain)
                        cookies_count += 1
                        
                except Exception as cookie_error:
                    print(f"設置單個 cookie 失敗: {cookie.get('name', 'unknown')}, 錯誤: {cookie_error}")
                    continue
            
            print(f"成功載入 {cookies_count} 個 cookies")
            
        except Exception as e:
            print(f"載入 cookies 失敗: {e}")
            import traceback
            traceback.print_exc()
    
    def load_cookies_to_browser(self, page):
        """將 cookies 載入到瀏覽器"""
        if not os.path.exists(self.cookies_file):
            return
        
        try:
            # 先訪問抖音主頁以設置域名
            page.get('https://www.douyin.com')
            time.sleep(2)
            
            # 讀取 JSON 格式的 cookies 文件
            with open(self.cookies_file, 'r', encoding='utf-8') as f:
                cookies_list = json.load(f)
            
            cookies_count = 0
            for cookie in cookies_list:
                try:
                    # 構建 cookie 字典，只使用必要的字段
                    cookie_dict = {
                        'name': cookie.get('name'),
                        'value': cookie.get('value'),
                        'domain': cookie.get('domain', '.douyin.com'),
                        'path': cookie.get('path', '/'),
                        'secure': cookie.get('secure', False),
                        'httpOnly': cookie.get('httpOnly', False)
                    }
                    
                    # 過濾掉無效的 cookies
                    if cookie_dict['name'] and cookie_dict['value']:
                        page.set.cookies(cookie_dict)
                        cookies_count += 1
                        
                except Exception as cookie_error:
                    print(f"設置單個 cookie 失敗: {cookie.get('name', 'unknown')}, 錯誤: {cookie_error}")
                    continue
            
            print(f"成功載入 {cookies_count} 個 cookies 到瀏覽器")
            
        except Exception as e:
            print(f"載入 cookies 到瀏覽器失敗: {e}")
            import traceback
            traceback.print_exc()

    def fetch_video_list(self, api_url):
        """使用API獲取視頻列表"""
        try:
            print("正在獲取視頻列表...")
            
            # 添加必要的cookies和headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Linux; U; Android 9; zh-cn; Redmi Note 5 Build/PKQ1.180904.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/71.0.3578.141 Mobile Safari/537.36 XiaoMi/MiuiBrowser/11.10.8',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Referer': 'https://www.douyin.com/',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
            }
            
            response = self.session.get(api_url, headers=headers)
            
            print(f"HTTP狀態碼: {response.status_code}")
            print(f"響應頭: {dict(response.headers)}")
            print(f"響應內容前500字符: {response.text[:500]}")
            
            if response.status_code == 200:
                if response.text.strip():
                    try:
                        data = response.json()
                        if 'aweme_list' in data:
                            print(f"成功獲取 {len(data['aweme_list'])} 個視頻")
                            return data['aweme_list']
                        else:
                            print("響應中未找到aweme_list")
                            print(f"可用的鍵: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                            return None
                    except json.JSONDecodeError as e:
                        print(f"JSON解析失敗: {e}")
                        print(f"響應內容: {response.text}")
                        return None
                else:
                    print("響應為空")
                    return None
            else:
                print(f"請求失敗，狀態碼: {response.status_code}")
                print(f"響應內容: {response.text}")
                return None
                
        except Exception as e:
            print(f"獲取視頻列表失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_video(self, video_url, filename):
        """下載視頻"""
        try:
            print(f"正在下載: {filename}")
            
            # 設置下載用的請求頭
            headers = {
                'User-Agent': 'Mozilla/5.0 (Linux; U; Android 9; zh-cn; Redmi Note 5 Build/PKQ1.180904.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/71.0.3578.141 Mobile Safari/537.36 XiaoMi/MiuiBrowser/11.10.8',
                'Referer': 'https://www.douyin.com/',
            }
            
            response = requests.get(video_url, headers=headers, stream=True)
            file_size = 0
            if response.status_code == 200:
                filepath = os.path.join(self.download_folder, filename)
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            file_size+=len(chunk)
                
                print(f"✓ 下載完成: {filename}")
                return file_size
            else:
                print(f"✗ 下載失敗: {filename}, 狀態碼: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ 下載失敗: {filename}, 錯誤: {e}")
            return False
    def extract_sec_user_id(self, user_url):
        """從用戶頁面URL提取sec_user_id"""
        try:
            # 使用正則表達式提取sec_user_id
            match = re.search(r'/user/([^?]+)', user_url)
            if match:
                return match.group(1)
            return None
        except Exception as e:
            print(f"提取sec_user_id失敗: {e}")
            return None

    def get_video_page_urls(self, page: ChromiumPage):
        """從當前頁面提取所有影片頁面網址"""
        container_xpath = "/html/body/div[2]/div[1]/div[4]/div[2]/div/div/div/div[3]/div/div/div[2]/div/div[2]"
        links = page.eles(f"xpath:{container_xpath}//a[@href]")
        urls = []
        for a in links:
            href = a.attr("href")
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.douyin.com/" + href.lstrip("/")
            if href not in urls:
                urls.append(href)
        return urls

    def fetch_mp4_from_page(self, page: ChromiumPage, video_page_url: str):
        """在影片頁面監聽並返回所有 mp4 連結"""
        page.listen.start()
        page.get(video_page_url)
        time.sleep(5)

        # 收集所有抓到的封包
        packets = list(page.listen.steps(timeout=5))
        page.listen.stop()

        mp4_urls = []
        for packet in packets:
            url = getattr(packet, "url", "")
            if not url:
                continue

            # 先依據 header 中的 content-type 判斷
            ctype = ""
            try:
                if packet.response and packet.response.headers:
                    ctype = packet.response.headers.get("Content-Type", "")
            except Exception:
                pass

            if "video/mp4" in ctype or "video" in url.lower() and '.js' not in url.lower() and '.css' not in url.lower():
                mp4_urls.append(url)

        # 去除重複並保持順序
        unique_urls = list(dict.fromkeys(mp4_urls))
        return unique_urls
    def get_video_page_urls(self, page: ChromiumPage):
        """從當前頁面提取所有影片頁面網址"""
        container_xpath = "/html/body/div[2]/div[1]/div[4]/div[2]/div/div/div/div[3]/div/div/div[2]/div/div[2]"
        links = page.eles(f"xpath:{container_xpath}//a[@href]")
        urls = []
        for a in links:
            href = a.attr("href")
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.douyin.com/" + href.lstrip("/")
            if href not in urls:
                urls.append(href)
        return urls


    def run(self, user_url, api_url=None):
        """主要運行函數"""
        print("=== 抖音視頻下載器 ===")
        print(f"用戶頁面: {user_url}")

        user_data_dir = None
        page = None
        try:
            co = ChromiumOptions()
            chrome_path = '/usr/bin/google-chrome'
            co.set_browser_path(chrome_path)
            co.set_argument('--no-sandbox')
            co.set_argument('--disable-dev-shm-usage')
            co.set_argument('--disable-web-security')
            co.set_argument('--disable-features=VizDisplayCompositor')
            co.set_argument('--disable-extensions')
            co.set_argument('--disable-plugins')
            co.set_argument('--disable-images')
            co.set_argument('--disable-setuid-sandbox')
            co.set_argument('--remote-debugging-address=0.0.0.0')
            co.set_argument('--headless=new')

            user_data_dir = tempfile.mkdtemp()
            co.set_user_data_path(user_data_dir)
            co.auto_port(9222)
            co.set_argument(f'--remote-debugging-port={9222}')

            print("正在啟動Chrome瀏覽器...")
            page = ChromiumPage(addr_or_opts=co)

            print("訪問用戶頁面...")
            page.get(user_url)
            time.sleep(5)

            login_panel = page.ele('#douyin-login-new-id')
            if login_panel:
                close_btn = page.ele('rect[fill="url(#pattern0_3645_22461)"]')
                if close_btn:
                    close_btn.click()
                    time.sleep(2)
            import tqdm
            for _ in tqdm.tqdm(range(5)):
                page.scroll.to_bottom()
                time.sleep(2)

            video_pages = self.get_video_page_urls(page)
            print(f"找到 {len(video_pages)} 個影片頁面")

            for vp in video_pages:
                print(f"打開: {vp}")
                mp4s = self.fetch_mp4_from_page(page, vp)
                if not mp4s:
                    print("未找到影片資源")
                    continue
                sizes = []
                for link in mp4s:
                    name = os.path.basename(urlparse(link).path)
                    if name == '' or name == None or not name: 
                        import random 
                        name = str(random.randint(100000000000000, 99999999999999999)) + '.mp4'
                    if not name.endswith('.mp4') or 'uuu' in name:
                        continue
                    size = self.download_video(link, name)
                    sizes.append((size, name))
                    print(f"size: {size}")
                    time.sleep(1)
                       
                sorted(sizes, key=lambda s: s[0]) # use size to sort
                print(f'biggest: {sizes[0]}')
                sizes.pop(0)
                for (size , name) in sizes:
                    if os.path.exists('douyin_videos2/'+ name):
                        os.remove(os.path.join('douyin_videos2', name))

                sizes = []
        
        except Exception as e:
            print(f"瀏覽器操作失敗: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if page:
                try:
                    page.quit()
                except Exception:
                    pass
            if user_data_dir:
                shutil.rmtree(user_data_dir, ignore_errors=True)                
def main():
    # 用戶URL
    # user_url = "https://www.douyin.com/user/MS4wLjABAAAAhGTvofJSpb_dRb51A_xGF5siEeiHB2ryBSRZ9V0NtM7C-UgZ9ACJLTO7HwEGnFSE?from_tab_name=main"
    # user_url = "https://www.douyin.com/user/MS4wLjABAAAA4UAJ57hn-vBHuN-OF1D5fv66HG7QSEC9KcGE5UKO0McCgah4U6hqVNPZZUpN7YsW?from_tab_name=main"
    # user_url = "https://www.douyin.com/user/MS4wLjABAAAASOZyC1TaCyf8fPZXxBiG67niXfFhkIZupGeJHE9S93Y?from_tab_name=main"
    # user_url = "https://www.douyin.com/user/MS4wLjABAAAADw1dDJd4zddv0m8KWQB7ztFV0Nt8QzIK7dpFvbsrXss?from_tab_name=main"
    # user_url = "https://www.douyin.com/user/MS4wLjABAAAAldMnLkHbNWaeczPl57exxGcU096_bdmDHxLaCh00gjpIpg3p6qKCjQcm5oHWlQ9u?from_tab_name=main"
    # user_url = 'https://www.douyin.com/user/MS4wLjABAAAAUrd5Hx-0bX1j7dHzfK8nyijvrbQMrKlAhcAdb8NWSfKmRCgSD7MwWIzY2BAEYLcN?from_tab_name=main'
    # user_url = 'https://www.douyin.com/user/MS4wLjABAAAAHz-Oepugg1914S5nJRfrnceErOymKZtwouCs1_RXCIs?from_tab_name=main'
    user_url = 'https://www.douyin.com/user/MS4wLjABAAAAt1NymUQV3H2ZmBNoXbsus8J1NkVcGc31JO1YTGvwJxM?from_tab_name=main'

    
    # 創建下載器實例
    downloader = DouyinVideoDownloader()
    
    # 運行下載器
    downloader.run(user_url)

if __name__ == "__main__":
    main()
