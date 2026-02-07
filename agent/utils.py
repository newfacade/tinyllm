# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
from typing import List, Dict

# 加载项目根目录的本地机密文件 secrets.env（如果存在），不影响环境变量已有值
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SECRETS_ENV_PATH = os.path.join(PROJECT_ROOT, 'secrets.env')

def _load_env_file(path: str):
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip()
                    if k and v and (k not in os.environ):
                        os.environ[k] = v
    except Exception:
        # 若解析失败，忽略，不影响后续使用环境变量
        pass

_load_env_file(SECRETS_ENV_PATH)


def ask(messages: List[Dict]):
        #需要指定api key 以及base_url
        client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
                model = 'deepseek-chat',
                messages = messages,
                stream=False
        )
        return response
