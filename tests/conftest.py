"""测试期自动加载 .env——⛔ 已存在的环境变量不覆盖。"""

from amb.core import load_dotenv

load_dotenv()
