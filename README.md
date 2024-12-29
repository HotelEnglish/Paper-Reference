
# AI 论文写作助手

一个基于 AI 的学术论文写作辅助工具，能够自动搜索相关文献并生成包含引用的论文内容。

## 功能特点

- 自动搜索相关学术文献
- 支持多个文献数据源（Crossref、Semantic Scholar、Google Scholar）
- 使用 GPT 生成论文内容
- APA 第7版引用格式
- 实时文献引用

## 技术栈

- 前端：React, Material-UI
- 后端：Flask, Python
- API：OpenAI, Google Scholar, Semantic Scholar, Crossref

## 本地开发

1. 克隆仓库：
```bash
git clone https://github.com/your-username/paper-assistant.git
cd Paper-Reference
```

2. 设置后端：
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. 配置环境变量：
创建 backend/.env 文件并添加必要的 API 密钥

4. 启动后端：
```bash
python app.py
```

5. 设置前端：
```bash
cd frontend
npm install
npm start
```

6. 访问 http://localhost:3000
