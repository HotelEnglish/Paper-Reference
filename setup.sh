# 创建项目目录
mkdir paper-assistant
cd paper-assistant
mkdir backend frontend

# 设置后端
cd backend
python -m venv venv

# Windows激活虚拟环境
.\venv\Scripts\activate
# Linux/Mac激活虚拟环境
# source venv/bin/activate

# 安装后端依赖
pip install -r requirements.txt

# 创建 .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore

# 返回项目根目录
cd ..

# 设置前端
cd frontend
npx create-react-app .
npm install axios @mui/material @emotion/react @emotion/styled

# 创建 .gitignore
echo "node_modules/" >> .gitignore
echo "build/" >> .gitignore 