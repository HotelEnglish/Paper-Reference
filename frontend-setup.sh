cd frontend
npx create-react-app .
rm -rf src/* public/*

# 创建必要的目录和文件
mkdir -p public src

# 安装依赖
npm install axios @mui/material @emotion/react @emotion/styled

# 创建 .gitignore
echo "node_modules/" >> .gitignore
echo "build/" >> .gitignore 