# write.py - 处理论文生成相关功能
from flask import Flask, request, jsonify
from flask_cors import CORS
from decouple import config
import logging
import openai
import httpx
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置速率限制
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# 配置CORS
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=CORS_ORIGINS)

# 配置OpenAI
OPENAI_API_KEY = config('OPENAI_API_KEY')
OPENAI_API_BASE = config('OPENAI_API_BASE')

AVAILABLE_MODELS = {
    'claude-3-5-sonnet-20240620': {
        'name': 'Claude 3.5 Sonnet',
        'max_tokens': 4000,
        'temperature': 0.7
    },
    'gpt-4o-mini': {
        'name': 'GPT-4 Mini',
        'max_tokens': 2000,
        'temperature': 0.7
    }
}

@app.route('/api/generate-paper', methods=['POST'])
@limiter.limit("10 per minute")
def generate_paper():
    try:
        data = request.json
        topic = data.get('text', '')
        model_id = data.get('model', 'gpt-4o-mini')
        citations = data.get('citations', [])
        
        if not topic:
            return jsonify({'success': False, 'error': '请输入论文主题'}), 400
            
        if model_id not in AVAILABLE_MODELS:
            return jsonify({'success': False, 'error': '不支持的模型'}), 400

        prompt = f"""作为一个学术论文写作助手，请根据主题"{topic}"用地道的英文生成一篇有深度、有新意的学术论文。
要求：
1. 包含摘要、关键词、引言、主要内容和结论
2. 每个段落都需要引用参考文献
3. 使用学术性的语言
4. 确保引用的合理性和准确性

可用的参考文献：
{chr(10).join(f"{i+1}. {citation}" for i, citation in enumerate(citations))}

请生成论文内容，并在每次引用时使用(作者, 年份)的格式。"""

        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            http_client=httpx.Client(
                timeout=60.0,
                follow_redirects=True
            )
        )

        model_info = AVAILABLE_MODELS[model_id]
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是一个专业的学术论文写作助手，擅长生成具有严谨学术性的内容。"},
                {"role": "user", "content": prompt}
            ],
            temperature=model_info['temperature'],
            max_tokens=model_info['max_tokens']
        )
        
        generated_content = response.choices[0].message.content

        return jsonify({
            'success': True,
            'paper': generated_content
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用的模型列表"""
    return jsonify({
        'success': True,
        'models': [{
            'id': model_id,
            'name': model_info['name']
        } for model_id, model_info in AVAILABLE_MODELS.items()]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(port=5002)
