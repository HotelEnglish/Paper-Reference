from flask import Flask, request, jsonify
from flask_cors import CORS
from decouple import config
import requests
from serpapi import GoogleSearch
import json
from datetime import datetime
import openai
from typing import List, Dict
from scholarly import scholarly
import semanticscholar as sch
from habanero import Crossref
import logging
import httpx
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

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

# 配置API密钥和代理
GOOGLE_SCHOLAR_API_KEY = config('GOOGLE_SCHOLAR_API_KEY')
SERPAPI_API_KEY = config('SERPAPI_API_KEY', default='')
PROXY_URL = config('PROXY_URL', default='http://api.wlai.vip')
OPENAI_API_KEY = config('OPENAI_API_KEY')
OPENAI_API_BASE = config('OPENAI_API_BASE')

# 配置OpenAI
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE

AVAILABLE_MODELS = {
    'claude-3-5-sonnet-20240620': {
        'name': 'Claude 3.5 Sonnet',
        'max_tokens': 4000,
        'temperature': 0.7
    },
    'gemini-2.0-flash-exp': {
        'name': 'Gemini 2.0 Flash',
        'max_tokens': 2048,
        'temperature': 0.7
    },
    'gpt-4o-mini': {
        'name': 'GPT-4 Mini',
        'max_tokens': 2000,
        'temperature': 0.7
    },
    'gpt-4o': {
        'name': 'GPT-4',
        'max_tokens': 4000,
        'temperature': 0.7
    },
    'o1-mini': {
        'name': 'O1 Mini',
        'max_tokens': 2000,
        'temperature': 0.7
    }
}

@app.route('/')
def index():
    """根路由，返回API状态信息"""
    return jsonify({
        'status': 'running',
        'version': '1.0',
        'endpoints': {
            'health_check': '/api/health',
            'generate_paper': '/api/generate-paper'
        }
    })

def search_crossref(query: str, max_results: int = 5) -> List[Dict]:
    """使用Crossref API搜索文献"""
    try:
        logger.info(f"Searching Crossref for: {query}")
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
            "select": "DOI,title,author,published-print,container-title",
            "mailto": "your-email@example.com"  # 添加邮箱以获得更好的服务
        }
        
        response = requests.get(url, params=params)
        papers = []
        
        if response.status_code == 200:
            data = response.json()
            for work in data['message']['items']:
                if work.get('title') and work.get('author'):
                    paper_info = {
                        'title': work['title'][0],
                        'authors': [
                            f"{author.get('given', '')} {author.get('family', '')}"
                            for author in work.get('author', [])
                        ],
                        'year': work.get('published-print', {}).get('date-parts', [['']])[0][0],
                        'journal': work.get('container-title', [''])[0] if work.get('container-title') else '',
                        'doi': work.get('DOI', ''),
                        'source': 'Crossref'
                    }
                    papers.append(paper_info)
                    
        logger.info(f"Found {len(papers)} papers from Crossref")
        return papers
    except Exception as e:
        logger.error(f"Crossref search error: {str(e)}")
        return []

def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict]:
    """使用Semantic Scholar API搜索文献"""
    try:
        logger.info(f"Searching Semantic Scholar for: {query}")
        # 使用 requests 直接调用 API
        url = f"https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,journal,venue,publicationVenue,citationCount,openAccessPdf"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            papers = []
            
            for paper in data.get('data', []):
                paper_info = {
                    'title': paper.get('title', ''),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'year': paper.get('year'),
                    'journal': (paper.get('publicationVenue', {}) or {}).get('name', ''),
                    'doi': paper.get('doi', ''),
                    'source': 'Semantic Scholar'
                }
                papers.append(paper_info)
            
            logger.info(f"Found {len(papers)} papers from Semantic Scholar")
            return papers
        else:
            logger.error(f"Semantic Scholar API error: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Semantic Scholar search error: {str(e)}")
        return []

def search_google_scholar(query: str, max_results: int = 5) -> List[Dict]:
    """使用Google Scholar API搜索文献"""
    try:
        logger.info(f"Searching Google Scholar for: {query}")
        if not SERPAPI_API_KEY:
            logger.warning("SERPAPI_API_KEY not configured")
            return []
            
        search_params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": max_results,
            "as_ylo": 2000  # 限制从2000年开始的文献
        }
        
        if PROXY_URL:
            search_params["proxy"] = PROXY_URL
        
        search = GoogleSearch(search_params)
        results = search.get_dict()
        papers = []
        
        if 'organic_results' in results:
            for result in results['organic_results']:
                paper_info = {
                    'title': result.get('title', ''),
                    'authors': [author.strip() for author in result.get('authors', '').split(',') if author.strip()],
                    'year': result.get('year', ''),
                    'journal': result.get('publication', ''),
                    'source': 'Google Scholar',
                    'link': result.get('link', '')
                }
                if paper_info['authors'] and paper_info['title']:  # 只添加有作者和标题的结果
                    papers.append(paper_info)
                    
        logger.info(f"Found {len(papers)} papers from Google Scholar")
        return papers
    except Exception as e:
        logger.error(f"Google Scholar search error: {str(e)}")
        return []

def format_apa7_citation(paper: Dict) -> str:
    """将论文信息转换为APA第7版格式"""
    authors = paper.get('authors', [])
    if len(authors) > 1:
        author_text = f"{', '.join(authors[:-1])}, & {authors[-1]}"
    else:
        author_text = authors[0] if authors else "Unknown"
    
    year = paper.get('year', 'n.d.')
    title = paper.get('title', '')
    journal = paper.get('journal', '')
    doi = paper.get('doi', '')
    source = paper.get('source', '')
    
    citation = f"{author_text}. ({year}). {title}. "
    
    if journal:
        citation += f"{journal}. "
    if doi:
        citation += f"https://doi.org/{doi}"
    
    citation += f" [{source}]"
    
    return citation

def search_all_sources(query: str) -> Dict[str, List[Dict]]:
    """从所有来源搜索文献"""
    results = {
        'crossref': search_crossref(query),
        'semantic_scholar': search_semantic_scholar(query),
        'google_scholar': search_google_scholar(query)
    }
    
    # 为每个结果添加引用格式
    for source, papers in results.items():
        for paper in papers:
            paper['citation'] = format_apa7_citation(paper)
    
    return results

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

@app.route('/api/generate-paper', methods=['POST'])
@limiter.limit("10 per minute")  # 添加速率限制
def generate_paper():
    try:
        data = request.json
        topic = data.get('text', '')
        model_id = data.get('model', 'gpt-4o-mini')  # 默认模型
        
        logger.info(f"Received paper generation request for topic: {topic} using model: {model_id}")
        
        if not topic:
            return jsonify({'success': False, 'error': '请输入论文主题'}), 400
            
        if model_id not in AVAILABLE_MODELS:
            return jsonify({'success': False, 'error': '不支持的模型'}), 400

        # 从所有来源搜索文献
        all_results = search_all_sources(topic)
        
        # 检查是否找到任何文献
        total_results = sum(len(papers) for papers in all_results.values())
        logger.info(f"Found total {total_results} papers across all sources")
        
        if total_results == 0:
            return jsonify({
                'success': False, 
                'error': '未找到相关文献，请尝试使用不同的关键词或确保API配置正确',
                'debug_info': {
                    'topic': topic,
                    'api_status': {
                        'serpapi': bool(SERPAPI_API_KEY),
                        'openai': bool(OPENAI_API_KEY)
                    }
                }
            }), 400
        
        # 准备用于生成论文的引用
        all_citations = []
        for source_papers in all_results.values():
            all_citations.extend([paper['citation'] for paper in source_papers])
        
        # 生成论文内容
        prompt = f"""作为一个学术论文写作助手，请根据主题"{topic}"用地道的英文生成一篇有深度、有新意的学术论文。
要求：
1. 包含摘要、关键词、引言、主要内容和结论
2. 每个段落都需要引用参考文献
3. 使用学术性的语言
4. 确保引用的合理性和准确性
5. 分别使用来自不同数据源的文献
6. 使用国际顶级期刊的水准和规范

可用的参考文献：
{chr(10).join(f"{i+1}. {citation}" for i, citation in enumerate(all_citations))}

请生成论文内容，并在每次引用时使用(作者, 年份)的格式。"""

        # 使用新版 OpenAI API，移除代理设置
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            http_client=httpx.Client(
                timeout=60.0,
                follow_redirects=True
            )
        )

        model_info = AVAILABLE_MODELS[model_id]
        
        try:
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
            
        except Exception as api_error:
            logger.error(f"OpenAI API error: {str(api_error)}")
            return jsonify({
                'success': False,
                'error': f'AI模型调用失败: {str(api_error)}'
            }), 500

        return jsonify({
            'success': True,
            'paper': generated_content,
            'references': all_citations,
            'search_results': {
                source: [{'title': p['title'], 'citation': p['citation']} for p in papers]
                for source, papers in all_results.items()
            }
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True) 